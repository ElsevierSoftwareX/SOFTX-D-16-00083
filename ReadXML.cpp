//Copyright (C) 2015, NRC "Kurchatov institute", http://www.nrcki.ru/e/engl.html, Moscow, Russia
//Author: Vladislav Neverov, vs-never@hotmail.com, neverov_vs@nrcki.ru
//
//This file is part of XaNSoNS.
//
//XaNSoNS is free software: you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//XaNSoNS is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program. If not, see <http://www.gnu.org/licenses/>.

//functions, that read parameters from xml file are here
#include "tinyxml2.h"
#include "typedefs.h"
#ifdef UseMPI
#include "mpi.h"
#endif
extern int myid;
using namespace tinyxml2;
int getN(const char*);
int readNeuData(const char *filename, map<string, unsigned int> *ID, vector<double> *SL,const vector <string> *names);
int loadFF(vector <double *> *FF, unsigned int Nq,const double *q ,const char* filename, map<string, unsigned int> *ID,const vector <string> *names);
void calcRotMatrix(vect3d <double> *cf0, vect3d <double> *cf1, vect3d <double> *cf2, vect3d <double> euler, unsigned int convention);
int setSLforPDF(vector <double *> FF, unsigned int Nq, const double *q, vector<double> *SL, double qPDF) {
	double a, b, dq;
	for (unsigned int iq = 0; iq < Nq - 1; iq++) {
		if ((qPDF >= q[iq]) && (qPDF <= q[iq + 1])){
			dq = q[iq] - q[iq + 1];
			for (vector<double *>::iterator ff = FF.begin(); ff != FF.end(); ff++) {
				a = ((*ff)[iq] - (*ff)[iq + 1]) / dq;
				b = ((*ff)[iq] * q[iq] - (*ff)[iq] * q[iq + 1]) / dq;
				SL->push_back(a*q[iq] + b);
			}
			break;
		}
	}
	return 0;
}
int checkFile(const char* );
unsigned int getNumberofElements(XMLElement* xMainNode,const char * name){
	unsigned int count = 0;
	XMLElement* blockNode=xMainNode->FirstChildElement(name);
	while(blockNode){
		blockNode = blockNode->NextSiblingElement(name);
		count++;
	}
	return count;
}
template <class X> int GetAttribute(XMLElement* xNode, const char * NodeName, const char *Attribute, X &obj,bool deflt=true, const char *defval="default value"){
	stringstream outs;
	const char* value=NULL;
	if (!xNode) {
		if (!deflt){
			if(!myid) cout << "Parsing error: " << NodeName << " node is missing." << endl;
			return -1;
		}
		if(!myid) cout << NodeName  << "-->"<< Attribute << " is set to "<< defval << "." << endl;
		return 0;
	}
	value = xNode->Attribute(Attribute);
	if (!value){
		if (!deflt){
			if(!myid) cout << "Parsing error:  Attribute "<< Attribute << " of " << NodeName << " node is missing." << endl;
			return -1;
		}
		if(!myid) cout << NodeName << "-->"<< Attribute << " is set to "<< defval << "." << endl;
		return 0;
	}
	outs << value;
	string tempstr(outs.str());
	if (!tempstr.length()){
		if (!deflt){
			if(!myid) cout << "Parsing error:  Attribute "<< Attribute << " of " << NodeName << " node is empty." << endl;
			return -1;
		}
		if(!myid) cout << NodeName << "-->" << Attribute << " is set to "<< defval << "." << endl;
		return 0;
	}
	outs >> obj;
	return 0;
}
template <class X> int GetWord(XMLElement* xNode, const char * NodeName, const char *Attribute, map <string, X> word, X &atr, bool deflt=true, const char *defval="default value"){
	int error=0;
	string tempstr;
	error=GetAttribute(xNode,NodeName,Attribute,tempstr,deflt,defval);
	if (error) return error;
	transform(tempstr.begin(), tempstr.end(),tempstr.begin(),::tolower);
	if (tempstr.length()){
		if (!word.count(tempstr)){
			if(!myid) cout << "Parsing error:  Attribute "<< Attribute << " of " << NodeName << " node has an unexpected value." << endl;
			return -2;
		}
		atr=word[tempstr];
	}
	return 0;
}
int GetCopiesFromXML(XMLElement* CopiesNode, block *Block){
	int error=0;
	XMLElement* copyNode = CopiesNode->FirstChildElement("Copy");
	for (unsigned int iCopy = 0; iCopy < Block->Ncopy; iCopy++) {
		error+=GetAttribute(copyNode,"Copy","r",Block->rCopy[iCopy],false);
		error+=GetAttribute(copyNode,"Copy","Euler",Block->euler[iCopy],false);
		Block->euler[iCopy]=Block->euler[iCopy]/180.*PI;
		calcRotMatrix(&Block->cf[0][iCopy], &Block->cf[1][iCopy], &Block->cf[2][iCopy], Block->euler[iCopy], Block->EulerConvention);
		copyNode = copyNode->NextSiblingElement("Copy");
	}
	return error;
}
int GetSymmFromXML(XMLElement* BlockNode, block *Block){
	int error=0;
	XMLElement* symmNode = BlockNode->FirstChildElement("SymmEqPos");
	for (unsigned int iSymm = 1; iSymm < Block->Nsymm; iSymm++) {
		error+=GetAttribute(symmNode,"SymmEqPos","r",Block->rSymm[iSymm],false);
		error+=GetAttribute(symmNode,"SymmEqPos","R1",Block->pSymm[0][iSymm],false);
		error+=GetAttribute(symmNode,"SymmEqPos","R2",Block->pSymm[1][iSymm],false);
		error+=GetAttribute(symmNode,"SymmEqPos","R3",Block->pSymm[2][iSymm],false);
		symmNode = symmNode->NextSiblingElement("SymmEqPos");
	}
	return error;
}
int GetAtomsFromXML(XMLElement* AtomsNode, block *Block, string elements[]){
	XMLElement* atomNode = AtomsNode->FirstChildElement("Atom");
	int error=0;
	for (unsigned int iAtom = 0; iAtom < Block->Natom; iAtom++) {
		unsigned int Z = 0;
		error+=GetAttribute(atomNode,"Atom","name",Block->name[iAtom],true,"table value according to it's Z");
		if (!Block->name[iAtom].length())	{
			error += GetAttribute(atomNode,"Atom", "Z", Z, false);
			if (error) {
				cout << "Parsing error: Neither Z or name of the atom with number" << iAtom << "is set." << endl;
				return error;
			}
			Block->name[iAtom] = elements[Z];
		}
		error+=GetAttribute(atomNode,"Atom","r",Block->rAtom[iAtom],false);
		error+=GetAttribute(atomNode,"Atom","occ",Block->occ[iAtom],true,"1.0");
		error+=GetAttribute(atomNode,"Atom","Uiso",Block->dev[iAtom],true,"0");
		Block->dev[iAtom]=sqrt(Block->dev[iAtom]);
		atomNode = atomNode->NextSiblingElement("Atom");
	}
	return error;
}
void SetDefault(config *cfg) {
	cfg->source = xray;
	cfg->scenario = Debye;
	cfg->Nblocks = 0;
	cfg->q.N = 1024;
	cfg->Nfi = 1024; 
	cfg->PDFtype = typePDF;
	cfg->PolarFactor = false;
	cfg->PrintAtoms = false;
	cfg->PrintPartialPDF = false;
	cfg->calcPartialIntensity = false;
	cfg->rearrangement = false;
	cfg->EulerConvention = EulerZXZ;
	cfg->lambda = -1.;
	cfg->hist_bin = 0.001;
	cfg->qPDF = 0;
	cfg->p0 = 0.;
	cfg->Euler.min.assign(0, 0, 0); cfg->Euler.max.assign(0, 0, 0); cfg->Euler.N.assign(1, 1, 1);
}
void GetAllnames(const block *Block, vector <string> *names, unsigned int Nblocks) {
	for (unsigned int iB = 0; iB < Nblocks; iB++) {
		for (unsigned int iAtom = 0; iAtom < Block[iB].Natom; iAtom++) {
			if (find(names->begin(), names->end(), Block[iB].name[iAtom]) == names->end()) names->push_back(Block[iB].name[iAtom]);
			if (Block[iB].rearrangement) {
				for (vector <string>::iterator iName = Block[iB].nameNeighb[iAtom].begin(); iName != Block[iB].nameNeighb[iAtom].end(); iName++) {
					if (find(names->begin(), names->end(), *iName) == names->end()) names->push_back(*iName);
				}
			}
		}
	}
}
int ReadConfig(config *cfg, double **q, block **Block, const char* FileName,string elements[],map<string, unsigned int> *ID,vector<double> *SL, vector<double *> *FF){
	int error=0;
	map<string, unsigned int> word;
	vector <string> names;
	map<string, bool> flag;
	word["xray"] = xray; word["neutron"] = neutron; word["debye"] = Debye; word["2d"] = s2D; word["debye_hist"] = Debye_hist; 
	word["pdfonly"] = PDFonly; word["debyepdf"] = DebyePDF; word["rdf"] = typeRDF; word["pdf"] = typePDF; word["rpdf"] = typeRPDF;
	word["xzx"] = EulerXZX; word["xyx"] = EulerXYX; word["yxy"] = EulerYXY; word["yzy"] = EulerYZY; word["zyz"] = EulerZYZ; word["zxz"] = EulerZXZ;
	word["xzy"] = EulerXZY; word["xyz"] = EulerXYZ; word["yxz"] = EulerYXZ; word["yzx"] = EulerYZX; word["zyx"] = EulerZYX; word["zxy"] = EulerZXY;
	flag["blocks"]=false; flag["yes"]=true; flag["no"]=false; flag["true"]=true; flag["false"]=false; flag["1"]=true; flag["0"]=false;
	XMLDocument doc;
	XMLError res = doc.LoadFile(FileName);
	if (res!=XML_SUCCESS)  {
		if (!myid) cout << "Error: file " << FileName << " does not exist or contains errors." << endl;
		return -1;
	}
	XMLElement* xMainNode=doc.RootElement();
	if (!myid) cout << "\nParsing calculation parameters..." << endl;
	XMLElement *calcNode = xMainNode->FirstChildElement ("Calculation");
	XMLElement *BlockNode = xMainNode->FirstChildElement ("Block");
	if (!calcNode) {
		if (!myid) cout << "Parsing error: Calculation node is missing." << endl;
		return -1;
	}
	SetDefault(cfg);
	error += GetAttribute(calcNode,"Calculation","name",cfg->name,false);
	error += GetWord(calcNode,"Calculation","source",word,cfg->source,true,"Xray");
	error += GetWord(calcNode,"Calculation","scenario",word,cfg->scenario,true,"Debye");
	error += GetWord(calcNode,"Calculation","PrintAtoms",flag,cfg->PrintAtoms,true,"No");
	error += GetAttribute(calcNode,"Calculation", "FFfilename", cfg->AIfilename, false);
	if (cfg->scenario != PDFonly) {
		error += GetWord(calcNode,"Calculation", "PolarFactor", flag, cfg->PolarFactor, true, "No");
		XMLElement *tempNode = calcNode->FirstChildElement ("q");
		error += GetAttribute(tempNode,"q", "min", cfg->q.min, false);
		error += GetAttribute(tempNode,"q", "max", cfg->q.max, false);
		error += GetAttribute(tempNode,"q", "N", cfg->q.N, true, "1024");
		if (cfg->PolarFactor) error += GetAttribute(calcNode,"Calculation", "wavelength", cfg->lambda, false);
		else error += GetAttribute(calcNode,"Calculation", "wavelength", cfg->lambda, true, "default");
		if (cfg->lambda < 0) cfg->lambda = 4.*PI / cfg->q.max;
		*q = new double[cfg->q.N];
		double deltaq = (cfg->q.max - cfg->q.min) / cfg->q.N;
		for (unsigned int iq = 0; iq<cfg->q.N; iq++) (*q)[iq] = cfg->q.min + iq*deltaq;
	}
	if (cfg->scenario == s2D) {
		XMLElement *tempNode = calcNode->FirstChildElement ("q");
		error += GetAttribute(tempNode,"q", "Nfi", cfg->Nfi, true, "1024");
		tempNode = calcNode->FirstChildElement ("Euler");
		error += GetWord(tempNode,"Euler", "convention",word, cfg->EulerConvention, true, "ZXZ");
		error += GetAttribute(tempNode,"Euler", "min", cfg->Euler.min, true, "0 0 0");
		error += GetAttribute(tempNode,"Euler", "max", cfg->Euler.max, true, "0 0 0");
		error += GetAttribute(tempNode,"Euler", "N", cfg->Euler.N, true, "1 1 1");
		cfg->Euler.max = cfg->Euler.max / 180.*PI;
		cfg->Euler.min = cfg->Euler.min / 180.*PI;
	}
	if (cfg->scenario == Debye) error += GetWord(calcNode,"Calculation", "PartialIntensity", flag, cfg->calcPartialIntensity, true, "No");
	if (cfg->scenario > Debye) error += GetAttribute(calcNode, "Calculation", "hist_bin", cfg->hist_bin, true, "0.001");
	if (cfg->scenario > Debye_hist) {
		XMLElement *tempNode = calcNode->FirstChildElement ("PDF");
		error += GetWord(tempNode,"PDF", "type", word, cfg->PDFtype, true, "PDF");
		if (cfg->source == xray) error += GetAttribute(tempNode,"PDF", "q", cfg->qPDF, true, "0");
		error += GetAttribute(tempNode,"PDF", "density", cfg->p0, true, "approximately calculated value");
		error += GetWord(tempNode,"PDF", "PrintPartial", flag, cfg->PrintPartialPDF, true, "No");
	}
	cfg->Nblocks = getNumberofElements(xMainNode,"Block");
	if (!cfg->Nblocks) {
		if (!myid) cout << "Parsing error: structural blocks are not specified." << endl;
		error -= 1;
	}
	if (cfg->Nblocks < 2) cfg->calcPartialIntensity = false;
	*Block = new block[cfg->Nblocks];
	if (!myid) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			cout << "\nParsing Block " << iB << "..." << endl;
			error += GetWord(BlockNode,"Block", "fractional", flag, (*Block)[iB].fractional, true, "No");
			error += GetWord(BlockNode,"Block", "centered", flag, (*Block)[iB].centered, true, "No");
			error += GetWord(BlockNode, "Block", "centeredAtoms", flag, (*Block)[iB].centeredAtoms, true, "No");
			error += GetWord(BlockNode,"Block", "mol_rotation", flag, (*Block)[iB].ro_mol, true, "No");
			error += GetAttribute(BlockNode, "Block", "mol_Uiso", (*Block)[iB].dev_mol, true, "0");
			(*Block)[iB].dev_mol = sqrt((*Block)[iB].dev_mol);
			XMLElement *tempNode = BlockNode->FirstChildElement ("Atoms");
			error += GetAttribute(tempNode,"Atoms", "filename", (*Block)[iB].CellFile);
			(*Block)[iB].CellFile.length() ? (*Block)[iB].Natom = getN((*Block)[iB].CellFile.c_str()) : (*Block)[iB].Natom = getNumberofElements(tempNode,"Atom");
			if (!(*Block)[iB].Natom){
				cout << "Error: cell info not specified." << endl;
				return error-1;
			}
			(*Block)[iB].Nsymm = getNumberofElements(BlockNode,"SymmEqPos") + 1;
			(*Block)[iB].Ncopy = 1;
			tempNode = BlockNode->FirstChildElement ("CutOff");
			if (tempNode){
				if (tempNode->Attribute("Rcut")) {					
					error += GetAttribute(tempNode,"CutOff", "Rcut", (*Block)[iB].Rcut, false);
					if ((*Block)[iB].Rcut>1.e-7) (*Block)[iB].cutoff = true;
				}
				if (tempNode->Attribute("RcutCopies")) {
					error += GetAttribute(tempNode,"CutOff", "RcutCopies", (*Block)[iB].RcutCopies, false);
					if ((*Block)[iB].RcutCopies>1.e-7) (*Block)[iB].cutoffcopies = true;
				}
			}
			tempNode = BlockNode->FirstChildElement ("Copies");
			if (tempNode){				
				error += GetAttribute(tempNode,"Copies", "filename", (*Block)[iB].CopiesFile);
				if ((*Block)[iB].CopiesFile.length()) error += checkFile((*Block)[iB].CopiesFile.c_str());
				(*Block)[iB].CopiesFile.length() ? (*Block)[iB].Ncopy = getN((*Block)[iB].CopiesFile.c_str()) : (*Block)[iB].Ncopy = getNumberofElements(tempNode,"Copy");
				error += GetWord(tempNode,"Copies", "convention", word, (*Block)[iB].EulerConvention, true, "ZYX");
			}
			tempNode = BlockNode->FirstChildElement ("Rearrangement");
			if (tempNode){
				(*Block)[iB].rearrangement = true;
				cfg->rearrangement = true;
				error += GetAttribute(tempNode,"Rearrangement", "filename", (*Block)[iB].NeighbFile, false);
				if ((*Block)[iB].NeighbFile.length()) error += checkFile((*Block)[iB].NeighbFile.c_str());
				error += GetAttribute(tempNode,"Rearrangement", "Rcorr", (*Block)[iB].Rcorr, false);
			}
			tempNode = BlockNode->FirstChildElement ("CellVectors");
			error += GetAttribute(tempNode,"CellVectors", "a", (*Block)[iB].e[0], true, "(1,0,0)");
			error += GetAttribute(tempNode,"CellVectors", "b", (*Block)[iB].e[1], true, "(0,1,0)");
			error += GetAttribute(tempNode,"CellVectors", "c", (*Block)[iB].e[2], true, "(0,0,1)");
			error += GetAttribute(tempNode, "CellVectors", "N", (*Block)[iB].Ncell, true, "(1,1,1)");
			(*Block)[iB].create();
			if ((*Block)[iB].CellFile.length()) (*Block)[iB].readAtoms(elements);
			else GetAtomsFromXML(BlockNode->FirstChildElement ("Atoms"), &(*Block)[iB], elements);
			if ((*Block)[iB].rearrangement) (*Block)[iB].readRearrangement(elements);
			if (BlockNode->FirstChildElement("Copies")){
				if ((*Block)[iB].CopiesFile.length()) (*Block)[iB].readCopies();
				else GetCopiesFromXML(BlockNode->FirstChildElement("Copies"), &(*Block)[iB]);
			}
			else {
				(*Block)[iB].euler[0].assign(0, 0, 0);
				(*Block)[iB].rCopy[0].assign(0, 0, 0);
				(*Block)[iB].cf[0][0].assign(1., 0, 0);
				(*Block)[iB].cf[1][0].assign(0, 1., 0);
				(*Block)[iB].cf[2][0].assign(0, 0, 1.);
			}
			(*Block)[iB].rSymm[0].assign(0, 0, 0);
			(*Block)[iB].pSymm[0][0].assign(1., 0, 0);
			(*Block)[iB].pSymm[1][0].assign(0, 1., 0);
			(*Block)[iB].pSymm[2][0].assign(0, 0, 1.);
			if (BlockNode->FirstChildElement("SymmEqPos")) GetSymmFromXML(BlockNode, &(*Block)[iB]);
			(*Block)[iB].redefSymm();
			if ((*Block)[iB].fractional) (*Block)[iB].redefAtoms();
			if ((*Block)[iB].centeredAtoms) (*Block)[iB].centerAtoms();
			if ((*Block)[iB].centered) 	(*Block)[iB].calcMean();
			BlockNode = BlockNode->NextSiblingElement("Block");
		}
		cout << "\nAll blocks have been parsed.\n" << endl;
		GetAllnames(*Block, &names, cfg->Nblocks);
		cfg->Nel = (unsigned int) names.size();
		if ((cfg->source == xray) && (cfg->scenario != PDFonly)) error += loadFF(FF, cfg->q.N, *q, cfg->AIfilename.c_str(), ID, &names);
		else error += readNeuData(cfg->AIfilename.c_str(), ID, SL, &names);
		if ((cfg->source == xray) && (cfg->scenario == DebyePDF)) setSLforPDF(*FF, cfg->q.N, *q, SL, cfg->qPDF);
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++) (*Block)[iB].sortAtoms(*ID);
	}	
#ifdef UseMPI
	MPI_Bcast(&cfg->Nel, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if ((cfg->source == xray) && (cfg->scenario != PDFonly)) {
		if (myid) {
			FF->resize(cfg->Nel);
			for (vector <double*>::iterator iFF = FF->begin(); iFF != FF->end(); iFF++) *iFF = new double [cfg->q.N];
		}
		for (vector <double*>::iterator iFF = FF->begin(); iFF != FF->end(); iFF++)	MPI_Bcast(*iFF, cfg->q.N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	if ((cfg->source == neutron) || (cfg->scenario > Debye_hist)) {
		if (myid) SL->resize(cfg->Nel);
		MPI_Bcast(&(*SL)[0], cfg->Nel, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	for (unsigned int iB = 0; iB<cfg->Nblocks; iB++){
		MPI_Bcast(&(*Block)[iB].Nid, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (myid) (*Block)[iB].id = new unsigned int[(*Block)[iB].Nid];
		MPI_Bcast((*Block)[iB].id, (*Block)[iB].Nid, MPI_INT, 0, MPI_COMM_WORLD);
	}
#endif
	return error;
}
#ifdef UseOCL
int GetOpenCLinfoFromInitDataXML(string *GPUtype, int *DeviceNUM){
	//This function implements compatibility with BOINC clients (version 7.0.12+) for OpenCL version of XaNSoNS
	XMLDocument doc;
	const char *FileName = "init_data.xml";
	XMLError res = doc.LoadFile(FileName);
	if (res != XML_SUCCESS)  return -1;
	XMLElement *GPUtypeElement = doc.RootElement()->FirstChildElement("gpu_type");
	*GPUtype=string(GPUtypeElement->GetText());
	XMLElement *GPUOpenCLindexElement = doc.RootElement()->FirstChildElement("gpu_opencl_dev_index");
	*DeviceNUM = atoi(GPUOpenCLindexElement->GetText());
	return 0;
}
#endif
