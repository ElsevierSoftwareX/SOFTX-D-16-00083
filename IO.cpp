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

//Input and output functions here.
#include "typedefs.h"
void calcRotMatrix(vect3d <double> *cf0, vect3d <double> *cf1, vect3d <double> *cf2, vect3d <double> euler, unsigned int convention);
int checkFile(const char* filename) {
	ifstream in(filename);
	if (in.is_open()) {
		in.close();
		return 0;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}
int findSpace(string line){
	int pos=0;
	int posS=(int)line.find(' ');
	int posT=(int)line.find('\t');
	if ((posS>0)&&(posT>0))	pos=MIN(posS,posT);
	if (posT<0) pos=posS;
	if (posS<0) pos=posT;
	return pos;
}
int block::printCopies(const char * filename){
	ofstream out;
	out.open(filename);
	if (out.is_open()){
		for (unsigned int iCopy=0;iCopy<Ncopy;iCopy++)	out << rCopy[iCopy] << " " << euler[iCopy] << "/n";
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << filename << "for writing." << endl;
	return -1;
}
int printPDF(const double *PDF, unsigned int Nhist, double hist_bin, string name, unsigned int source, unsigned int PDFtype) {
	ostringstream outname;
	string PDFtype_str;
	if (PDFtype == typeRDF) PDFtype_str = "RDF";
	else if (PDFtype == typePDF) PDFtype_str = "PDF";
	else if (PDFtype == typeRPDF) PDFtype_str = "rPDF";
	if (source == xray) outname << name << "_xray_" << PDFtype_str << ".txt";
	else outname << name << "_neut_" << PDFtype_str << ".txt";
	ofstream out;
	out.open(outname.str().c_str());
	if (out.is_open()){
		unsigned int NhistLast;
		for (NhistLast = Nhist - 1; NhistLast > 0; NhistLast--) {
			if (ABS(PDF[NhistLast])>1.e-10) break;
		}
		out.setf(ios::scientific);
		for (unsigned int i = 0; i < NhistLast + 1; i++)	out << (i+0.5)*hist_bin << "	" << PDF[i] << "\n";
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}
int printPartialPDF(const double *PDF, unsigned int Nhist, double hist_bin, string name, unsigned int source, unsigned int PDFtype, unsigned int Nel, map <string, unsigned int> ID) {
	unsigned int count = 0;
	int error = 0;
	string PDFtype_str;
	if (PDFtype == typeRDF) PDFtype_str = "RDF";
	else if (PDFtype == typePDF) PDFtype_str = "PDF";
	else if (PDFtype == typeRPDF) PDFtype_str = "rPDF";
	map <string, unsigned int>::iterator it = ID.begin();
	for (unsigned int iEl = 0; iEl < Nel; iEl++) {
		map <string, unsigned int>::iterator jt = it;
		for (unsigned int jEl = iEl; jEl < Nel; jEl++, count += Nhist) {
			ostringstream outname;
			if (source == xray) outname << name << "_xray_partial_" << PDFtype_str << "_" << it->first << "_" << jt->first << ".txt";
			else outname << name << "_neut_partial_" << PDFtype_str << "_" << it->first << "_" << jt->first << ".txt";
			ofstream out;
			out.open(outname.str().c_str());
			if (out.is_open()){
				unsigned int NhistLast;
				for (NhistLast = Nhist-1; NhistLast > 0; NhistLast--) {
					if (PDF[count + NhistLast]>1.e-10) break;
				}
				out.setf(ios::scientific);
				for (unsigned int i = 0; i < NhistLast+1; i++)	out << (i+0.5)*hist_bin << "	" << PDF[count + i] << "\n";
				out.close();
			}
			else {
				error -= 1;
				cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
			}
			jt++;
		}
		it++;
	}
	return error;
}
int printI(const double *I,unsigned int Nq,const double *q, string name, unsigned int source){
	//Printing 1D intensity profile
	ostringstream outname;
	if (source == xray) outname << name << "_xray_1D.txt";
	else outname << name << "_neut_1D.txt";
	ofstream out;
	out.open(outname.str().c_str());
	if (out.is_open()){
		out.setf(ios::scientific);
		for (unsigned int iq = 0; iq<Nq; iq++)	out << q[iq] << "	" << I[iq] << "\n";
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}
int printPartialI(const double *I, unsigned int Nq, const double *q, string name, unsigned int source, unsigned int Nblocks){
	unsigned int count = 0;
	int error = 0;
	for (unsigned int iB = 0; iB < Nblocks; iB++) {
		for (unsigned int jB = iB; jB < Nblocks; jB++, count += Nq) {
			ostringstream outname;
			if (source == xray) outname << name << "_xray_" << iB << "-" << jB << "_1D.txt";
			else outname << name << "_neut_" << iB << "-" << jB << "_1D.txt";
			ofstream out;
			out.open(outname.str().c_str());
			if (out.is_open()){
				out.setf(ios::scientific);
				for (unsigned int iq = 0; iq < Nq; iq++)	out << q[iq] << "	" << I[count + iq] << "\n";
				out.close();
			}
			else {
				error -= 1;
				cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
			}
		}
	}
	return error;
}
int printI2(const double * const *I, unsigned int Nq, unsigned int Nfi, string name, unsigned int source){
	//Printing 2D intensity in polar coordinates
	//raws - q; columns - fi
	ostringstream outname;
	if (source == xray) outname << name << "_xray_2D.txt";
	else outname << name << "_neut_2D.txt";
	ofstream out;
	out.open(outname.str().c_str());
	if (out.is_open()){
		out.setf(ios::scientific);
		for (unsigned int iq=0;iq<Nq;iq++){
			for (unsigned int ifi=0;ifi<Nfi;ifi++)	out << I[iq][ifi] << "	";
			out << "\n";
		}
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}
int block::readAtoms(string elements[]){
	//Reading atomic numbers and atoms coordinates from file
	//string deforder="name r Uiso occ";
	ifstream in(CellFile.c_str());
	int iAtom=0;
	if (in.is_open()){
		string line;
		while(getline(in,line)){
			if ((!line.empty())&&(line[0]!='#')){
				istringstream streamline(line);				
				string tmp;
				int Ztmp = 0;
				double Uiso = 0;
				streamline >> tmp;
				streamline >> rAtom[iAtom];
				if (streamline.fail()) continue;
				if (streamline.good()) streamline >> Uiso;
				if (Uiso) dev[iAtom] = sqrt(Uiso);
				if (streamline.good()) streamline >> occ[iAtom];
				Ztmp = atoi(tmp.c_str());
				(Ztmp) ? name[iAtom] = elements[Ztmp] : name[iAtom] = tmp;
				iAtom++;				
			}
		}
		in.close();
		return 0;
	}
	cout << "Error: cannot open file " << CellFile << endl;
	return -1;
}
int block::readRearrangement(string elements[]){
	ifstream in(NeighbFile.c_str());
	vect3d <double> value;
	int ZN,iAtom,NeighbMax;
	if (in.is_open()){
		string line;
		while(getline(in,line)){
			if ((!line.empty()) && (line[0] != '#')){
				istringstream streamline(line);
				streamline >> iAtom >> NeighbMax;
				int iNeighb=0;
				while(iNeighb<NeighbMax){
					getline(in,line);
					if ((!line.empty()) && (line[0] != '#')){
						istringstream streamline(line);
						string tmp;
						streamline >> tmp;						
						ZN=atoi(tmp.c_str());
						(ZN) ? nameNeighb[iAtom].push_back(elements[ZN]) : nameNeighb[iAtom].push_back(tmp);
						streamline >> value;
						rAtomNeighb[iAtom].push_back(value);
						iNeighb++;
					}
				}
			}
		}
		return 0;
	}
	cout << "Error: cannot open file " << NeighbFile << endl;
	return -1;
}
int getN(const char*filename){
	//Calculating lines in file.
	unsigned int N=0;
	ifstream in(filename);
	if (in.is_open()){
		string line;
		while (getline(in, line)){
			if ((!line.empty()) && (line[0] != '#')){
				istringstream streamline(line);
				string tmp;
				vect3d <double> rtmp;
				streamline >> tmp;
				streamline >> rtmp;
				if (!streamline.fail()) N++;
			}
		}
		in.close();
		return N;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}
int block::readCopies(){
	//Reading coordinates and orientations for copies of an object.
	ifstream in(CopiesFile.c_str());
	if (in.is_open()){
		int iCopy=0;
		vect3d <double> cosEul,sinEul;
		string line;
		while(getline(in,line)){
			if ((!line.empty()) && (line[0] != '#')){
				istringstream streamline(line);
				streamline >> rCopy[iCopy] >> euler[iCopy];
				euler[iCopy]=euler[iCopy]/180.f*PI;
				calcRotMatrix(&cf[0][iCopy], &cf[1][iCopy], &cf[2][iCopy], euler[iCopy], EulerConvention);
				iCopy++;
			}
		}
		in.close();
		return 0;
	}
	cout << "Error: cannot open file " << CopiesFile << endl;
	return -1;
};
int printAtoms (const vector < vect3d<double> > *ra,string name,map <string, unsigned int> ID, unsigned int Ntot) {
	ofstream out;
	ostringstream outname;	
	outname << name << "_atoms.xyz";
	out.open(outname.str().c_str());
	if (out.is_open()){
		out << Ntot << "\n\n";
		out.setf(ios::scientific);
		string TypeName;
		map <string, unsigned int>::iterator it = ID.begin();
		for (unsigned int iEl=0;iEl<ID.size();iEl++){			
			TypeName = it->first;
			for (vector<vect3d <double> >::const_iterator ri = ra[iEl].begin(); ri != ra[iEl].end(); ri++){
				out << TypeName << " " << *ri << "\n";
			}
			it++;
		}
		out.close();
		return 0;
	}
	cout << "Error: cannot open file " << outname.str().c_str() << " for writing." << endl;
	return -1;
}
int readNeuData(const char *filename, map<string, unsigned int> *ID,vector<double> *SL,const vector <string> *names){
	ifstream in(filename);
	if (in.is_open()){
		string line,elstr,otherstr;
		unsigned int id=0;
		double iSL;
		while(getline(in,line)){
			if ((!line.empty()) && (line[0] != '#')){
				int pos=findSpace(line);
				elstr=line.substr(0,pos);
				if (find(names->begin(), names->end(), elstr) != names->end()){
					otherstr = line.substr(pos + 1);
					istringstream streamline(otherstr);
					(*ID)[elstr] = id;
					streamline >> iSL;
					SL->push_back(iSL);
					id++;
				}
			}
		}
		in.close();
		return 0;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}
int loadFF(vector <double *> *FF, unsigned int Nq,const double *q ,const char* filename, map<string, unsigned int> *ID,const vector <string> *names){
	//Loading needed form factor profiles from table and lineary interpolating them.
	ifstream in(filename);
	if (in.is_open()){
		//Loading
		vector <vector <double> > ff;
		vector <double> qff;
		unsigned int pid=0, iFFstart=0;
		double a,b,dq,value;
		string line,elstr,otherstr;
		while (getline(in,line)){
			if ((!line.empty()) && (line[0] != '#')){
				int pos=findSpace(line);
				elstr=line.substr(0,pos);
				otherstr = line.substr(pos + 1);
				istringstream streamline(otherstr);
				if (elstr == "q") {
					while (streamline >> value)	qff.push_back(value);
				}
				else if (find(names->begin(), names->end(), elstr) != names->end()){
					vector <double> tempvec;
					double *temp;
					temp = new double[Nq];
					FF->push_back(temp);
					(*ID)[elstr] = pid;
					while (streamline >> value)	tempvec.push_back(value);
					ff.push_back(tempvec);
					pid++;
				}
			}
		}
		in.close();
		if (FF->size() < names->size()) {
			cout << "Error: not enough form factors" << endl;
			return -1;
		}
		//Interpolating
		for (unsigned int iq = 0; iq < Nq; iq++) {
			for (unsigned int iff = iFFstart; iff < qff.size() - 1; iff++) {
				if ((q[iq] >= qff[iff]) && (q[iq] <= qff[iff + 1])){
					dq = qff[iff] - qff[iff + 1];
					for (unsigned int i = 0; i < ff.size(); i++){
						a = (ff[i][iff] - ff[i][iff + 1]) / dq;
						b = (ff[i][iff + 1] * qff[iff] - ff[i][iff] * qff[iff + 1]) / dq;
						(*FF)[i][iq] = (a*q[iq] + b);
					}
					iFFstart = iff;
					break;
				}
			}
		}			
		return 0;
	}
	cout << "Error: cannot open file " << filename << endl;
	return -1;
}
