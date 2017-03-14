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

#include "typedefs.h"
#include <random>
#include <chrono>
#ifdef UseOMP
#include <omp.h>
#endif
#ifdef UseMPI
#include "mpi.h"
#endif
extern int myid,numprocs;
int printAtoms(const vector < vect3d<double> > *ra, string name, map <string, unsigned int> ID, unsigned int Ntot);
vect3d <double> cos(vect3d <double> t){ //cosine for vect3d <double>
	vect3d <double> temp(cos(t.x), cos(t.y), cos(t.z));
	return temp;
}
vect3d <float> cos(vect3d <float> t){ //cosine for vect3d <float>
	vect3d <float> temp(cosf(t.x), cosf(t.y), cosf(t.z));
	return temp;
}
vect3d <double> sin(vect3d <double> t){ //sine for vect3d <double>
	vect3d <double> temp(sin(t.x), sin(t.y), sin(t.z));
	return temp;
}
vect3d <float> sin(vect3d <float> t){ //sine for vect3d <float>
	vect3d <float> temp(sinf(t.x), sinf(t.y), sinf(t.z));
	return temp;
}
void calcRotMatrix(vect3d <double> *cf0, vect3d <double> *cf1, vect3d <double> *cf2, vect3d <double> euler, unsigned int convention) {
	//Calculates rotational matrix by euler angles in the case of intristic rotation and in the user-defined convention 
	//see http://en.wikipedia.org/wiki/Euler_angles#Relationship_to_other_representations for details
	//cf0, cf1 and cf2 are for the first, second and third row respectively 
	vect3d <double> cosEul, sinEul;
	cosEul = cos(euler);
	sinEul = sin(euler);
	switch (convention){
		case EulerXZX: //XZX
			cf0->assign(cosEul.y,-cosEul.z*sinEul.y,sinEul.y*sinEul.z);
			cf1->assign(cosEul.x*sinEul.y,cosEul.x*cosEul.y*cosEul.z-sinEul.x*sinEul.z,-cosEul.z*sinEul.x-cosEul.x*cosEul.y*sinEul.z);
			cf2->assign(sinEul.x*sinEul.y,cosEul.x*sinEul.z+cosEul.y*cosEul.z*sinEul.x,cosEul.x*cosEul.z-cosEul.y*sinEul.x*sinEul.z);
			break;
		case EulerXYX: //XYX
			cf0->assign(cosEul.y, sinEul.y*sinEul.z, cosEul.z*sinEul.y);
			cf1->assign(sinEul.x*sinEul.y, cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, -cosEul.x*sinEul.z - cosEul.y*cosEul.z*sinEul.x);
			cf2->assign(-cosEul.x*sinEul.y, cosEul.z*sinEul.x + cosEul.x*cosEul.y*sinEul.z, cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z);
			break;
		case EulerYXY: //YXY
			cf0->assign(cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, sinEul.x*sinEul.y, cosEul.x*sinEul.z + cosEul.y*cosEul.z*sinEul.x);
			cf1->assign(sinEul.y*sinEul.z, cosEul.y, -cosEul.z*sinEul.y);
			cf2->assign(-cosEul.z*sinEul.x - cosEul.x*cosEul.y*sinEul.z, cosEul.x*sinEul.y, cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z);
			break;
		case EulerYZY: //YZY
			cf0->assign(cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z, -cosEul.x*sinEul.y, cosEul.z*sinEul.x + cosEul.x*cosEul.y*sinEul.z);
			cf1->assign(cosEul.z*sinEul.y, cosEul.y, sinEul.y*sinEul.z);
			cf2->assign(-cosEul.x*sinEul.z - cosEul.y*cosEul.z*sinEul.x, sinEul.x*sinEul.y, cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z);
			break;
		case EulerZYZ: //ZYZ
			cf0->assign(cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z, -cosEul.z*sinEul.x - cosEul.x*cosEul.y*sinEul.z, cosEul.x*sinEul.y);
			cf1->assign(cosEul.x*sinEul.z + cosEul.y*cosEul.z*sinEul.x, cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, sinEul.x*sinEul.y);
			cf2->assign(-cosEul.z*sinEul.y, sinEul.y*sinEul.z, cosEul.y);
			break;
		case EulerZXZ: //ZXZ
			cf0->assign(cosEul.x*cosEul.z - cosEul.y*sinEul.x*sinEul.z, -cosEul.x*sinEul.z - cosEul.y*cosEul.z*sinEul.x, sinEul.x*sinEul.y);
			cf1->assign(cosEul.z*sinEul.x + cosEul.x*cosEul.y*sinEul.z, cosEul.x*cosEul.y*cosEul.z - sinEul.x*sinEul.z, -cosEul.x*sinEul.y);
			cf2->assign(sinEul.y*sinEul.z, cosEul.z*sinEul.y, cosEul.y);
			break;
		case EulerXZY: //XZY
			cf0->assign(cosEul.y*cosEul.z,-sinEul.y,cosEul.y*sinEul.z);
			cf1->assign(sinEul.x*sinEul.z+cosEul.x*cosEul.z*sinEul.y,cosEul.x*cosEul.y,cosEul.x*sinEul.y*sinEul.z-cosEul.z*sinEul.x);
			cf2->assign(cosEul.z*sinEul.x*sinEul.y-cosEul.x*sinEul.z,cosEul.y*sinEul.x,cosEul.x*cosEul.z+sinEul.x*sinEul.y*sinEul.z);
			break;
		case EulerXYZ: //XYZ
			cf0->assign(cosEul.y*cosEul.z, -cosEul.y*sinEul.z, sinEul.y);
			cf1->assign(cosEul.x*sinEul.z + cosEul.z*sinEul.x*sinEul.y, cosEul.x*cosEul.z - sinEul.x*sinEul.y*sinEul.z,-cosEul.y*sinEul.x);
			cf2->assign(sinEul.x*sinEul.z - cosEul.x*cosEul.z*sinEul.y,cosEul.z*sinEul.x + cosEul.x*sinEul.y*sinEul.z,cosEul.x*cosEul.y);
			break;
		case EulerYXZ: //YXZ
			cf0->assign(cosEul.x*cosEul.z + sinEul.x*sinEul.y*sinEul.z, cosEul.z*sinEul.x*sinEul.y - cosEul.x*sinEul.z,cosEul.y*sinEul.x);
			cf1->assign(cosEul.y*sinEul.z, cosEul.y*cosEul.z,-sinEul.y);
			cf2->assign(cosEul.x*sinEul.y*sinEul.z - cosEul.z*sinEul.x,cosEul.x*cosEul.z*sinEul.y+sinEul.x*sinEul.z,cosEul.x*cosEul.y);
			break;
		case EulerYZX: //YZX
			cf0->assign(cosEul.x*cosEul.y, sinEul.x*sinEul.z - cosEul.x*cosEul.z*sinEul.y, cosEul.z*sinEul.x+cosEul.x*sinEul.y*sinEul.z);
			cf1->assign(sinEul.y, cosEul.y*cosEul.z, -cosEul.y*sinEul.z);
			cf2->assign(-cosEul.y*sinEul.x, cosEul.x*sinEul.z + cosEul.z*sinEul.x*sinEul.y, cosEul.x*cosEul.z - sinEul.x*sinEul.y*sinEul.z);
			break;
		case EulerZYX: //ZYX
			cf0->assign(cosEul.x*cosEul.y, -cosEul.z*sinEul.x + cosEul.x*sinEul.z*sinEul.y, sinEul.x*sinEul.z + cosEul.x*cosEul.z*sinEul.y);
			cf1->assign(cosEul.y*sinEul.x, cosEul.x*cosEul.z + sinEul.x*sinEul.y*sinEul.z, -cosEul.x*sinEul.z + cosEul.z*sinEul.x*sinEul.y);
			cf2->assign(-sinEul.y, cosEul.y*sinEul.z, cosEul.y*cosEul.z);
			break;
		case EulerZXY: //ZXY
			cf0->assign(cosEul.x*cosEul.z - sinEul.x*sinEul.y*sinEul.z, -cosEul.y*sinEul.x, cosEul.x*sinEul.z+cosEul.z*sinEul.x*sinEul.y);
			cf1->assign(cosEul.z*sinEul.x + cosEul.x*sinEul.y*sinEul.z, cosEul.x*cosEul.y, sinEul.x*sinEul.z - cosEul.x*cosEul.z*sinEul.y);
			cf2->assign(-cosEul.y*sinEul.z, sinEul.y, cosEul.y*cosEul.z);
			break;
	}
}
void block::redefAtoms(){
	//If the atomic coordinates are set relative to the elementary cell vectors (in [0,1] intervals)
	//this funvction calculates their absolute values by multiplying the coordinates and the cell vectors.
	for (unsigned int iAtom=0;iAtom<Natom;iAtom++){
		rAtom[iAtom]=e[0]*rAtom[iAtom].x+e[1]*rAtom[iAtom].y+e[2]*rAtom[iAtom].z;
		if (rearrangement) {
			for (unsigned int iNeighb=0;iNeighb<rAtomNeighb[iAtom].size();iNeighb++){
				rAtomNeighb[iAtom][iNeighb]=e[0]*rAtomNeighb[iAtom][iNeighb].x+e[1]*rAtomNeighb[iAtom][iNeighb].y+e[2]*rAtomNeighb[iAtom][iNeighb].z;
			}
		}
	}
}
void block::redefSymm(){
	//Calculates the absolute coordinates of translational vectors for symmetry equivalent positions.
	for (unsigned int iSymm=0;iSymm<Nsymm;iSymm++){
		rSymm[iSymm]=e[0]*rSymm[iSymm].x+e[1]*rSymm[iSymm].y+e[2]*rSymm[iSymm].z;
	}
}
void block::centerAtoms(){
	vect3d <double> rTemp;
	for (unsigned int iAtom = 0; iAtom < Natom; iAtom++)	rTemp += rAtom[iAtom];
	rTemp = rTemp / (double)Natom;
	for (unsigned int iAtom = 0; iAtom < Natom; iAtom++) rAtom[iAtom] -= rTemp;
}
void block::calcMean(){
	unsigned int Nat=Natom*Nsymm*Ncell.x*Ncell.y*Ncell.z;
	vect3d <double> r,rC,ra;
	for (unsigned int iAtom=0;iAtom<Natom;iAtom++){
		for (unsigned int iSymm = 0; iSymm<Nsymm; iSymm++){
			r.assign(pSymm[0][iSymm].dot(rAtom[iAtom]),pSymm[1][iSymm].dot(rAtom[iAtom]),pSymm[2][iSymm].dot(rAtom[iAtom]));
			rC=r+rSymm[iSymm];
			for (unsigned int iCellX = 0; iCellX<Ncell.x; iCellX++){
				for (unsigned int iCellY = 0; iCellY<Ncell.y; iCellY++){
					for (unsigned int iCellZ = 0; iCellZ<Ncell.z; iCellZ++){
						ra=rC+e[0]*(double)iCellX+e[1]*(double)iCellY+e[2]*(double)iCellZ;
						rMean+=ra;
					}
				}
			}
		}
	}
	rMean=rMean/(double)Nat;
}
void block::sortAtoms(map <string, unsigned int> ID){
	unsigned int Nel = (unsigned int) ID.size();
	vector <vect3d <double> > *rAtomVect;
	vector <double> *occVect;
	rAtomVect = new vector <vect3d <double> >[Nel];
	occVect = new vector <double>[Nel];
	for (unsigned int iAtom = 0; iAtom<Natom; iAtom++)	{
		occVect[ID[name[iAtom]]].push_back(occ[iAtom]);
		rAtomVect[ID[name[iAtom]]].push_back(rAtom[iAtom]);
	}
	for (unsigned int iEl = 0; iEl<Nel; iEl++) if (rAtomVect[iEl].size()) NatomEl.push_back( (unsigned int) rAtomVect[iEl].size());
	Nid = (unsigned int) NatomEl.size();
	NatomElAll.resize(Nid);
	delete[] name;
	name = NULL;
	id = new unsigned int[Nid];
	unsigned int k = 0, j = 0;
	for (unsigned int iEl = 0; iEl<Nel; iEl++) {
		if (rAtomVect[iEl].size()){
			id[j] = iEl; j++;
			for (unsigned int iAtom = 0; iAtom<rAtomVect[iEl].size(); iAtom++) {
				occ[k] = occVect[iEl][iAtom];
				rAtom[k] = rAtomVect[iEl][iAtom];
				k++;
			}
		}
	}
	if (rearrangement) {
		idNeighb = new vector <unsigned> [Natom];
		for (unsigned int iAtom = 0; iAtom < Natom; iAtom++)	{
			for (vector <string>::iterator iName = nameNeighb[iAtom].begin(); iName != nameNeighb[iAtom].end(); iName++) idNeighb[iAtom].push_back(ID[*iName]);
		}
		delete[] nameNeighb;
		nameNeighb = NULL;
	}
	delete[] rAtomVect;
	delete[] occVect;
}
vect3d<double> GetCenter(const vector < vect3d<double> > *r, unsigned int Nel, unsigned int Ntot){
	vect3d <double> rC(0,0,0);
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++)	rC += *ri;
	}
	return rC /= Ntot;
}
double GetEnsembleSize(const vector < vect3d<double> > *r, unsigned int Nel, unsigned int Ntot) {
	double Rmax = 0;
	vect3d <double> rC = GetCenter(r, Nel, Ntot);
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++)	Rmax=MAX(Rmax,(rC - *ri).sqr());
	}
	return 2.*sqrt(Rmax);
}
double GetAtomicDensity(const vector < vect3d<double> > *r, unsigned int Nel, unsigned int Ntot){
	double Rmax = SQR(0.33333*GetEnsembleSize(r, Nel, Ntot));
	vect3d <double> rC = GetCenter(r, Nel, Ntot);
	unsigned int count = 0;
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++)	{
			if ((rC - *ri).sqr() < Rmax) count++;
		}
	}
	Rmax = sqrt(Rmax);	
	return count / (4. / 3.*PI*Rmax*Rmax*Rmax);
}
unsigned int GetHistSize(const vector < vect3d<double> > *r,unsigned int Nel,double hist_bin){
	vect3d<double> Rmax, Rmin;
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		if (r[iEl].size()) {
			Rmax = Rmin = *r[iEl].begin();
			break;
		}
	}
	for (unsigned int iEl = 0; iEl < Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = r[iEl].begin(); ri != r[iEl].end(); ri++){
			if (Rmax.x < ri->x) Rmax.x = ri->x;
			if (Rmax.y < ri->y) Rmax.y = ri->y;
			if (Rmax.z < ri->z) Rmax.z = ri->z;
			if (Rmin.x > ri->x) Rmin.x = ri->x;
			if (Rmin.y > ri->y) Rmin.y = ri->y;
			if (Rmin.z > ri->z) Rmin.z = ri->z;
		}
	}
	return (unsigned int)((Rmax-Rmin).mag()/hist_bin)+1;
}
double Rmax2(vect3d <double> rAtom,vector< vect3d <double> > rAtomNeighb){
	double rmax2=0;
	for (unsigned int iNeib=0;iNeib<rAtomNeighb.size();iNeib++){
		rmax2=MAX(rmax2,(rAtom-rAtomNeighb[iNeib]).sqr());
	}	
	return rmax2;
}
int block::calcAtoms(vector < vect3d <double> > *ra){
	unsigned int i = 0, j = 0, is, ia, it;
	vect3d <double> celltrans, *rAtomCell,rTrans,ra0,ra1,*cf_mol[3],Euler,cosEul,sinEul,rMeanC,*r_mol_dev;
	vector <atom_info> *occNum;
	atom_info temp;
	double randval;
	bool *isOcc;
	normal_distribution<double> *distr, distr_mol = normal_distribution<double>(0, dev_mol);
	unsigned int seed = (unsigned int) chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator (seed);
	uniform_real_distribution<double> uniform_real(0.0,1.0);
	distr = new normal_distribution<double> [Natom];
	for (unsigned int iAtom=0;iAtom<Natom;iAtom++) distr[iAtom]=normal_distribution<double> (0,dev[iAtom]);
	if (cutoffcopies) {
		for (unsigned int iCopy=0;iCopy<Ncopy;iCopy++) rMeanC+=rCopy[iCopy];
		rMeanC=rMeanC/(double)Ncopy;
	}
	rAtomCell=new vect3d <double> [Natom*Nsymm];
	r_mol_dev = new vect3d <double>[Nsymm];
	for (unsigned int iAtom=0;iAtom<Natom;iAtom++){
		for (unsigned int iSymm = 0; iSymm<Nsymm; iSymm++){
			ra0.assign(pSymm[0][iSymm].dot(rAtom[iAtom]),pSymm[1][iSymm].dot(rAtom[iAtom]),pSymm[2][iSymm].dot(rAtom[iAtom]));
			rAtomCell[iAtom*Nsymm+iSymm]=ra0+rSymm[iSymm];
		}
	}
	occNum=new vector <atom_info>[Natom*Nsymm];
	unsigned int iEl = 0;
	unsigned int Nctype = NatomEl[0];
	for (unsigned int iAtom = 0; iAtom<Natom; iAtom++){
		if (iAtom>=Nctype) {iEl++; Nctype+=NatomEl[iEl];}
		for (unsigned int iSymm = 0; iSymm<Nsymm; iSymm++){
			temp.num=iAtom;
			temp.type=iEl;
			temp.symm=iSymm;
			temp.occ=occ[iAtom];
			occNum[i].push_back(temp);
			unsigned int jEl = iEl;
			unsigned int Nctypej = Nctype;
			unsigned int jSymmSt = iSymm + 1;
			j=i+1;
			for (unsigned int jAtom = iAtom; jAtom<Natom; jAtom++){
				if (jAtom>=Nctypej) {jEl++; Nctypej+=NatomEl[jEl];}
				for (unsigned int jSymm = jSymmSt; jSymm<Nsymm; jSymm++){
					int cellStX = -2, cellFinX = 3, cellStY = -2, cellFinY = 3, cellStZ = -2, cellFinZ = 3;
					if (Ncell.x==1) {cellStX=0; cellFinX=1;};
					if (Ncell.y==1) {cellStY=0; cellFinY=1;};
					if (Ncell.z==1) {cellStZ=0; cellFinZ=1;};
					for (int iCellX = cellStX; iCellX<cellFinX; iCellX++){
						for (int iCellY = cellStY; iCellY<cellFinY; iCellY++){
							for (int iCellZ = cellStZ; iCellZ<cellFinZ; iCellZ++){
								celltrans=e[0]*(double)iCellX+e[1]*(double)iCellY+e[2]*(double)iCellZ;
								if ((rAtomCell[i]-rAtomCell[j]+celltrans).sqr()<0.25)	{									
									temp.num=jAtom;
									temp.type=jEl;
									temp.symm=jSymm;
									if (jEl!=occNum[i].back().type) temp.occ=occNum[i].back().occ+occ[jAtom];
									else temp.occ=occNum[i].back().occ;
									occNum[i].push_back(temp);
								}
							}
						}
					}
					j++;
				}
				jSymmSt=0;
			}
			i++;
		}
	}
	for (unsigned int iEl = 0; iEl<Nid; iEl++) NatomElAll[iEl] = 0;
	isOcc=new bool[Natom*Nsymm];	
	for (unsigned int iCopy = 0; iCopy<Ncopy; iCopy++){
		for (unsigned int iCellX = 0; iCellX<Ncell.x; iCellX++){
			for (unsigned int iCellY = 0; iCellY<Ncell.y; iCellY++){
				for (unsigned int iCellZ = 0; iCellZ<Ncell.z; iCellZ++){
					celltrans=e[0]*(double)iCellX+e[1]*(double)iCellY+e[2]*(double)iCellZ;
					if (dev_mol >= 1.e-7){
						for (unsigned int iSymm = 0; iSymm < Nsymm; iSymm++){
							double fi = 2 * PI*uniform_real(generator);
							double theta = PI*uniform_real(generator);
							double Radd = ABS(distr_mol(generator));
							r_mol_dev[iSymm].assign(Radd*sin(theta)*cos(fi), Radd*sin(theta)*sin(fi), Radd*cos(theta));
						}
					}
					if (ro_mol){
						for (unsigned int k = 0; k<3; k++) cf_mol[k] = new vect3d <double>[Nsymm];
						for (unsigned int iMol = 0; iMol<Nsymm; iMol++){
							Euler.assign(2.f*PI*uniform_real(generator),PI*uniform_real(generator),2.f*PI*uniform_real(generator));
							cosEul=cos(Euler);
							sinEul=sin(Euler);
							cf_mol[0][iMol].assign(cosEul.x*cosEul.y-sinEul.x*sinEul.y*cosEul.z,-cosEul.x*sinEul.y-sinEul.x*cosEul.z*cosEul.y,sinEul.x*sinEul.z);
							cf_mol[1][iMol].assign(sinEul.x*cosEul.y+cosEul.x*sinEul.y*cosEul.z,-sinEul.x*sinEul.y+cosEul.x*cosEul.y*cosEul.z,-cosEul.x*sinEul.z);
							cf_mol[2][iMol].assign(sinEul.z*sinEul.y,sinEul.z*cosEul.y,cosEul.z);
						}
					}
					for (unsigned int i = 0; i<Natom*Nsymm; i++) isOcc[i] = false;
					for (unsigned int iAtom = 0; iAtom<Natom*Nsymm; iAtom++){
						if (!isOcc[iAtom]) {
							randval=uniform_real(generator);
							for (unsigned int j = 0; j<occNum[iAtom].size(); j++) isOcc[occNum[iAtom][j].num*Nsymm + occNum[iAtom][j].symm] = true;
							for (unsigned int j = 0; j<occNum[iAtom].size(); j++) {
								if (randval<=occNum[iAtom][j].occ) {
									ia=occNum[iAtom][j].num;
									it=occNum[iAtom][j].type;
									is=occNum[iAtom][j].symm;
									ra0.assign(pSymm[0][is].dot(rAtom[ia]),pSymm[1][is].dot(rAtom[ia]),pSymm[2][is].dot(rAtom[ia]));
									if (ro_mol)	ra1.assign(ra0.dot(cf_mol[0][is]),ra0.dot(cf_mol[1][is]),ra0.dot(cf_mol[2][is]));
									else ra1=ra0;
									ra1 += rSymm[is] + celltrans + r_mol_dev[is];
									if ((!cutoff)||((ra1-rMean).sqr()<=SQR(Rcut))) {
										rTrans.assign((ra1-rMean).dot(cf[0][iCopy]),(ra1-rMean).dot(cf[1][iCopy]),(ra1-rMean).dot(cf[2][iCopy]));
										ra1=rCopy[iCopy]+rTrans;
										if (dev[ia]>=1.e-7) {
											double fi=2*PI*uniform_real(generator);
											double theta=PI*uniform_real(generator);
											double Radd=ABS(distr[ia](generator));
											vect3d <double> radd(Radd*sin(theta)*cos(fi),Radd*sin(theta)*sin(fi),Radd*cos(theta));
											ra1+=radd;
										}
										if ((!cutoffcopies) || ((ra1 - rMeanC).sqr() <= SQR(RcutCopies)))	{
											ra[id[it]].push_back(ra1);
											NatomElAll[it]++;
										}
									}
									break;
								}
							}
						}
					}
					if (ro_mol) for (unsigned int k = 0; k<3; k++) delete[] cf_mol[k];
				}
			}
		}
	}
	delete[] isOcc;
	delete[] occNum;
	delete[] rAtomCell;
	delete[] r_mol_dev;
	for (unsigned int iEl = 0; iEl<Nid; iEl++) N += NatomElAll[iEl];
	delete[] distr;
	return 0;
}
unsigned int CalcAndPrintAtoms(config *cfg, block *Block, vector < vect3d <double> > **ra, unsigned int **NatomEl, map <string, unsigned int> ID){
	unsigned int Ntot = 0;
	*ra = new vector < vect3d <double> >[cfg->Nel];
	*NatomEl = new unsigned int[cfg->Nel];
	if (!myid) {
		for (unsigned int iB = 0; iB<cfg->Nblocks; iB++) Block[iB].calcAtoms(*ra);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			(*NatomEl)[iEl] = (unsigned int) (*ra)[iEl].size();
			Ntot += (*NatomEl)[iEl];
		}
		if (cfg->scenario > Debye) cfg->Nhist = GetHistSize(*ra, cfg->Nel, cfg->hist_bin);
		if (cfg->PrintAtoms)	printAtoms(*ra, cfg->name, ID, Ntot);
		if ((cfg->scenario > Debye_hist) && (!cfg->p0)) {
			cfg->p0 = GetAtomicDensity(*ra, cfg->Nel, Ntot);
			cout << "Approximate atomic density of the sample is " << cfg->p0 << " A^(-3).\n" << endl;
		}
	}
#ifdef UseMPI
	MPI_Bcast(&Ntot, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (cfg->scenario > Debye_hist) MPI_Bcast(&cfg->p0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(*NatomEl, cfg->Nel, MPI_INT, 0, MPI_COMM_WORLD);
	if (cfg->scenario == s2D){
		//cout << myid << endl;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			if (!myid) {
				for (unsigned int pid = 1; pid<(unsigned int) (numprocs); pid++){
					unsigned int ist = (*NatomEl)[iEl] / numprocs*pid;
					(pid<(*NatomEl)[iEl]%numprocs) ? ist += pid : ist += (*NatomEl)[iEl]%numprocs;
					unsigned int Nsend = (*NatomEl)[iEl] / numprocs;
					if (pid<(*NatomEl)[iEl]%numprocs) Nsend++;
					MPI_Send(&(*ra)[iEl][ist], 3*Nsend, MPI_DOUBLE, pid, pid, MPI_COMM_WORLD);
				}
			}
			unsigned int NatomElNew = (*NatomEl)[iEl] / numprocs;
			if ((unsigned int)(myid)<(*NatomEl)[iEl] % (unsigned int)(numprocs)) NatomElNew++;
			(*NatomEl)[iEl] = NatomElNew;
			(*ra)[iEl].resize(NatomElNew);
			MPI_Status status;
			if (myid) 	MPI_Recv(&(*ra)[iEl][0], 3*NatomElNew, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD, &status);
		}
	}
	else {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			if (myid) (*ra)[iEl].resize((*NatomEl)[iEl]);
			MPI_Bcast(&(*ra)[iEl][0], 3*(*NatomEl)[iEl], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
	}
	if ((cfg->scenario == Debye) && (cfg->calcPartialIntensity)) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++) {
			MPI_Bcast(&Block[iB].N, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (myid) Block[iB].NatomElAll.resize(Block[iB].Nid);
			MPI_Bcast(&Block[iB].NatomElAll[0], Block[iB].Nid, MPI_INT, 0, MPI_COMM_WORLD);
		}
	}
	if (cfg->scenario > Debye) MPI_Bcast(&cfg->Nhist, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
	return Ntot;
}
void PolarFactor2D(double **I2D, const double *q, double lambda, unsigned int Nq, unsigned int Nfi){
	double sintheta, cos2theta, factor;
	for (unsigned int iq = 0; iq<Nq; iq++){
		sintheta = q[iq] * (lambda * 0.25 / PI);
		cos2theta = 1. - 2. * SQR(sintheta);
		factor = 0.5 * (1. + SQR(cos2theta));
		for (unsigned int ifi = 0; ifi<Nfi; ifi++) I2D[iq][ifi] *= factor;
	}
}
void PolarFactor1D(double *I, const double *q, double lambda,unsigned int Nq){
	double sintheta, cos2theta, factor;
	for (unsigned int iq = 0; iq<Nq; iq++){
		sintheta = q[iq] * (lambda * 0.25 / PI);
		cos2theta = 1. - 2. * SQR(sintheta);
		factor = 0.5 * (1. + SQR(cos2theta));
		I[iq] *= factor;
	}
}
void RearrangementInt(double *I0, const double *q, const config *cfg, const block *Block, vector<double*> FF, vector<double> SL, unsigned int Ntot){
	double rij2, rij, qrij, rmax2=0, *Itemp = NULL, *I = NULL;
	long long int N = SQR(Ntot);
	unsigned int Nint = cfg->q.N;
	Itemp = new double[cfg->q.N];
	if (cfg->calcPartialIntensity) Nint = (cfg->Nblocks + 1) * cfg->q.N;
	I = new double[Nint];
	for (unsigned int iq = 0; iq < Nint; iq++) I[iq] = 0;
	for (unsigned int iB = 0; iB<cfg->Nblocks; iB++){
		unsigned int iqST = 0;
		if (cfg->calcPartialIntensity) iqST += cfg->q.N;
		if (Block[iB].rearrangement){			
			unsigned int coeff = Block[iB].Ncopy*Block[iB].Ncell.x*Block[iB].Ncell.y*Block[iB].Ncell.z*Block[iB].Nsymm;
			unsigned int ibegin=0;
			for (unsigned int iEl=0;iEl<Block[iB].Nid;iEl++){
				unsigned int jbegin = 0;
				unsigned int id = Block[iB].id[iEl];
				for (unsigned int jEl = 0; jEl<Block[iB].Nid; jEl++){
					unsigned int jd = Block[iB].id[jEl];
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) Itemp[iq] = 0;
					for (unsigned int iAtom=ibegin;iAtom<ibegin+Block[iB].NatomEl[iEl];iAtom++){
						rmax2=Rmax2(Block[iB].rAtom[iAtom],Block[iB].rAtomNeighb[iAtom])+SQR(Block[iB].Rcorr);
						for (unsigned int jAtom=jbegin;jAtom<jbegin+Block[iB].NatomEl[jEl];jAtom++){
							rij2=(Block[iB].rAtom[iAtom]-Block[iB].rAtom[jAtom]).sqr();
							if ((rij2<rmax2)&&(rij2>0.000001)){								
								N-=coeff;
								rij = sqrt(rij2);
								for (unsigned int iq = 0; iq < cfg->q.N; iq++){
									qrij = rij*q[iq];
									Itemp[iq] += sin(qrij) / (qrij + 0.00000001);
								}
							}
						}
					}
					if (cfg->source == xray) for (unsigned int iq = 0; iq < cfg->q.N; iq++) I[iqST+iq] -= (coeff*FF[id][iq] * FF[jd][iq]) * Itemp[iq];
					else for (unsigned int iq = 0; iq < cfg->q.N; iq++) I[iqST+iq] -= (coeff*SL[id] * SL[jd]) * Itemp[iq];
					jbegin += Block[iB].NatomEl[jEl];
				}
				for (unsigned int iAtom = ibegin; iAtom<ibegin + Block[iB].NatomEl[iEl]; iAtom++){
					for (unsigned int iNeib = 0; iNeib < Block[iB].rAtomNeighb[iAtom].size(); iNeib++){
						unsigned int idN = Block[iB].idNeighb[iAtom][iNeib];
						rij = (Block[iB].rAtom[iAtom] - Block[iB].rAtomNeighb[iAtom][iNeib]).mag();
						N += coeff;
						if (cfg->source == xray) {
							for (unsigned int iq = 0; iq < cfg->q.N; iq++){
								qrij = rij*q[iq];
								I[iqST+iq] += (coeff * FF[id][iq] * FF[idN][iq]) *sin(qrij) / (qrij + 0.00000001);
							}
						}
						else {
							for (unsigned int iq = 0; iq < cfg->q.N; iq++){
								qrij = rij*q[iq];
								I[iqST+iq] += (coeff * SL[id] * SL[idN]) *sin(qrij) / (qrij + 0.00000001);
							}
						}
					}
				}
				ibegin += Block[iB].NatomEl[iEl];				
			}
		}		
	}
	if (cfg->calcPartialIntensity) {
		unsigned int iqST = cfg->q.N;
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++, iqST += cfg->q.N){
			unsigned int i0qST = cfg->q.N * (1 + iB * (2 * cfg->Nblocks - iB + 1) / 2);
			if (cfg->PolarFactor) PolarFactor1D(I + iqST, q, cfg->lambda, cfg->q.N);
			for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
				I[iq] += I[iqST + iq];
				I0[i0qST + iq] = (I0[i0qST + iq] * Ntot + I[iqST + iq]) / sqrt(double(N));
			}
		}
	}
	else if (cfg->PolarFactor) PolarFactor1D(I, q, cfg->lambda, cfg->q.N);
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) I0[iq] = (I0[iq] * Ntot + I[iq]) / sqrt(double(N));
	delete [] Itemp;
	delete[] I;
}
#if !defined(UseOCL) && !defined(UseCUDA)
void calcIntDebye(double **I, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, unsigned int Ntot, int NumOMPthreads = 1) {
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	double *Itemp = NULL;
	if (!myid) {
		*I = new double[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = 0;
	}
	Itemp = new double[cfg->q.N*NumOMPthreads];
#ifdef UseMPI
	double *Iloc = NULL;
	Iloc = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) Iloc[iq] = 0;
#endif
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++){
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
		{
			int tid = 0;
#ifdef UseOMP
			tid = omp_get_thread_num();
#endif	
			unsigned int id = myid*NumOMPthreads + tid, jAtomST = 0;
			for (unsigned int iq = 0; iq < cfg->q.N; iq++) Itemp[tid*cfg->q.N + iq] = 0;
			unsigned int step = 2 * id + 1, count = 0;
			for (unsigned int iAtom = id; iAtom < NatomEl[iEl]; iAtom += step, count++) {
				(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
				(jEl == iEl) ? jAtomST = iAtom + 1 : jAtomST = 0;
				for (unsigned int jAtom = jAtomST; jAtom < NatomEl[jEl]; jAtom++) {
					double rij = (ra[iEl][iAtom] - ra[jEl][jAtom]).mag();
					for (unsigned int iq = 0; iq < cfg->q.N; iq++)	{
						double qrij = rij*q[iq];
						Itemp[tid*cfg->q.N + iq] += sin(qrij) / (qrij + 0.00000001);
					}
				}
			}
		}
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
		for (int iq = 0; iq < (int) cfg->q.N; iq++) {
			for (int tid = 1; tid < NumOMPthreads; tid++) Itemp[iq] += Itemp[tid * cfg->q.N + iq];
		}
#endif
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += Itemp[iq] * FF[iEl][iq] * FF[jEl][iq] * 2.;
#else
					(*I)[iq] += Itemp[iq] * FF[iEl][iq] * FF[jEl][iq] * 2.;
#endif
				}
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += Itemp[iq] * SL[iEl] * SL[jEl] * 2.;
#else
					(*I)[iq] += Itemp[iq] * SL[iEl] * SL[jEl] * 2.;
#endif
				}
			}
		}
	}
#ifdef UseMPI
	MPI_Reduce(Iloc, *I, cfg->q.N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
	if (!myid) {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(FF[iEl][iq]);
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(SL[iEl]);
			}
		}
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] /= Ntot;
		if (cfg->PolarFactor) PolarFactor1D(*I, q, cfg->lambda, cfg->q.N);
	}
	delete[] Itemp;
#ifdef UseMPI
	delete[] Iloc;
#endif
	t2 = chrono::steady_clock::now();
	if (!myid) cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}
void calcIntPartialDebye(double **I, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, const block *Block, unsigned int Ntot, int NumOMPthreads = 1) {
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	double *Itemp = NULL;
	unsigned int *NatomElBlock = NULL;
	unsigned int Nparts = (cfg->Nblocks * (cfg->Nblocks + 1)) / 2, Isize = Nparts * cfg->q.N;
	if (!myid) {
		*I = new double[Isize + cfg->q.N];
		for (unsigned int iq = 0; iq < Isize + cfg->q.N; iq++) (*I)[iq] = 0;
	}
	Itemp = new double[Isize*NumOMPthreads];
#ifdef UseMPI
	double *Iloc = NULL;
	Iloc = new double[Isize];
	for (unsigned int iq = 0; iq < Isize; iq++) Iloc[iq] = 0;
#endif
	NatomElBlock = new unsigned int[cfg->Nel*cfg->Nblocks];
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			NatomElBlock[iEl*cfg->Nblocks + iB] = 0;
			for (unsigned int iBtype = 0; iBtype < Block[iB].Nid; iBtype++) {
				if (Block[iB].id[iBtype] == iEl) {
					NatomElBlock[iEl*cfg->Nblocks + iB] = Block[iB].NatomElAll[iBtype];
					break;
				}
			}
		}
	}
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){		
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++){
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
			{
				int tid = 0;
#ifdef UseOMP
				tid = omp_get_thread_num();
#endif
				unsigned int id = myid*NumOMPthreads + tid;
				for (unsigned int iq = 0; iq < Isize; iq++) Itemp[tid*Isize + iq] = 0;
				unsigned int iAtomSB = 0, jAtomSB = 0, jBlockST = 0, jAtomST = 0;
				for (unsigned int iB = 0; iB < cfg->Nblocks; iAtomSB += NatomElBlock[iEl*cfg->Nblocks + iB], iB++) {
					unsigned int step = 2 * id + 1, count = 0, Istart = 0;
					for (unsigned int iAtom = iAtomSB + id; iAtom < iAtomSB + NatomElBlock[iEl*cfg->Nblocks + iB]; iAtom += step, count++) {
						(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
						if (jEl == iEl) {
							jBlockST = iB;
							jAtomSB = iAtomSB;
							jAtomST = iAtom + 1;
						}
						else jAtomSB = 0;
						for (unsigned int jB = jBlockST; jB < cfg->Nblocks; jAtomSB += NatomElBlock[jEl*cfg->Nblocks + jB], jB++) {
							(jB > iB) ? Istart = cfg->q.N * (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + jB) : Istart = cfg->q.N * (cfg->Nblocks * jB - (jB * (jB + 1)) / 2 + iB);
							for (unsigned int jAtom = MAX(jAtomSB, jAtomST); jAtom < jAtomSB + NatomElBlock[jEl*cfg->Nblocks + jB]; jAtom++) {
								double rij = (ra[iEl][iAtom] - ra[jEl][jAtom]).mag();
								for (unsigned int iq = 0; iq < cfg->q.N; iq++)	{
									double qrij = rij*q[iq];
									Itemp[tid*Isize + Istart + iq] += sin(qrij) / (qrij + 0.00000001);
								}
							}
						}
					}
				}				
			}
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
			for (int iq = 0; iq < (int)Isize; iq++) {
				for (int tid = 1; tid < NumOMPthreads; tid++) Itemp[iq] += Itemp[tid * Isize + iq];
			}
#endif
			if (cfg->source == xray) {
				for (unsigned iPart = 0; iPart < Nparts; iPart++) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
						Iloc[iPart *cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * FF[iEl][iq] * FF[jEl][iq] * 2.;
#else
						(*I)[(iPart + 1)*cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * FF[iEl][iq] * FF[jEl][iq] * 2.;
#endif
					}
				}
			}
			else {
				for (unsigned iPart = 0; iPart < Nparts; iPart++) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
						Iloc[iPart *cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * SL[iEl] * SL[jEl] * 2.;
#else
						(*I)[(iPart +1)*cfg->q.N + iq] += Itemp[iPart *cfg->q.N + iq] * SL[iEl] * SL[jEl] * 2.;
#endif
					}
				}
			}
		}
	}
#ifdef UseMPI
	MPI_Reduce(Iloc, *I + cfg->q.N, Isize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	delete[] Iloc;
#endif
	if (!myid) {
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
			for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
				unsigned int Istart = cfg->q.N * (1 + (cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + iB));
				if (cfg->source == xray) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[Istart + iq] += NatomElBlock[iEl*cfg->Nblocks + iB] * SQR(FF[iEl][iq]);
				}
				else {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[Istart + iq] += NatomElBlock[iEl*cfg->Nblocks + iB] * SQR(SL[iEl]);
				}
			}
		}
		for (unsigned int iq = cfg->q.N; iq < Isize + cfg->q.N; iq++) (*I)[iq] /= Ntot;
		if (cfg->PolarFactor) {
			for (unsigned iPart = 1; iPart < Nparts + 1; iPart++) PolarFactor1D(*I + iPart * cfg->q.N, q, cfg->lambda, cfg->q.N);
		}
		for (unsigned iPart = 1; iPart < Nparts + 1; iPart++) {
			for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += (*I)[iPart*cfg->q.N + iq];
		}
	}
	delete[] NatomElBlock;
	delete[] Itemp;
	t2 = chrono::steady_clock::now();
	if (!myid) cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}
void calcHist(unsigned long long int **rij_hist, const vector < vect3d <double> > *ra, const unsigned int *NatomEl, unsigned int Nel, unsigned int Nhist, double bin, int NumOMPthreads = 1) {
	unsigned int NhistEl = (Nel*(Nel + 1)) / 2 * Nhist;
#ifdef UseMPI
	unsigned long long int *rij_hist_loc;
	rij_hist_loc = new unsigned long long int[NhistEl* NumOMPthreads];
	for (unsigned int i = 0; i < NhistEl* NumOMPthreads; i++) rij_hist_loc[i] = 0;
#endif
	if (!myid) {
		*rij_hist = new unsigned long long int[NhistEl* NumOMPthreads];
		for (unsigned int i = 0; i < NhistEl* NumOMPthreads; i++) (*rij_hist)[i] = 0;
	}
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
{
	int tid = 0;
#ifdef UseOMP
	tid = omp_get_thread_num();
#endif	
	unsigned int id = myid*NumOMPthreads + tid, Nstart = NhistEl*tid, jAtomST = 0;
	for (unsigned int iEl = 0; iEl < Nel; iEl++) {
		for (unsigned int jEl = iEl; jEl < Nel; jEl++, Nstart += Nhist) {
			unsigned int step = 2 * id + 1, count = 0;
			for (unsigned int iAtom = id; iAtom < NatomEl[iEl]; iAtom += step, count++) {
				(count % 2) ? step = 2 * id + 1 : step = 2 * (numprocs*NumOMPthreads - id) - 1;
				(jEl == iEl) ? jAtomST = iAtom + 1 : jAtomST = 0;
				for (unsigned int jAtom = jAtomST; jAtom < NatomEl[jEl]; jAtom++) {
					double rij = (ra[iEl][iAtom] - ra[jEl][jAtom]).mag();
#ifdef UseMPI
					rij_hist_loc[Nstart + (unsigned int)(rij / bin)]+=1;
#else
					(*rij_hist)[Nstart + (unsigned int)(rij / bin)] += 1;
#endif
				}
			}
		}
	}
}
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
for (int i = 0; i < (int) NhistEl; i++) {
#ifdef UseMPI
	for (int tid = 1; tid < NumOMPthreads; tid++) rij_hist_loc[i] += rij_hist_loc[tid*NhistEl + i];
#else
	for (int tid = 1; tid < NumOMPthreads; tid++) (*rij_hist)[i] += (*rij_hist)[tid*NhistEl + i];
#endif
}
#endif
#ifdef UseMPI
	MPI_Reduce(rij_hist_loc, *rij_hist, NhistEl, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	delete[] rij_hist_loc;
#endif	
}
void calcInt1DHist(double **I, unsigned long long int *rij_hist, const unsigned int *NatomEl, unsigned int Nel, const config *cfg, vector <double*> FF, vector<double> SL, const double *q, unsigned int Ntot, int NumOMPthreads = 1) {
	unsigned int Nhist = cfg->Nhist/numprocs, iStart = 0;
	double *Itemp = NULL;
	*I = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = 0;
#ifdef UseMPI
	double *Iloc = NULL;
	Iloc = new double[cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) Iloc[iq] = 0;
	Iloc = new double [cfg->q.N];
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) Iloc[iq] = 0;
	const int reminder = int(cfg->Nhist)%numprocs;
	if (myid<reminder) Nhist++;
	unsigned int NhistEl = (Nel*(Nel + 1)) / 2;
	iStart = cfg->Nhist/numprocs*myid;
	(myid<reminder) ? iStart += myid : iStart += (unsigned int) (reminder);
	if (myid) rij_hist = new unsigned long long int[NhistEl * Nhist];
	for (unsigned int i = 0; i < NhistEl; i++) {
		MPI_Status status;
		if (!myid) {
			for (int pid = 1; pid<numprocs; pid++){
				unsigned int ist = cfg->Nhist/numprocs*pid;
				(pid<reminder) ? ist += pid : ist += (unsigned int)(reminder);
				unsigned int Nsend = cfg->Nhist / numprocs;
				if (pid<reminder) Nsend++;
				MPI_Send(&rij_hist[i*cfg->Nhist + ist], Nsend, MPI_LONG_LONG_INT, pid, pid, MPI_COMM_WORLD);
			}
		}		
		else MPI_Recv(&rij_hist[i*Nhist], Nhist, MPI_LONG_LONG_INT, 0, myid, MPI_COMM_WORLD, &status);
	}
#endif
	unsigned int Nhist0 = Nhist, Nstart = 0;
	if (!myid) Nhist0 = cfg->Nhist;
	Itemp = new double[cfg->q.N * NumOMPthreads];
	for (unsigned int iEl = 0; iEl < Nel; iEl++) {
		for (unsigned int jEl = iEl; jEl < Nel; jEl++, Nstart += Nhist0) {
#ifdef UseOMP
#pragma omp parallel num_threads(NumOMPthreads) 
#endif
			{
				int tid = 0;
#ifdef UseOMP
				tid = omp_get_thread_num();
#endif
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) Itemp[tid * cfg->q.N + iq] = 0;

				for (int i = tid; i < (int)Nhist; i += NumOMPthreads) {
					if (rij_hist[Nstart + i]) {
						double rij = ((double)(iStart + i)+0.5) * cfg->hist_bin;
						for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
							double qrij = rij * q[iq];
							Itemp[tid * cfg->q.N + iq] += rij_hist[Nstart + i] * sin(qrij) / (qrij + 0.00000001);
						}
					}
				}
			}
#ifdef UseOMP
#pragma omp parallel for num_threads(NumOMPthreads) 
			for (int iq = 0; iq < (int) cfg->q.N; iq++) {
				for (int tid = 1; tid < NumOMPthreads; tid++) Itemp[iq] += Itemp[tid * cfg->q.N + iq];
			}
#endif
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += 2. * Itemp[iq] * FF[iEl][iq] * FF[jEl][iq];
#else
					(*I)[iq] += 2. * Itemp[iq] * FF[iEl][iq] * FF[jEl][iq];
#endif
				}
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) {
#ifdef UseMPI
					Iloc[iq] += 2. * Itemp[iq] * SL[iEl] * SL[jEl];
#else
					(*I)[iq] += 2. * Itemp[iq] * SL[iEl] * SL[jEl];
#endif
				}
			}
		}
	}
#ifdef UseMPI
	MPI_Reduce(Iloc, *I, cfg->q.N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
	if (!myid) {
		for (unsigned int iEl = 0; iEl < Nel; iEl++) {
			if (cfg->source == xray) {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(FF[iEl][iq]);
			}
			else {
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] += NatomEl[iEl] * SQR(SL[iEl]);
			}
		}
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] /= Ntot;
		if (cfg->PolarFactor) PolarFactor1D(*I, q, cfg->lambda, cfg->q.N);
	}
	delete[] Itemp;
#ifdef UseMPI
	delete[] Iloc;
#endif
}
void calcPDFandDebye(double **I, double **PDF, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, unsigned int Ntot, int NumOMPthreads = 1) {
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	unsigned long long int *rij_hist = NULL;
	calcHist(&rij_hist, ra, NatomEl, cfg->Nel, cfg->Nhist, cfg->hist_bin, NumOMPthreads);
	t2 = chrono::steady_clock::now();
	if (!myid) cout << "Histogram calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	if ((cfg->scenario > Debye_hist)&&(!myid)) {
		t1 = chrono::steady_clock::now();
		unsigned int NPDF = (1 + (cfg->Nel*(cfg->Nel + 1)) / 2) * cfg->Nhist;
		*PDF = new double[NPDF];
		for (unsigned int i = 0; i < NPDF; i++) (*PDF)[i] = 0;
		double Faverage2 = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			Faverage2 += SL[iEl] * NatomEl[iEl];
		}
		Faverage2 /= Ntot;
		Faverage2 *= Faverage2;
		unsigned int Nstart = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
				double mult, sub, r, multIJ = SL[iEl] * SL[jEl] / Faverage2;
				switch (cfg->PDFtype){
				case typeRDF:
					mult = 2. / (cfg->hist_bin*Ntot);
					for (unsigned int i = Nstart; i < Nstart + cfg->Nhist; i++) (*PDF)[cfg->Nhist + i] = rij_hist[i] * mult;
					break;
				case typePDF:
					mult = 0.5 / (PI*cfg->hist_bin*cfg->p0*Ntot);
					for (unsigned int i = Nstart; i < Nstart + cfg->Nhist; i++) {
						r = (i + 0.5) * cfg->hist_bin;
						(*PDF)[cfg->Nhist + i] = rij_hist[i] * mult / SQR(r);
					}					
					break;
				case typeRPDF:
					mult = 2. / (cfg->hist_bin*Ntot);
					(jEl > iEl) ? sub = 8.*PI*cfg->p0*double(NatomEl[iEl]) * double(NatomEl[jEl]) / SQR(double(Ntot)) : sub = 4.*PI*cfg->p0*SQR(double(NatomEl[iEl])) / SQR(double(Ntot));
					for (unsigned int i = Nstart; i < Nstart + cfg->Nhist; i++) {
						r = (i + 0.5) * cfg->hist_bin;
						(*PDF)[cfg->Nhist + i] = rij_hist[i] * mult / r - sub * r;
					}
					break;
				}
				for (unsigned int i = 0; i < cfg->Nhist; i++) (*PDF)[i] += (*PDF)[cfg->Nhist + Nstart + i] * multIJ;
			}
		}
		t2 = chrono::steady_clock::now();
		if (!myid) cout << "PDF calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	if ((cfg->scenario == Debye_hist) || (cfg->scenario == DebyePDF)) {
		t1 = chrono::steady_clock::now();
		calcInt1DHist(I, rij_hist, NatomEl, cfg->Nel, cfg, FF, SL, q, Ntot, NumOMPthreads);
		t2 = chrono::steady_clock::now();
		if (!myid) cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	if (rij_hist != NULL) delete [] rij_hist;
}
void calcInt2D(double ***I2D, double **I, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, unsigned int Ntot, int NumOMPthreads = 1) {
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	double *A_im = NULL, *A_real = NULL, *sintheta = NULL, *costheta = NULL, *sinfi = NULL, *cosfi = NULL, qr, ar, ai, lFF;
#ifdef UseMPI
	double *A_im_sum = NULL, *A_real_sum = NULL;
#endif
	const unsigned int N2D = cfg->q.N*cfg->Nfi;
	const double deltafi = 2.*PI / cfg->Nfi;
	unsigned int index;
	vect3d <double> CS[3], qv;
	if (!myid) {
		*I = new double[cfg->q.N];
		*I2D = new double*[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++){
			(*I)[iq] = 0;
			(*I2D)[iq] = new double[cfg->Nfi];
			for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++) (*I2D)[iq][ifi] = 0;
		}
#ifdef UseMPI
		A_im_sum = new double[N2D];
		A_real_sum = new double[N2D];
#endif
	}
	A_im = new double[N2D];
	A_real = new double[N2D];
	sintheta = new double[cfg->q.N];
	costheta = new double[cfg->q.N];
	sinfi = new double[cfg->Nfi];
	cosfi = new double[cfg->Nfi];
	for (unsigned int iq = 0; iq<cfg->q.N; iq++){
		sintheta[iq] = q[iq] * (cfg->lambda * 0.25 / PI);
		costheta[iq] = 1. - SQR(sintheta[iq]);
	}
	for (unsigned int ifi = 0; ifi<cfg->Nfi; ifi++){
		cosfi[ifi] = cos(ifi*deltafi);
		sinfi[ifi] = sin(ifi*deltafi);
	}
	double dalpha = (cfg->Euler.max.x - cfg->Euler.min.x) / cfg->Euler.N.x, dbeta = (cfg->Euler.max.y - cfg->Euler.min.y) / cfg->Euler.N.y, dgamma = (cfg->Euler.max.z - cfg->Euler.min.z) / cfg->Euler.N.z;
	if (cfg->Euler.N.x<2) dalpha = 0;
	if (cfg->Euler.N.y<2) dbeta = 0;
	if (cfg->Euler.N.z<2) dgamma = 0;
	vect3d <double> cf0, cf1, cf2;
	for (unsigned int ia = 0; ia < cfg->Euler.N.x; ia++){
		double alpha = cfg->Euler.min.x + (ia + 0.5)*dalpha;
		for (unsigned int ib = 0; ib < cfg->Euler.N.y; ib++){
			double beta = cfg->Euler.min.y + (ib + 0.5)*dbeta;
			for (unsigned int ig = 0; ig < cfg->Euler.N.z; ig++){
				double gamma = cfg->Euler.min.z + (ig + 0.5)*dgamma;
				vect3d <double> euler(alpha,beta,gamma);
				calcRotMatrix(&cf0,&cf1,&cf2, euler, cfg->EulerConvention);
				CS[0].assign(cf0.x, cf1.x, cf2.x);
				CS[1].assign(cf0.y, cf1.y, cf2.y);
				CS[2].assign(cf0.z, cf1.z, cf2.z);
#ifdef UseOMP
#pragma omp parallel for private(index,qv,ar,ai,qr,lFF) num_threads(NumOMPthreads)
#endif
				for (int iq = 0; iq < (int)cfg->q.N; iq++){
					for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++){
						index = cfg->Nfi*iq + ifi;
						A_real[index] = 0;
						A_im[index] = 0;
						qv.assign(costheta[iq] * cosfi[ifi], costheta[iq] * sinfi[ifi], -sintheta[iq]);
						qv = qv*q[iq];
						qv.assign(qv.dot(CS[0]), qv.dot(CS[1]), qv.dot(CS[2]));
						for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
							ar = 0; ai = 0;
							for (vector<vect3d <double> >::const_iterator ri = ra[iEl].begin(); ri != ra[iEl].end(); ri++){
								qr = qv.dot(*ri);
								ar += cos(qr);
								ai += sin(qr);
							}
							(cfg->source == xray) ? lFF = FF[iEl][iq] : lFF = SL[iEl];
							A_real[index] += lFF * ar;
							A_im[index] += lFF * ai;
						}
					}
				}
#ifdef UseMPI
				MPI_Reduce(A_real, A_real_sum, N2D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				MPI_Reduce(A_im, A_im_sum, N2D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
				if (!myid) {
					for (unsigned int iq = 0; iq < cfg->q.N; iq++){
						for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++){
							index = cfg->Nfi*iq + ifi;
#ifdef UseMPI
							(*I2D)[iq][ifi] += (SQR(A_real_sum[index]) + SQR(A_im_sum[index]));
#else
							(*I2D)[iq][ifi] += (SQR(A_real[index]) + SQR(A_im[index]));
#endif
						}
					}
				}
			}
		}
	}
	if (!myid) {
		double norm = 1. / (Ntot*cfg->Euler.N.x*cfg->Euler.N.y*cfg->Euler.N.z);
		for (unsigned int iq = 0; iq < cfg->q.N; iq++){
			for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++) (*I2D)[iq][ifi] *= norm;
		}
		if (cfg->PolarFactor) PolarFactor2D(*I2D, q, cfg->lambda, cfg->q.N, cfg->Nfi);
		for (unsigned int iq = 0; iq < cfg->q.N; iq++){
			for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++)	(*I)[iq] += (*I2D)[iq][ifi];
			(*I)[iq] /= cfg->Nfi;
		}
	}
#ifdef UseMPI
	if (!myid) {
		delete[] A_im_sum;
		delete[] A_real_sum;
	}
#endif
	delete[] sintheta;
	delete[] costheta;
	delete[] sinfi;
	delete[] cosfi;
	delete[] A_im;
	delete[] A_real;
	t2 = chrono::steady_clock::now();
	if (!myid) cout << "2D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}
#endif
