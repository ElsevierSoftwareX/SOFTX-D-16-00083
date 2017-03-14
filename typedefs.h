//Copyright (C) 2015, Vladislav Neverov, NRC "Kurchatov institute"
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

//macros, structures and class definitions are here
#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_
#include <math.h>
#include <string>
#include <iostream>
#include <map>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;

//#define UseOMP
//#define UseMPI
//#define UseCUDA

//sizes of thread blocks for CUDA and OpenCL
#define BlockSize1Dsmall 256
#define BlockSize1Dmedium 512
#define BlockSize1Dlarge 1024
#define BlockSize2Dsmall 16
#define BlockSize2Dlarge 32

//computational scenarios
#define s2D 0 //2D diffraction pattern
#define Debye 1 //Powder diffraction pattern using the Debye equation
#define Debye_hist 2 //Powder diffraction pattern using the histogram approximation
#define PDFonly 3 //PDF only
#define DebyePDF 4 //PDF and powder diffraction pattern using the histogram approximation

//source types
#define neutron 0
#define xray 1

//PDF types
#define typeRDF 0
#define typePDF 1
#define typeRPDF 2

//Euler conventions
#define EulerXZX 0
#define EulerXYX 1
#define EulerYXY 2
#define EulerYZY 3
#define EulerZYZ 4
#define EulerZXZ 5
#define EulerXZY 6
#define EulerXYZ 7
#define EulerYXZ 8
#define EulerYZX 9
#define EulerZYX 10
#define EulerZXY 11

//constants
#define PI 3.1415926535897932384626433832795028841971693993
#define PIf 3.14159265f

//some macros
#define BOOL(x) ((x) ? 1 : 0)
#define ABS(x) ((x)<0 ?-(x):(x))
#define SQR(x) ((x)*(x))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

template <class T> class vect3d{
	//based on Will Perone 3D vector class, http://willperone.net/Code/vector3.php
public:
	T x, y, z;
	vect3d(){ x = 0; y = 0; z = 0; }
	vect3d(const T X, const T Y, const T Z) { x = X; y = Y; z = Z; }; //constructor
	void operator ()(const T X, const T Y, const T Z) { x = X; y = Y; z = Z; }
	void assign(const T X, const T Y, const T Z) { x = X; y = Y; z = Z; };
	bool operator==(const vect3d<T> &t){ return (x == t.x && y == t.y && z == t.z); };
	bool operator!=(const vect3d<T> &t){ return (x != t.x || y != t.y || z != t.z); };
	const vect3d <T> &operator=(const vect3d <T> &t){
		x = t.x; y = t.y; z = t.z;
		return *this;
	};
	const vect3d<T> operator -(void) const { return vect3d<T>(-x, -y, -z); }
	const vect3d <T> operator+(const vect3d <T> &t) const { return vect3d <T>(x + t.x, y + t.y, z + t.z); };
	const vect3d <T> operator-(const vect3d <T> &t) const { return vect3d <T>(x - t.x, y - t.y, z - t.z); };
	const vect3d <T> &operator+=(const vect3d <T> &t){
		x += t.x;	y += t.y;	z += t.z;
		return *this;
	};
	const vect3d <T> &operator-=(const vect3d <T> &t){
		x -= t.x;	y -= t.y;	z -= t.z;
		return *this;
	};
	const vect3d<T> &operator *=(const T &t){
		x *= t; y *= t; z *= t;
		return *this;
	}
	const vect3d<T> &operator /=(const T &t){
		x /= t; y /= t; z /= t;
		return *this;
	}
	const vect3d<T> operator *(const T &t) const {
		vect3d<T> temp;
		temp = *this;
		return temp *= t;
	}
	const vect3d<T> operator /(const T &t) const {
		vect3d<T> temp;
		temp = *this;
		return temp /= t;
	}

	const vect3d <T> operator*(const vect3d <T> &t) const {
		vect3d <T> temp(y*t.z - z*t.y, -x*t.z + z*t.x, x*t.y - y*t.x);
		return temp;
	};
	T dot(const vect3d <T> &t) const { return x*t.x + y*t.y + z*t.z; }
	T mag() const { return sqrt(SQR(x) + SQR(y) + SQR(z)); }
	T sqr() const { return SQR(x) + SQR(y) + SQR(z); }
	void normalize(){ *this /= mag(); }
	vect3d <T> norm() const { return *this / mag(); }
	void project(const vect3d <T> &t){ *this = t*(this->dot(t) / t.sqr()); }
	vect3d <T> proj(const vect3d <T> &t) const { return t*(this->dot(t) / t.sqr()); }
	const vect3d <T> operator % (const vect3d <T> &t) const {
		vect3d <T> temp(x*t.x, y*t.y, z*t.z);
		return temp;
	}
};
extern vect3d <double> cos(vect3d <double> t);
extern vect3d <float> cos(vect3d <float> t);
extern vect3d <double> sin(vect3d <double> t);
extern vect3d <float> sin(vect3d <float> t);
template <class T> ostream &operator<<(ostream &stream, vect3d <T> t){
	//vect3d output
	stream.setf(ios::scientific);
	stream << t.x << "  ";
	stream << t.y << "  ";
	stream << t.z;
	stream.unsetf(ios::scientific);
	return stream;
};
template <class T> istream &operator>>(istream &stream, vect3d <T> &t){
	//vect3d input
	stream >> t.x >> t.y >> t.z;
	return stream;
};
typedef struct {
//mesh structure
//part of config.calc structure
//min. & max. values of 1D mesh
//N - number of points between min and max
	double min, max;
	unsigned int N;
} mesh;
typedef struct {
	vect3d <double> min, max;
	vect3d <unsigned int> N;
} mesh3d;
typedef struct {
	double occ;
	unsigned int num;
	unsigned int type;
	unsigned int symm;
} atom_info;
typedef struct {
//structure for calculation parameters
//part of config structure
//what will be calculated defined here
//Nblocks - number of different structures in ensemble
//scenario - scenario of calculation (1D Debye or 2D exact calculation)
//Nfi - number of points for fi angle of 2D pattern in 2D exact calculation
//CS[3] - Euler's angles - orientation of ensemble in incident wave vector coordinate system (wave vector along Z)
//CS[3] is not defined in start.xml, defined on every step by elements of theta & fi meshs. Only needed for 2D case/
//q - wavevector mesh
//theta, fi - meshs for avareging over incident vawe vector for 2D case
//AIfilename - filename for form factors table
	unsigned int scenario, source, Nblocks, Nel, Nfi, Nhist, PDFtype, EulerConvention;
	double lambda, hist_bin, qPDF, p0;
	bool PolarFactor, PrintAtoms, PrintPartialPDF, calcPartialIntensity, rearrangement;
	mesh q;
	mesh3d Euler;
	string name, AIfilename;
} config;
class block {
public:
	vector <unsigned int> NatomEl,NatomElAll;
	unsigned int Nsymm, Ncopy, Natom, Nid, N, EulerConvention;
	unsigned int *id;
	string *name;
	vect3d <unsigned int> Ncell;
	vect3d <double> rMean;
	double *occ,*dev,Rcorr,Rcut,RcutCopies, dev_mol;
	bool centered,centeredAtoms,fractional,rearrangement,cutoff,ro_mol,cutoffcopies;
	string CellFile, CopiesFile, NeighbFile;
	vect3d<double> *rAtom, *euler, *cf[3],*rCopy,e[3],*rSymm,*pSymm[3];
	vector< vect3d <double> > *rAtomNeighb;
	vector< string > *nameNeighb;
	vector< unsigned int > *idNeighb;
	block(){
		Natom = 0; Nsymm = 0; Ncopy = 0; N = 0; Ncell.assign(1, 1, 1); Nid = 0;
		Rcorr = 0; Rcut = 0; RcutCopies = 0; dev_mol = 0;
		EulerConvention = EulerZYX;
		e[0].assign(1,0,0);	e[1].assign(0,1,0);	e[2].assign(0,0,1);
		centered = false; centeredAtoms = false; fractional = false; rearrangement = false; cutoff = false; cutoffcopies = false; ro_mol = false;
		id = NULL; name = NULL; rAtom = NULL; euler = NULL; rCopy = NULL; rSymm = NULL; rAtomNeighb = NULL; nameNeighb = NULL; idNeighb = NULL;
		cf[0] = NULL; cf[1] = NULL; cf[2] = NULL; pSymm[0] = NULL; pSymm[1] = NULL; pSymm[2] = NULL, occ=NULL, dev = NULL;
	};
	void create(){
		//id and idNeighb will be created in sortAtoms()
		if (Natom) {
			name=new string[Natom];
			rAtom=new vect3d<double>[Natom];
			occ=new double[Natom];
			dev=new double[Natom];
			for (unsigned int i=0;i<Natom;i++) {name[i]="";occ[i]=1.;dev[i]=0;}
			if (rearrangement) {
				rAtomNeighb=new vector< vect3d <double> >[Natom];
				nameNeighb = new vector <string>[Natom];
			}
		}
		if (Ncopy) {
			rCopy=new vect3d<double>[Ncopy];
			euler=new vect3d<double>[Ncopy];
			for (unsigned int i = 0; i<3; i++) cf[i] = new vect3d<double>[Ncopy];
		}
		if (Nsymm) {
			rSymm=new vect3d<double>[Nsymm];
			for (unsigned int i = 0; i<3; i++) pSymm[i] = new vect3d<double>[Nsymm];
		}
	}
	void redefAtoms();
	void centerAtoms();
	void redefSymm();
	void calcMean();
	void sortAtoms(map <string, unsigned int>);
	int calcAtoms(vector < vect3d <double> > *ra);
	int readCopies();
	int readAtoms(string []);
	int readRearrangement(string[]);
	int printCopies(const char *);
	~block(){
		if (Natom) {
			delete[] id; 
			delete[] rAtom;
			delete[] occ;
			delete[] dev;
			delete[] name;
			if (rearrangement) {
				delete[] rAtomNeighb;
				delete[] idNeighb;
				delete[] nameNeighb;
			}
		}
		if (Ncopy) {
			delete[] rCopy;
			delete[] euler;
			for (unsigned int i = 0; i<3; i++) delete[] cf[i];
		}
		if (Nsymm) {
			delete[] rSymm;
			for (unsigned int i = 0; i<3; i++) delete[] pSymm[i];
		}
	};
};
#endif
