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

//main file
#include "typedefs.h"
#include <chrono>
#ifdef UseMPI
#include "mpi.h"
#endif
#ifdef UseOMP
#include <omp.h>
#endif
#ifdef UseOCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif
struct float4;
int myid,numprocs;
int ReadConfig(config *cfg, double **q, block **Block, const char* FileName, string elements[], map<string, unsigned int> *ID, vector<double> *SL, vector<double *> *FF);
unsigned int CalcAndPrintAtoms(config *cfg, block *Block, vector < vect3d <double> > **ra, unsigned int **NatomEl, map <string, unsigned int> ID);
int printI(const double *I,unsigned int Nq, const double *q, string name, unsigned int source);
int printI2(const double * const *I2D, unsigned int Nq, unsigned int Nfi, string name, unsigned int source);
int printPartialI(const double *I, unsigned int Nq, const double *q, string name, unsigned int source, unsigned int Nblocks);
int printPDF(const double *PDF, unsigned int Nhist, double hist_bin, string name, unsigned int source, unsigned int PDFtype);
int printPartialPDF(const double *PDF, unsigned int Nhist, double hist_bin, string name, unsigned int source, unsigned int PDFtype, unsigned int Nel, map <string, unsigned int> ID);
void RearrangementInt(double *I, const double *q, const config *cfg, const block *Block, vector<double*> FF, vector<double> SL, unsigned int Ntot);
#ifdef UseCUDA
void calcIntDebyeCuda(int DeviceNUM, double **I, const config *cfg, const unsigned int *NatomEl, const float4 *ra, const float * const * dFF, vector<double> SL, const float *dq);
void calcIntPartialDebyeCuda(int DeviceNUM, double **I, const config *cfg, const unsigned int *NatomEl, const float4 *ra, const float * const * dFF, vector<double> SL, const float *dq, const block *Block);
void calcPDFandDebyeCuda(int DeviceNUM, double **I, double **PDF, const config *cfg, const unsigned int *NatomEl, const float4 *ra, const float * const * dFF, vector<double> SL, const float *dq);
void calcInt2DCuda(int DeviceNUM, double ***I2D, double **I, const config *cfg, const unsigned int *NatomEl, const float4 *ra, const float * const * dFF, vector<double> SL, const float *dq);
void delDataFromDevice(float4 *ra, float **dFF, float *dq, unsigned int Nel);
void dataCopyCUDA(const double *q, const config *cfg, const vector < vect3d <double> > *ra, float4 **dra, float ***dFF, float **dq, vector <double*> FF, unsigned int Ntot);
int SetDeviceCuda(int *DeviceNUM);
#elif UseOCL
int GetOpenCLinfoFromInitDataXML(string *GPUtype, int *DeviceNUM); //for BOINC
int GetOpenCLPlatfromNum(string GPUtype, int DeviceNUM); //for BOINC
int SetDeviceOCL(cl_device_id *OCLdevice, int DeviceNUM, int PlatformNUM);
int createContextOCL(cl_context *OCLcontext, cl_program *OCLprogram, cl_device_id OCLdevice, char* argv0, unsigned int scenario);
void dataCopyOCL(cl_context OCLcontext, cl_device_id OCLdevice, const double *q, const config *cfg, const vector < vect3d <double> > *ra, cl_mem *dra, cl_mem **dFF, cl_mem *dq, vector <double*> FF, unsigned int Ntot);
void delDataFromDeviceOCL(cl_context OCLcontext, cl_program OCLprogram, cl_mem ra, cl_mem *dFF, cl_mem dq, unsigned int Nel);
void calcInt2D_OCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double ***I2D, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem dra, const cl_mem *dFF, vector<double> SL, cl_mem dq);
void calcPDFandDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, double **PDF, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq);
void calcIntDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq);
void calcIntPartialDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq, const block *Block);
#else
void calcInt2D(double ***I2D, double **I, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, unsigned int Ntot, int NumOMPthreads);
void calcIntDebye(double **I, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, unsigned int Ntot, int NumOMPthreads);
void calcIntPartialDebye(double **I, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, const block *Block, unsigned int Ntot, int NumOMPthreads);
void calcPDFandDebye(double **I, double **PDF, const config *cfg, const unsigned int *NatomEl, const vector < vect3d <double> > *ra, vector <double*> FF, vector<double> SL, const double *q, unsigned int Ntot, int NumOMPthreads);
#endif
int main(int argc, char *argv[]){
	config cfg;
	int error = 0;
	unsigned int *NatomEl=NULL, Ntot = 0;
	block *Block = NULL;
	vector < vect3d <double> > *ra = NULL;
	double *q = NULL, **I2D = NULL, *I = NULL, *PDF = NULL;
#ifdef UseCUDA
	int DeviceNUM = -1;
	float *dq = NULL, **dFF = NULL;
	float4 *dra = NULL;//using float4 structure for three coordinates to assure for global memory access coalescing
#elif UseOCL
	int DeviceNUM = -1, PlatformNUM=-1;
	cl_mem dq = NULL, *dFF = NULL, dra = NULL;
	cl_context OCLcontext;
	cl_device_id OCLdevice;
	cl_program OCLprogram = NULL;
#else
	int NumOMPthreads=1;
#endif
	string elements[] = { "na", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
		"Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
		"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
		"Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
		"Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
		"Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Uun" };
	map<string, unsigned int> ID;
	vector<double> SL;
	vector<double *> FF;
	chrono::steady_clock::time_point t1, t2;
#ifdef UseMPI
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#else
	myid = 0;
	numprocs = 1;
#endif
	if (argc < 2) {
		if (!myid)	cout << "Error: Start XML filename not specified." << endl;
#ifdef UseMPI
		MPI_Abort(MPI_COMM_WORLD, -1);
#endif
		return -1;
	}
#ifdef UseOMP
	if (argc > 2) NumOMPthreads = atoi(argv[2]);
	else NumOMPthreads = omp_get_max_threads();
	if (!myid)	cout << "Number of OpenMP threads is set to " << NumOMPthreads << endl;
#endif
	error = ReadConfig(&cfg, &q, &Block, argv[1], elements, &ID, &SL, &FF);//reading the calculation parameters, from-factors, etc., creating structural blocks
	if (error) {
		delete[] Block;
#ifdef UseMPI
		MPI_Abort(MPI_COMM_WORLD, -1);
#else
		return -1;
#endif
	}	
	if (!myid) 	t1 = chrono::steady_clock::now();
	Ntot = CalcAndPrintAtoms(&cfg, Block, &ra, &NatomEl, ID);//creating the atomic ensemble
#ifdef UseCUDA
	if (argc > 2) DeviceNUM = atoi(argv[2]);
	error = SetDeviceCuda(&DeviceNUM);//queries CUDA devices, changes DeviceNUM to proper device number if required
	if (error) { delete[] Block; return -1; }
	dataCopyCUDA(q, &cfg, ra, &dra, &dFF, &dq, FF, Ntot);//copying all the necessary data to the device memory
#elif UseOCL
	if (argc > 2) DeviceNUM = atoi(argv[2]);
	if (argc > 3) PlatformNUM = atoi(argv[3]);
	if ((DeviceNUM < 0) && (PlatformNUM < 0)) {//trying to get the OpenCL info from init_data.xml (for compatibility with BOINC client)
		//Note, if something goes wrong here, the programm may start the calculation on the GPU device it is not allowed to...
		string GPUtype;
		error=GetOpenCLinfoFromInitDataXML(&GPUtype,&DeviceNUM);
		if (!error) PlatformNUM = GetOpenCLPlatfromNum(GPUtype, DeviceNUM);
	}
	error = SetDeviceOCL(&OCLdevice, DeviceNUM, PlatformNUM);
	if (error) { delete[] Block; return -1; }
	error = createContextOCL(&OCLcontext, &OCLprogram, OCLdevice, argv[0], cfg.scenario);//copying all the necessary data to the device memory 
	if (error) { delete[] Block; return -1; }
	dataCopyOCL(OCLcontext, OCLdevice, q, &cfg, ra, &dra, &dFF, &dq, FF, Ntot);
#endif
	switch (cfg.scenario) {
		case s2D://calculating the 2D scatteing intensity
#ifdef UseCUDA
			calcInt2DCuda(DeviceNUM, &I2D, &I, &cfg, NatomEl, dra, dFF, SL, dq);
#elif UseOCL
			calcInt2D_OCL(OCLcontext, OCLdevice, OCLprogram, &I2D, &I, &cfg, NatomEl, dra, dFF, SL, dq);
#else
			calcInt2D(&I2D, &I, &cfg, NatomEl, ra, FF, SL, q, Ntot,NumOMPthreads);
#endif
			if (!myid) {
				t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				printI2(I2D, cfg.q.N, cfg.Nfi, cfg.name, cfg.source);
				printI(I, cfg.q.N, q, cfg.name, cfg.source);
			}
			break;
		case Debye://calculating the 1D scatteing intensity using the Debye formula
#ifdef UseCUDA
			if (cfg.calcPartialIntensity) calcIntPartialDebyeCuda(DeviceNUM, &I, &cfg, NatomEl, dra, dFF, SL, dq, Block);
			else calcIntDebyeCuda(DeviceNUM, &I, &cfg, NatomEl, dra, dFF, SL, dq);
#elif UseOCL
			if (cfg.calcPartialIntensity) calcIntPartialDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &cfg, NatomEl, dra, dFF, SL, dq, Block);
			else calcIntDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &cfg, NatomEl, dra, dFF, SL, dq);
#else
			if (cfg.calcPartialIntensity) calcIntPartialDebye(&I, &cfg, NatomEl, ra, FF, SL, q, Block, Ntot,NumOMPthreads);
			else calcIntDebye(&I, &cfg, NatomEl, ra, FF, SL, q, Ntot,NumOMPthreads);
#endif
			if (!myid) {
				if (cfg.rearrangement) RearrangementInt(I,q,&cfg,Block,FF,SL,Ntot);
				t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				if (cfg.calcPartialIntensity) printPartialI(I + cfg.q.N, cfg.q.N, q, cfg.name, cfg.source, cfg.Nblocks);
				printI(I, cfg.q.N, q, cfg.name, cfg.source);
			}
			break;
		case Debye_hist://calculating the 1D scatteing intensity using the histogram of interatomic distances
#ifdef UseCUDA
			calcPDFandDebyeCuda(DeviceNUM, &I, &PDF, &cfg, NatomEl, dra, dFF, SL, dq);
#elif UseOCL
			calcPDFandDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &PDF, &cfg, NatomEl, dra, dFF, SL, dq);
#else
			calcPDFandDebye(&I, &PDF, &cfg, NatomEl, ra, FF, SL, q, Ntot,NumOMPthreads);
#endif
			if (!myid) {
				if (cfg.rearrangement) RearrangementInt(I, q, &cfg, Block, FF, SL, Ntot);
				t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				printI(I, cfg.q.N, q, cfg.name, cfg.source);
			}
			break;
		case PDFonly://calculating the PDF
#ifdef UseCUDA
			calcPDFandDebyeCuda(DeviceNUM, &I, &PDF, &cfg, NatomEl, dra, dFF, SL, dq);
#elif UseOCL
			calcPDFandDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &PDF, &cfg, NatomEl, dra, dFF, SL, dq);
#else
			calcPDFandDebye(&I, &PDF, &cfg, NatomEl, ra, FF, SL, q, Ntot,NumOMPthreads);
#endif
			if (!myid) {
				t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				if (cfg.PrintPartialPDF) printPartialPDF(PDF + cfg.Nhist, cfg.Nhist, cfg.hist_bin, cfg.name, cfg.source, cfg.PDFtype, cfg.Nel, ID);
				printPDF(PDF, cfg.Nhist, cfg.hist_bin, cfg.name,cfg.source,cfg.PDFtype);
			}
			break;
		case DebyePDF://calculating the PDF and 1D scatteing intensity using the histogram of interatomic distances
#ifdef UseCUDA
			calcPDFandDebyeCuda(DeviceNUM, &I, &PDF, &cfg, NatomEl, dra, dFF, SL, dq);
#elif UseOCL
			calcPDFandDebyeOCL(OCLcontext, OCLdevice, OCLprogram, &I, &PDF, &cfg, NatomEl, dra, dFF, SL, dq);
#else
			calcPDFandDebye(&I, &PDF, &cfg, NatomEl, ra, FF, SL, q, Ntot,NumOMPthreads);
#endif
			if (!myid) {
				if (cfg.rearrangement) RearrangementInt(I, q, &cfg, Block, FF, SL, Ntot);
				t2 = chrono::steady_clock::now();
				cout << "Total calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2-t1).count() << " s" << endl;
				printI(I, cfg.q.N, q, cfg.name, cfg.source);
				if (cfg.PrintPartialPDF) printPartialPDF(PDF + cfg.Nhist, cfg.Nhist, cfg.hist_bin, cfg.name, cfg.source, cfg.PDFtype, cfg.Nel, ID);
				printPDF(PDF, cfg.Nhist, cfg.hist_bin, cfg.name, cfg.source, cfg.PDFtype);
			}
			break;
	}
#ifdef UseCUDA
	delDataFromDevice(dra, dFF, dq, cfg.Nel);//deleting the data from the device memory
#elif UseOCL
	delDataFromDeviceOCL(OCLcontext, OCLprogram,dra,dFF,dq,cfg.Nel);
#endif
	if (I2D!=NULL) {
		for (unsigned int iq = 0; iq < cfg.q.N; iq++) delete[] I2D[iq];
		delete[] I2D;
	}
	if (I!=NULL) delete[] I;
	if (q != NULL) delete[] q;
	if (PDF != NULL) delete[] PDF;
	if (cfg.source == xray)	for (unsigned int i=0;i<FF.size();i++)	delete[] FF[i];
	delete[] Block;
	delete[] ra;
	delete[] NatomEl;	
#ifdef UseMPI
	MPI_Finalize();
#endif
	return 0;
}
