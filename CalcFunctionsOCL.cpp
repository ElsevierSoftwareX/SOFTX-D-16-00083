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
#include <chrono>
#ifdef UseOCL
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//Nvidia specific OpenCL extensions
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
/* cl_nv_device_attribute_query extension - no extension #define since it has no functions */
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV       0x4000
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV       0x4001
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV            0x4005
#endif

//AMD specific OpenCL extensions
#ifndef CL_DEVICE_WAVEFRONT_WIDTH_AMD 
/* cl_amd_device_attribute_query extension - no extension #define since it has no functions */
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD               0x4043
#endif

//Calculates rotational matrix (from CalcFunctions.cpp)
void calcRotMatrix(vect3d <double> *cf0, vect3d <double> *cf1, vect3d <double> *cf2, vect3d <double> euler, unsigned int convention);

/**
	Organazies the computations of the 2D scattering intensity in the polar coordinates (q,q_fi) of the reciprocal space with OpenCL

	@param OCLcontext OpenCL context
	@param OCLdevice  OpenCL device
	@param OCLprogram OpenCL program object for OCLcontext
	@param ***I2D     2D scattering intensity array (host). The memory is allocated inside the function.
	@param **I        1D (averaged over the polar angle) scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg       Configuration of simulation parameters
	@param *NatomEl	  Array containing the total number of atoms of each chemical element (host)
	@param ra         Atomic coordinate array (device)
	@param *dFF       X-ray atomic form-factor arrays for all chemical elements (device)
	@param SL         Vector with neutron scattering lengths for all chemical elements
	@param dq         Scattering vector magnitude array (device)
*/
void calcInt2D_OCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double ***I2D, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem *dFF, vector<double> SL, cl_mem dq);

/**
	Organazies the computations of the histogram of interatomic distances with OpenCL

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param OCLprogram  OpenCL program object for OCLcontext
	@param *rij_hist   Histogram of interatomic distances (device). The memory is allocated inside the function.
	@param ra          Atomic coordinate array (device)
	@param *NatomEl    Array containing the total number of atoms of each chemical element (host)
	@param Nel         Total number of different chemical elements in the nanoparticle
	@param Nhist       Size of the partial histogram of interatomic distances
	@param bin         Width of the histogram bin
*/
void calcHistOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, cl_mem *rij_hist, cl_mem ra, const unsigned int *NatomEl, unsigned int Nel, unsigned int Nhist, float bin);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances with OpenCL

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param OCLprogram  OpenCL program object for OCLcontext
	@param **I         Scattering intensity array (host). The memory is allocated inside the function
	@param rij_hist    Histogram of interatomic distances (device).
	@param *NatomEl    Array containing the total number of atoms of each chemical element (host)
	@param *cfg        Configuration of simulation parameters
	@param *dFF        X-ray atomic form-factor arrays for all chemical elements (device)
	@param SL          Vector with neutron scattering lengths for all chemical elements
	@param dq          Scattering vector magnitude array (device)
	@param Ntot        Total number of atoms in the nanoparticle
*/
void calcInt1DHistOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, cl_mem rij_hist, const unsigned int *NatomEl, const config *cfg, const cl_mem * dFF, vector<double> SL, cl_mem dq, unsigned int Ntot);

/**
	Depending on the computational scenario organazies the computations of the scattering intensity (powder diffraction pattern) or PDF using the histogram of interatomic distances with OpenCL

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param OCLprogram  OpenCL program object for OCLcontext
	@param **I         Scattering intensity array (host). The memory is allocated inside the function.
	@param **PDF       PDF array (host). The memory is allocated inside the function.
	@param *cfg        Configuration of simulation parameters
	@param *NatomEl    Array containing the total number of atoms of each chemical element (host)
	@param ra          Atomic coordinate array (device)
	@param *dFF        X-ray atomic form-factor arrays for all chemical elements (device)
	@param SL          Vector with neutron scattering lengths for all chemical elements
	@param dq          Scattering vector magnitude array (device)
*/
void calcPDFandDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, double **PDF, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation) with OpenCL

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param OCLprogram  OpenCL program object for OCLcontext
	@param **I         Scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg        Configuration of simulation parameters
	@param *NatomEl    Array containing the total number of atoms of each chemical element (host)
	@param ra          Atomic coordinate array (device)
	@param *dFF        X-ray atomic form-factor arrays for all chemical elements (device)
	@param SL          Vector with neutron scattering lengths for all chemical elements
	@param dq          Scattering vector magnitude array (device)
*/
void calcIntDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq);

/**
	Organazies the computations of the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation) with OpenCL

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param OCLprogram  OpenCL program object for OCLcontext
	@param **I         Partial + total scattering intensity array (host). The memory is allocated inside the function.
	@param *cfg        Configuration of simulation parameters
	@param *NatomEl    Array containing the total number of atoms of each chemical element (host)
	@param ra          Atomic coordinate array (device)
	@param *dFF        X-ray atomic form-factor arrays for all chemical elements (device)
	@param SL          Vector with neutron scattering lengths for all chemical elements
	@param dq          Scattering vector magnitude array (device)
	@param *Block      Array of the structural blocks
*/
void calcIntPartialDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq, const block *Block);

/**
	Returns BOINC-compatible device vendor name: NVIDIA, intel_gpu or ATI

	@param device_id OpenCL device ID
*/
const char* GetVendor(cl_device_id device_id);

/**
	Returns OpenCL platfrom number for the specified vecndor name and device number. 
	Checks if the device with the given number is present in the OpenCL platfrom. Returns -1 if error

	@param GPUtype   OpenCL device vendor
	@param DeviceNUM OpenCL device number
*/
int GetOpenCLPlatfromNum(string GPUtype, int DeviceNUM);

/**
	Returns the theoretical peak performance of the OpenCL device. Return 0 if the vendor is unsupported.

	@param OCLdevice OpenCL device
	@param show      If True, show the device information on screen
*/
unsigned int GetGFLOPS(cl_device_id OCLdevice, bool show);

/**
	Queries all OpenCL devices in all OpenCL platforms. Checks and sets the OpenCL device
	Returns 0 if OK and -1 if no OpenCL devices found

	@param *OCLdevice  OpenCL device
	@param DeviceNUM   OpenCL device number (-1 if default)
	@param PlatformNUM OpenCL platform number (-1 if default)
*/
int SetDeviceOCL(cl_device_id *OCLdevice, int DeviceNUM, int PlatformNUM);

/**
	Creates OpenCL context and builds OpenCL kernels

	@param *OCLcontext  OpenCL context
	@param *OCLprogram  OpenCL program object
	@param OCLdevice    OpenCL device
	@param *argv0       Absolute path to the executable (first argument in the argv[]). It is used to get the path to the .cl files
	@param scenario     Computational scenario
*/
int createContextOCL(cl_context *OCLcontext, cl_program *OCLprogram, cl_device_id OCLdevice, char* argv0, unsigned int scenario);

/**
	Copies the atomic coordinates (ra), scattering vector magnitude (q) and the x-ray atomic form-factors (FF) to the device memory

	@param OCLcontext  OpenCL context
	@param OCLdevice   OpenCL device
	@param *q          Scattering vector magnitude (host)
	@param *cfg        Configuration of simulation parameters
	@param *ra         Atomic coordinates (host)
	@param *dra        Atomic coordinates (device). The memory is allocated inside the function
	@param **dFF       X-ray atomic form-factors (device). The memory is allocated inside the function
	@param *dq         Scattering vector magnitude (device). The memory is allocated inside the function
	@param FF          X-ray atomic form-factors (host)
	@param Ntot        Total number of atoms in the nanoparticle
*/
void dataCopyOCL(cl_context OCLcontext, cl_device_id OCLdevice, const double *q, const config *cfg, const vector < vect3d <double> > *ra, cl_mem *dra, cl_mem **dFF, cl_mem *dq, vector <double*> FF, unsigned int Ntot);

/**
	Deletes the atomic coordinates (ra), scattering vector magnitude (dq), the x-ray atomic form-factors (dFF) from the device memory and frees the OpenCL context and program object

	@param OCLcontext  OpenCL context
	@param OCLprogram  OpenCL program object
	@param ra          Atomic coordinates (device)
	@param *dFF        X-ray atomic form-factors (device)
	@param dq          Scattering vector magnitude (device)
	@param Nel         Total number of different chemical elements in the nanoparticle
*/
void delDataFromDeviceOCL(cl_context OCLcontext, cl_program OCLprogram, cl_mem ra, cl_mem *dFF, cl_mem dq, unsigned int Nel);

//Returns BOINC-compatible device vendor name: NVIDIA, intel_gpu or ATI
const char* GetVendor(cl_device_id device_id) {
	char *vendor = NULL;
	size_t info_size;
	clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) return "NVIDIA";
	if (vendor_str.find("intel") != string::npos) return "intel_gpu";
	if ((vendor_str.find("amd") != string::npos) || (vendor_str.find("advanced") != string::npos)) return "ATI";
	return "";
}

//Returns OpenCL platfrom number for the specified vecndor name and device number.
//Checks if the device with the given number is present in the OpenCL platfrom.Returns - 1 if error
int GetOpenCLPlatfromNum(string GPUtype, int DeviceNUM) {
	if (DeviceNUM < 0) return -1;
	if (GPUtype.empty()) return -1;
	cl_platform_id *platform_id = NULL;
	cl_uint ret_num_platforms;
	clGetPlatformIDs(0, NULL, &ret_num_platforms);
	if (!ret_num_platforms) {
		cout << "Error: No OpenCL platfroms found." << endl;
		return -1;
	}
	platform_id = new cl_platform_id[ret_num_platforms];
	clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
	for (cl_uint i = 0; i < ret_num_platforms; i++) {
		cl_uint ret_num_devices;
		cl_device_id *device_id;
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &ret_num_devices);
		if (DeviceNUM >= (int)ret_num_devices) {
			continue;
		}
		device_id = new cl_device_id[ret_num_devices];
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, ret_num_devices, device_id, NULL);
		string gpu_type(GetVendor(device_id[DeviceNUM]));
		delete[] device_id;
		if (gpu_type == GPUtype) return i;
	}
	delete[] platform_id;
	return -1;
}

//Returns the theoretical peak performance of the OpenCL device. Return 0 if the vendor is unsupported.
unsigned int GetGFLOPS(cl_device_id OCLdevice, bool show = false){
	cl_uint CU, GPUclock;
	unsigned int GFLOPS = 0;
	char *vendor = NULL, *name = NULL;
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CU, NULL);
	clGetDeviceInfo(OCLdevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &GPUclock, NULL);
	clGetDeviceInfo(OCLdevice, CL_DEVICE_NAME, 0, NULL, &info_size);
	name = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_NAME, info_size, name, NULL);
	if (show) {
		cout << "GPU: " << name << "\n";
		cout << "Number of compute units: " << CU << "\n";
		cout << "GPU clock rate: " << GPUclock << " MHz\n";
	}
	delete[] name;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) {
		cl_uint CCmaj = 0;
		cl_uint CCmin = 0;
		clGetDeviceInfo(OCLdevice, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &CCmaj, NULL);
		clGetDeviceInfo(OCLdevice, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &CCmin, NULL);
		unsigned int cc = CCmaj * 10 + CCmin; //compute capability
		GFLOPS = CU * 128 * 2 * GPUclock / 1000;
		switch (cc){
		case 10:
		case 11:
		case 12:
		case 13:
			GFLOPS = CU * 8 * 2 * GPUclock / 1000;
			break;
		case 20:
			GFLOPS = CU * 32 * 2 * GPUclock / 1000;
			break;
		case 21:
			GFLOPS = CU * 48 * 2 * GPUclock / 1000;
			break;
		case 30:
		case 35:
		case 37:
			GFLOPS = CU * 192 * 2 * GPUclock / 1000;
			break;
		case 50:
		case 52:
		case 61:
			GFLOPS = CU * 128 * 2 * GPUclock / 1000;
			break;
		case 60:
			GFLOPS = CU * 64 * 2 * GPUclock / 1000;
			break;
		}
	}
	else if (vendor_str.find("intel") != string::npos) {
		GFLOPS = CU * 16 * GPUclock / 1000;
	}
	else if ((vendor_str.find("amd") != string::npos) || (vendor_str.find("advanced") != string::npos)) {
		cl_uint WW = 0;
		cl_int ret;
		ret=clGetDeviceInfo(OCLdevice, CL_DEVICE_WAVEFRONT_WIDTH_AMD, sizeof(cl_uint), &WW, NULL);
		if ((ret != CL_SUCCESS) || (!WW)) WW = 64;//if CL_DEVICE_WAVEFRONT_WIDTH_AMD is not implemented
		GFLOPS = CU * WW * 2 * GPUclock / 1000;
	}
	else {
		cout << "Error. Unsupported device vendor." << endl;
		return 0;
	}
	if (show) 	cout << "Theoretical peak performance: " << GFLOPS << " GFLOPs\n" << endl;
	return GFLOPS;
}

//Queries all OpenCL devices in all OpenCL platforms. Checks and sets the OpenCL device
//Returns 0 if OK and - 1 if no OpenCL devices found
int SetDeviceOCL(cl_device_id *OCLdevice, int DeviceNUM, int PlatformNUM){
	unsigned int MaxGFOLPS = 0;
	cl_device_id **device_id = NULL;
	cl_platform_id *platform_id = NULL;
	cl_uint *ret_num_devices = NULL;
	char **p_name = NULL;
	cl_uint ret_num_platforms;
	clGetPlatformIDs(0, NULL, &ret_num_platforms);
	if (!ret_num_platforms) {
		cout << "Error: No OpenCL platfroms found." << endl;
		return -1;
	}
	platform_id = new cl_platform_id[ret_num_platforms];
	device_id = new cl_device_id*[ret_num_platforms];
	ret_num_devices = new cl_uint[ret_num_platforms];
	clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
	p_name = new char*[ret_num_platforms];
	for (cl_uint i = 0; i < ret_num_platforms; i++) {
		size_t info_size = 0;
		clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 0, NULL, &info_size);
		p_name[i] = new char[info_size / sizeof(char)];
		clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, info_size, p_name[i], NULL);
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &ret_num_devices[i]);
		if (!ret_num_devices[i]) {
			device_id[i] = NULL;
			continue;
		}
		device_id[i] = new cl_device_id[ret_num_devices[i]];
		clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, ret_num_devices[i], device_id[i], NULL);
	}
	if (PlatformNUM >= (int)ret_num_platforms) {
		cout << "Error: Unable to set the OpenCL platfrom " << PlatformNUM << ".\n";
		cout << "The total number of available platforms is " << ret_num_platforms << ". Will use the default platform.\n" << endl;
		PlatformNUM = -1;
	}
	if (PlatformNUM > -1) {
		cout << "Selected OpenCL platform:\n" << p_name[PlatformNUM] << endl;
		if (DeviceNUM >= (int)ret_num_devices[PlatformNUM]) {
			cout << "Error: Unable to set the OpenCL device " << DeviceNUM << ".\n";
			cout << "The total number of the devices for the selected platfrom is " << ret_num_devices[PlatformNUM] << ". Will use the fastest device available.\n" << endl;
			DeviceNUM = -1;
		}
		if (DeviceNUM > -1) {
			*OCLdevice = device_id[PlatformNUM][DeviceNUM];
			cout << "Selected OpenCL device:\n";
			GetGFLOPS(*OCLdevice, true);
			for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
			delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
			return 0;
		}
		cout << "Platform contains the following OpenCL devices:\n";
		unsigned int GFOLPS = 0, MaxGFOLPS = 0;
		for (cl_uint i = 0; i< ret_num_devices[PlatformNUM]; i++) {
			cout << "Device " << i << ":\n";
			GFOLPS = GetGFLOPS(device_id[PlatformNUM][i], true);
			if (GFOLPS > MaxGFOLPS) {
				MaxGFOLPS = GFOLPS;
				DeviceNUM = (int)i;
			}
		}
		*OCLdevice = device_id[PlatformNUM][DeviceNUM];
		cout << "Will use device " << DeviceNUM << "." << endl;
		return 0;
	}
	if (DeviceNUM > -1) {
		cout << "Device " << DeviceNUM << " is found in the following OpenCL platforms:\n";
		for (cl_uint i = 0; i < ret_num_platforms; i++) {
			if (DeviceNUM >= (int)ret_num_devices[i]) continue;
			cout << "Platfrom " << i << ": " << p_name[i] << "\n";
			unsigned int GFOLPS = GetGFLOPS(device_id[i][DeviceNUM], false);
			if (GFOLPS > MaxGFOLPS) {
				MaxGFOLPS = GFOLPS;
				PlatformNUM = (int)i;
			}
		}
		if (MaxGFOLPS) {
			*OCLdevice = device_id[PlatformNUM][DeviceNUM];
			cout << "Will use OpenCL platform " << PlatformNUM << "." << endl;
			cout << "Selected OpenCL device:\n";
			GetGFLOPS(*OCLdevice, true);
			for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
			delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
			return 0;
		}
		cout << "Error: No OpenCL platfroms found containing the device with number " << DeviceNUM << ". Will use the fastest device available.\n" << endl;
		DeviceNUM = -1;
	}
	cout << "The following OpenCL platforms are found:\n";
	for (cl_uint i = 0; i < ret_num_platforms; i++) {
		cout << "Platform " << i << ": " << p_name[i] << "\n";
		cout << "Platform contains the following OpenCL devices:\n";
		for (cl_uint j = 0; j < ret_num_devices[i]; j++) {
			cout << "Device " << j << ":\n";
			unsigned int GFOLPS = GetGFLOPS(device_id[i][j], true);
			if (GFOLPS > MaxGFOLPS) {
				MaxGFOLPS = GFOLPS;
				DeviceNUM = (int)j;
				PlatformNUM = (int)i;
			}
		}
	}
	if (MaxGFOLPS) {
		*OCLdevice = device_id[PlatformNUM][DeviceNUM];
		cout << "Will use device " << DeviceNUM << " in the platform " << PlatformNUM << "." << endl;
		for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
		delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
		return 0;
	}
	cout << "Error: No OpenCL devices found." << endl;
	for (cl_uint i = 0; i < ret_num_platforms; i++) { delete[] p_name[i]; delete[] device_id[i]; }
	delete[] p_name; delete[] device_id; delete[] platform_id; delete[] ret_num_devices;
	return -1;
}

//Creates OpenCL context and builds OpenCL kernels
int createContextOCL(cl_context *OCLcontext, cl_program *OCLprogram, cl_device_id OCLdevice, char* argv0, unsigned int scenario){
	cl_int err;
	char *source_str = NULL;
	size_t source_size;	
	string path2kernels(argv0);
	size_t pos = path2kernels.rfind("/");
	if (pos == string::npos) pos = path2kernels.rfind("\\");
	switch (scenario){
	case s2D:
		path2kernels = path2kernels.replace(pos + 1, string::npos, "kernels2D.cl");
		break;
	case Debye:
		path2kernels = path2kernels.replace(pos + 1, string::npos, "kernelsDebye.cl");
		break;
	case Debye_hist:
	case PDFonly:
	case DebyePDF:
		path2kernels = path2kernels.replace(pos + 1, string::npos, "kernelsPDF.cl");
		break;
	}
	ifstream is(path2kernels, ifstream::binary);
	if (is) {
		is.seekg(0, is.end);
		int length = (int)is.tellg();
		is.seekg(0, is.beg);
		source_str = new char[length];
		is.read(source_str, length);
		if (!is) cout << "error" << endl;
		source_size = sizeof(char)*length;
	}
	else {
		cout << "Failed to load OpenCL kernels from file.\n" << endl;
		exit(1);
	}
	size_t info_size = 0;
	char *info = NULL;	
	clGetDeviceInfo(OCLdevice, CL_DEVICE_EXTENSIONS, 0, NULL, &info_size);
	info = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_EXTENSIONS, info_size, info, NULL);
	string extensions(info);
	delete[] info;
	*OCLcontext = clCreateContext(NULL, 1, &OCLdevice, NULL, NULL, NULL);
	*OCLprogram = clCreateProgramWithSource(*OCLcontext, 1, (const char **)&source_str, &source_size, &err);
	if ((extensions.find("cl_khr_int64_base_atomics") != string::npos) || (extensions.find("cl_nv_") != string::npos)) err = clBuildProgram(*OCLprogram, 1, &OCLdevice, "-cl-fast-relaxed-math", NULL, NULL);
	else err = clBuildProgram(*OCLprogram, 1, &OCLdevice, "-cl-fast-relaxed-math -DCustomInt64atomics", NULL, NULL);
	if (err) {
		size_t lengthErr;
		char *buffer;
		clGetProgramBuildInfo(*OCLprogram, OCLdevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &lengthErr);
		buffer = new char[lengthErr];
		clGetProgramBuildInfo(*OCLprogram, OCLdevice, CL_PROGRAM_BUILD_LOG, lengthErr, buffer, NULL);
		cout << "--- Build log ---\n " << buffer << endl;
		delete[] buffer;
	}
	delete[] source_str;
	return 0;
}

//Copies the atomic coordinates (ra), scattering vector magnitude (q) and the x-ray atomic form-factors (FF) to the device memory
void dataCopyOCL(cl_context OCLcontext, cl_device_id OCLdevice, const double *q, const config *cfg, const vector < vect3d <double> > *ra, cl_mem *dra, cl_mem **dFF, cl_mem *dq, vector <double*> FF, unsigned int Ntot){
	cl_bool UMflag = false;
	char *vendor;
	size_t info_size;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
	//copying the main data to the device memory
	if (cfg->scenario != PDFonly) {//we are calculating not only PDFs but the diffraction patterns too
		*dq = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY, cfg->q.N * sizeof(cl_float), NULL, NULL);
		cl_float *qfloat;//temporary float array for the scattering vector magnitude
		if (UMflag) qfloat = (cl_float *)clEnqueueMapBuffer(queue, *dq, true, CL_MAP_READ, 0, cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
		else qfloat = new cl_float[cfg->q.N];
		for (unsigned int iq = 0; iq < cfg->q.N; iq++) qfloat[iq] = (cl_float)q[iq];//converting scattering vector magnitude from double to float
		if (UMflag) clEnqueueUnmapMemObject(queue, *dq, (void *)qfloat, 0, NULL, NULL);
		else {
			clEnqueueWriteBuffer(queue, *dq, true, 0, cfg->q.N * sizeof(cl_float), (void *)qfloat, 0, NULL, NULL);
			delete[] qfloat;
		}
		if (cfg->source == xray) {
			*dFF = new cl_mem[cfg->Nel];//this array will store pointers to the atomic form-factor arrays stored in the device memory
			for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
				(*dFF)[iEl] = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY, cfg->q.N * sizeof(cl_float), NULL, NULL);
				cl_float *FFfloat;//temporary float array for the atomic form-factor
				if (UMflag) FFfloat = (cl_float *)clEnqueueMapBuffer(queue, (*dFF)[iEl], true, CL_MAP_READ, 0, cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
				else FFfloat = new cl_float[cfg->q.N];
				for (unsigned int iq = 0; iq < cfg->q.N; iq++) FFfloat[iq] = (cl_float)FF[iEl][iq];//converting form-factors from double to float
				if (UMflag) clEnqueueUnmapMemObject(queue, (*dFF)[iEl], (void *)FFfloat, 0, NULL, NULL);
				else {
					clEnqueueWriteBuffer(queue, (*dFF)[iEl], true, 0, cfg->q.N * sizeof(cl_float), (void *)FFfloat, 0, NULL, NULL);
					delete[] FFfloat;
				}
			}
		}
	}	
	cl_float4 *hra = NULL;//temporary host array for atomic coordinates
	*dra = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY, Ntot * sizeof(cl_float4), NULL, NULL);
	if (UMflag) hra = (cl_float4 *)clEnqueueMapBuffer(queue, *dra, true, CL_MAP_READ, 0, Ntot * sizeof(cl_float4), 0, NULL, NULL, NULL);
	else hra = new cl_float4[Ntot];
	unsigned int iAtom = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){
		for (vector<vect3d <double> >::const_iterator ri = ra[iEl].begin(); ri != ra[iEl].end(); ri++, iAtom++){//converting atomic coordinates from vect3d <double> to float4
			hra[iAtom].s[0] = (float)ri->x;
			hra[iAtom].s[1] = (float)ri->y;
			hra[iAtom].s[2] = (float)ri->z;
			hra[iAtom].s[3] = 0;
		}
	}
	if (UMflag) clEnqueueUnmapMemObject(queue, *dra, (void *)hra, 0, NULL, NULL);
	else {
		clEnqueueWriteBuffer(queue, *dra, true, 0, Ntot * sizeof(cl_float4), (void *)hra, 0, NULL, NULL);
		delete[] hra;
	}
	clFlush(queue);
	clFinish(queue);	
	clReleaseCommandQueue(queue);	
}

//Deletes the atomic coordinates (ra), scattering vector magnitude (dq), the x-ray atomic form-factors (dFF) from the device memory and frees the OpenCL context and program object
void delDataFromDeviceOCL(cl_context OCLcontext, cl_program OCLprogram, cl_mem ra, cl_mem *dFF, cl_mem dq, unsigned int Nel){
	clReleaseProgram(OCLprogram);
	clReleaseContext(OCLcontext);
	clReleaseMemObject(ra);//deallocating device memory for the atomic coordinates array
	if (dq != NULL) clReleaseMemObject(dq);//deallocating memory for the scattering vector magnitude array
	if (dFF != NULL) {//Xray source
		for (unsigned int i = 0; i < Nel; i++) if (dFF[i] != NULL) clReleaseMemObject(dFF[i]);//deallocating device memory for the atomic form-factors
		delete[] dFF;//deleting pointer array
	}
}

//Organazies the computations of the 2D scattering intensity in the polar coordinates (q,q_fi) of the reciprocal space with OpenCL
void calcInt2D_OCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double ***I2D, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem *dFF, vector<double> SL, cl_mem dq){
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	unsigned int MaxAtomsPerLaunch = 0, BlockSize2D = BlockSize2Dsmall, Ntot = 0;
	cl_bool kernelExecTimeoutEnabled = true, UMflag = false;
	cl_float *hI = NULL;
	cl_mem dI, dAr, dAi, dCS;
	*I = new double[cfg->q.N]; //array for 1d scattering intensity I[q] (I2D[q][fi] averaged over polar angle fi)
	*I2D = new double*[cfg->q.N]; //array for 2d scattering intensity 
	for (unsigned int iq = 0; iq<cfg->q.N; iq++){
		(*I)[iq] = 0;
		(*I2D)[iq] = new double[cfg->Nfi];
	}	
	unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	char *vendor = NULL;
	size_t info_size;
    cl_device_type device_type;
    clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled=false;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	if (kernelExecTimeoutEnabled){ //killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel execution time in seconds
		const double k = 4.e-8; // t = k * MaxAtomsPerLaunch * Nq * Nfi / GFLOPS
		MaxAtomsPerLaunch = (unsigned int)((tmax * GFLOPS) / (k * cfg->q.N * cfg->Nfi)); //maximum number of atoms per kernel launch
	}
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //total number of atoms
	unsigned int Nm = cfg->q.N * cfg->Nfi; //dimension of 2D intensity array
	//allocating memory on the device for amplitude and intensity 2D arrays
	//GPU has linear memory, so we stretch 2D arrays into 1D arrays
	dAr = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Nm * sizeof(cl_float), NULL, NULL);
	dAi = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Nm * sizeof(cl_float), NULL, NULL);
	dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Nm * sizeof(cl_float), NULL, NULL);
	size_t local_work_size[2] = { BlockSize2D, BlockSize2D }; //2d work-group size
	size_t global_work_size[2] = { (cfg->Nfi / BlockSize2D + BOOL(cfg->Nfi % BlockSize2D))*BlockSize2D, (cfg->q.N / BlockSize2D + BOOL(cfg->q.N % BlockSize2D))*BlockSize2D };//2d global work-items numbers
	cl_float4 CS[3];//three rows of the transposed rotational matrix for the host and the device
	cl_uint Nst, Nfin;
	dCS = clCreateBuffer(OCLcontext, CL_MEM_READ_ONLY, 3 * sizeof(cl_float4), NULL, NULL); //allocating the device memory for the transposed rotational matrix
	//creating kernels
	cl_kernel zeroInt2DKernel = clCreateKernel(OCLprogram, "zeroInt2DKernel", NULL); //reseting the 2D intensity matrix
	clSetKernelArg(zeroInt2DKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(zeroInt2DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(zeroInt2DKernel, 2, sizeof(cl_uint), (void *)&cfg->Nfi);
	cl_kernel zeroAmp2DKernel = clCreateKernel(OCLprogram, "zeroAmp2DKernel", NULL); //reseting 2D amplitude arrays
	clSetKernelArg(zeroAmp2DKernel, 0, sizeof(cl_mem), (void *)&dAr);
	clSetKernelArg(zeroAmp2DKernel, 1, sizeof(cl_mem), (void *)&dAi);
	clSetKernelArg(zeroAmp2DKernel, 2, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(zeroAmp2DKernel, 3, sizeof(cl_uint), (void *)&cfg->Nfi);
	cl_kernel calcInt2DKernelXray = NULL, calcInt2DKernelNeutron = NULL;
	cl_float lambdaf = (cl_float)cfg->lambda;
	if (cfg->source == xray) {
		calcInt2DKernelXray = clCreateKernel(OCLprogram, "calcInt2DKernelXray", NULL); //reseting 2D amplitude arrays
		clSetKernelArg(calcInt2DKernelXray, 0, sizeof(cl_mem), (void *)&dAr);
		clSetKernelArg(calcInt2DKernelXray, 1, sizeof(cl_mem), (void *)&dAi);
		clSetKernelArg(calcInt2DKernelXray, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcInt2DKernelXray, 3, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcInt2DKernelXray, 4, sizeof(cl_uint), (void *)&cfg->Nfi);
		clSetKernelArg(calcInt2DKernelXray, 5, sizeof(cl_mem), (void *)&dCS);
		clSetKernelArg(calcInt2DKernelXray, 6, sizeof(cl_float), (void *)&lambdaf);
		clSetKernelArg(calcInt2DKernelXray, 7, sizeof(cl_mem), (void *)&ra);
	}
	else {
		calcInt2DKernelNeutron = clCreateKernel(OCLprogram, "calcInt2DKernelNeutron", NULL); //reseting 2D amplitude arrays
		clSetKernelArg(calcInt2DKernelNeutron, 0, sizeof(cl_mem), (void *)&dAr);
		clSetKernelArg(calcInt2DKernelNeutron, 1, sizeof(cl_mem), (void *)&dAi);
		clSetKernelArg(calcInt2DKernelNeutron, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcInt2DKernelNeutron, 3, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcInt2DKernelNeutron, 4, sizeof(cl_uint), (void *)&cfg->Nfi);
		clSetKernelArg(calcInt2DKernelNeutron, 5, sizeof(cl_mem), (void *)&dCS);
		clSetKernelArg(calcInt2DKernelNeutron, 6, sizeof(cl_float), (void *)&lambdaf);
		clSetKernelArg(calcInt2DKernelNeutron, 7, sizeof(cl_mem), (void *)&ra);
	}
	cl_kernel Sum2DKernel = clCreateKernel(OCLprogram, "Sum2DKernel", NULL); //calculating the 2d scattering intensity using the scattering amplitude
	clSetKernelArg(Sum2DKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(Sum2DKernel, 1, sizeof(cl_mem), (void *)&dAr);
	clSetKernelArg(Sum2DKernel, 2, sizeof(cl_mem), (void *)&dAi);
	clSetKernelArg(Sum2DKernel, 3, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(Sum2DKernel, 4, sizeof(cl_uint), (void *)&cfg->Nfi);
	//creating command queue
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
	clEnqueueNDRangeKernel(queue, zeroInt2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);//reseting the 2D intensity matrix
	//2d scattering intensity should be calculated for the preset orientation of the sample (or averaged over multiple orientations specified by mesh)
	double dalpha = (cfg->Euler.max.x - cfg->Euler.min.x) / cfg->Euler.N.x, dbeta = (cfg->Euler.max.y - cfg->Euler.min.y) / cfg->Euler.N.y, dgamma = (cfg->Euler.max.z - cfg->Euler.min.z) / cfg->Euler.N.z;
	if (cfg->Euler.N.x<2) dalpha = 0;
	if (cfg->Euler.N.y<2) dbeta = 0;
	if (cfg->Euler.N.z<2) dgamma = 0;
	vect3d <double> cf0, cf1, cf2; //three rows of the rotational matrix
	for (unsigned int ia = 0; ia < cfg->Euler.N.x; ia++){
		double alpha = cfg->Euler.min.x + (ia + 0.5)*dalpha;
		for (unsigned int ib = 0; ib < cfg->Euler.N.y; ib++){
			double beta = cfg->Euler.min.y + (ib + 0.5)*dbeta;
			for (unsigned int ig = 0; ig < cfg->Euler.N.z; ig++){
				double gamma = cfg->Euler.min.z + (ig + 0.5)*dgamma;
				vect3d <double> euler(alpha, beta, gamma);
				calcRotMatrix(&cf0, &cf1, &cf2, euler, cfg->EulerConvention); //calculating the rotational matrix
				CS[0].s[0] = (cl_float)cf0.x; CS[0].s[1] = (cl_float)cf1.x; CS[0].s[2] = (cl_float)cf2.x; CS[0].s[3] = 0; //transposing the rotational matrix
				CS[1].s[0] = (cl_float)cf0.y; CS[1].s[1] = (cl_float)cf1.y; CS[1].s[2] = (cl_float)cf2.y; CS[1].s[3] = 0;
				CS[2].s[0] = (cl_float)cf0.z; CS[2].s[1] = (cl_float)cf1.z; CS[2].s[2] = (cl_float)cf2.z; CS[2].s[3] = 0;
				clEnqueueWriteBuffer(queue,dCS,true,0,3*sizeof(cl_float4),CS,0,NULL,NULL);//copying transposed rotational matrix from the host memory to the device memory
				clEnqueueNDRangeKernel(queue, zeroAmp2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL); //reseting 2D amplitude arrays
				unsigned int inp = 0;
				for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++){ //looping over chemical elements (or ions)
					if (MaxAtomsPerLaunch) { //killswitch is enabled so MaxAtomsPerLaunch is set
						for (unsigned int i = 0; i < NatomEl[iEl] / MaxAtomsPerLaunch + BOOL(NatomEl[iEl] % MaxAtomsPerLaunch); i++) { //looping over the iterations
							Nst = inp + i*MaxAtomsPerLaunch; //index for the first atom on the current iteration step
							Nfin = MIN(Nst + MaxAtomsPerLaunch, inp + NatomEl[iEl]) - Nst; //index for the last atom on the current iteration step
							if (cfg->source == xray) {
								clSetKernelArg(calcInt2DKernelXray, 8, sizeof(cl_uint), (void *)&Nst);
								clSetKernelArg(calcInt2DKernelXray, 9, sizeof(cl_uint), (void *)&Nfin);
								clSetKernelArg(calcInt2DKernelXray, 10, sizeof(cl_mem), (void *)&dFF[iEl]);
								clEnqueueNDRangeKernel(queue, calcInt2DKernelXray, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
							}
							else {//neutron scattering
								float SLf = float(SL[iEl]);
								clSetKernelArg(calcInt2DKernelNeutron, 8, sizeof(cl_uint), (void *)&Nst);
								clSetKernelArg(calcInt2DKernelNeutron, 9, sizeof(cl_uint), (void *)&Nfin);
								clSetKernelArg(calcInt2DKernelNeutron, 10, sizeof(cl_float), (void *)&SLf);
								clEnqueueNDRangeKernel(queue, calcInt2DKernelNeutron, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
							}
							clFlush(queue);
							clFinish(queue);
						}
					}
					else { //killswitch is disabled so we execute the kernels for the entire ensemble of atoms
						Nst = inp;
						Nfin = NatomEl[iEl];
						if (cfg->source == xray) { 
							clSetKernelArg(calcInt2DKernelXray, 8, sizeof(cl_uint), (void *)&Nst);
							clSetKernelArg(calcInt2DKernelXray, 9, sizeof(cl_uint), (void *)&Nfin);
							clSetKernelArg(calcInt2DKernelXray, 10, sizeof(cl_mem), (void *)&dFF[iEl]);
							clEnqueueNDRangeKernel(queue, calcInt2DKernelXray, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
						}
						else {//neutron scattering
							cl_float SLf = (cl_float)SL[iEl];
							clSetKernelArg(calcInt2DKernelNeutron, 8, sizeof(cl_uint), (void *)&Nst);
							clSetKernelArg(calcInt2DKernelNeutron, 9, sizeof(cl_uint), (void *)&Nfin);
							clSetKernelArg(calcInt2DKernelNeutron, 10, sizeof(cl_float), (void *)&SLf);
							clEnqueueNDRangeKernel(queue, calcInt2DKernelNeutron, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
						}
					}
					inp += NatomEl[iEl];
				}
				clEnqueueNDRangeKernel(queue, Sum2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);//calculating the 2d scattering intensity by the scattering amplitude
			}
		}
	}
	cl_float norm = 1.f / (Ntot*cfg->Euler.N.x*cfg->Euler.N.y*cfg->Euler.N.z); //normalizing factor
	cl_kernel Norm2DKernel = clCreateKernel(OCLprogram, "Norm2DKernel", NULL); //normalizing the 2d scattering intensity
	clSetKernelArg(Norm2DKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(Norm2DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(Norm2DKernel, 2, sizeof(cl_uint), (void *)&cfg->Nfi);
	clSetKernelArg(Norm2DKernel, 3, sizeof(cl_float), (void *)&norm);
	clEnqueueNDRangeKernel(queue, Norm2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL); //normalizing the 2d scattering intensity
	cl_kernel PolarFactor2DKernel = NULL;
	if (cfg->PolarFactor) { //multiplying the 2d intensity by polar factor
		PolarFactor2DKernel = clCreateKernel(OCLprogram, "PolarFactor2DKernel", NULL); //multiplying the 2d intensity by polarization factor
		clSetKernelArg(PolarFactor2DKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(PolarFactor2DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor2DKernel, 2, sizeof(cl_uint), (void *)&cfg->Nfi);
		clSetKernelArg(PolarFactor2DKernel, 3, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcInt2DKernelNeutron, 4, sizeof(cl_float), (void *)&lambdaf);
		clEnqueueNDRangeKernel(queue, PolarFactor2DKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	}
	//copying the 2d intensity matrix from the device memory to the host memory 
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dI, true, CL_MAP_READ, 0, Nm * sizeof(cl_float), 0, NULL, NULL, NULL); 
	else {
		hI = new cl_float[Nm];
		clEnqueueReadBuffer(queue,dI,true,0, Nm * sizeof(cl_float), (void *) hI, 0, NULL, NULL);
	}
	for (unsigned int iq = 0; iq < cfg->q.N; iq++){
		for (unsigned int ifi = 0; ifi < cfg->Nfi; ifi++)	{
			(*I2D)[iq][ifi] = double(hI[iq*cfg->Nfi + ifi]);
			(*I)[iq] += (*I2D)[iq][ifi]; //calculating the 1d intensity (averaging I2D[q][fi] over the polar angle fi)
		}
		(*I)[iq] /= cfg->Nfi;
	}
	if (UMflag) clEnqueueUnmapMemObject(queue,dI,(void *) hI,0,NULL,NULL);
	else delete[] hI;
	//deallocating the device memory
	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(zeroInt2DKernel);
	clReleaseKernel(zeroAmp2DKernel);
	if (cfg->source == xray) clReleaseKernel(calcInt2DKernelXray);
	else clReleaseKernel(calcInt2DKernelNeutron);
	clReleaseKernel(Sum2DKernel);
	clReleaseKernel(Norm2DKernel);
	if (cfg->PolarFactor) clReleaseKernel(PolarFactor2DKernel);
	clReleaseMemObject(dCS);
	clReleaseMemObject(dAr);
	clReleaseMemObject(dAi);
	clReleaseMemObject(dI);
	clReleaseCommandQueue(queue);
	t2 = chrono::steady_clock::now();
	cout << "2D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}

//rganazies the computations of the histogram of interatomic distances with OpenCL
void calcHistOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, cl_mem *rij_hist, cl_mem ra, const unsigned int *NatomEl, unsigned int Nel, unsigned int Nhist, float bin){
	unsigned int GridSizeExecMax = 2048;
	unsigned int BlockSize = BlockSize1Dsmall, BlockSize2D = BlockSize2Dsmall; //size of the work-groups (256 by default, 16x16)
	unsigned int NhistTotal = (Nel*(Nel + 1)) / 2 * Nhist;//NhistEl - number of partial (Element1<-->Element2) histograms
	cl_bool kernelExecTimeoutEnabled = true;
	unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	char *vendor = NULL;
	size_t info_size;
	cl_device_type device_type;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	if (kernelExecTimeoutEnabled){ //killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 1.e-6; // t = k * GridSizeExecMax^2 * BlockSize2D^2 / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / k) / BlockSize2D), GridSizeExecMax);
	}
	*rij_hist = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, NhistTotal * sizeof(cl_ulong), NULL, NULL);
	//creating kernels
	cl_kernel zeroHistKernel = clCreateKernel(OCLprogram, "zeroHistKernel", NULL); //reseting the histogram array
	clSetKernelArg(zeroHistKernel, 0, sizeof(cl_mem), (void *)rij_hist);
	clSetKernelArg(zeroHistKernel, 1, sizeof(cl_uint), (void *)&NhistTotal);
	cl_kernel calcHistKernel = clCreateKernel(OCLprogram, "calcHistKernel", NULL);
	clSetKernelArg(calcHistKernel, 0, sizeof(cl_mem), (void *)&ra);
	clSetKernelArg(calcHistKernel, 5, sizeof(cl_mem), (void *)rij_hist);
	clSetKernelArg(calcHistKernel, 7, sizeof(cl_float), (void *)&bin);
	unsigned int GSzero = NhistTotal / BlockSize + BOOL(NhistTotal % BlockSize);//Size of the grid for zeroHistKernel (it must not be large than 65535)
	size_t local_work_size_zero = BlockSize;
	size_t global_work_size_zero = GSzero*local_work_size_zero;
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);//Replace 0 with CL_QUEUE_PROFILING_ENABLE for profiling
	clEnqueueNDRangeKernel(queue, zeroHistKernel, 1, NULL, &global_work_size_zero, &local_work_size_zero, 0, NULL, NULL);
	size_t local_work_size[2] = { BlockSize2D, BlockSize2D};//2D thread block size
	unsigned int Nstart = 0, jAtom0, iAtomST = 0;
	cl_uint diag = 0;
	for (unsigned int iEl = 0; iEl < Nel; iAtomST += NatomEl[iEl], iEl++) {
		unsigned int jAtomST = iAtomST;
		for (unsigned int jEl = iEl; jEl < Nel; jAtomST += NatomEl[jEl], jEl++, Nstart += Nhist) {//each time we move to the next pair of elements (iEl,jEl) we also move to the respective part of histogram (Nstart += Nhist)
			clSetKernelArg(calcHistKernel, 6, sizeof(cl_uint), (void *)&Nstart);
			for (unsigned int iAtom = 0; iAtom < NatomEl[iEl]; iAtom += BlockSize2D*GridSizeExecMax){
				unsigned int GridSizeExecY = MIN((NatomEl[iEl] - iAtom) / BlockSize2D + BOOL((NatomEl[iEl] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of the grid on the current step
				unsigned int iMax = iAtomST + MIN(iAtom + BlockSize2D*GridSizeExecY, NatomEl[iEl]);//index of the last i-th (row) atom
				unsigned int i0 = iAtomST + iAtom;
				(iEl == jEl) ? jAtom0 = iAtom : jAtom0 = 0;//loop should exclude subdiagonal grids
				for (unsigned int jAtom = jAtom0; jAtom < NatomEl[jEl]; jAtom += BlockSize2D*GridSizeExecMax){
					unsigned int GridSizeExecX = MIN((NatomEl[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
					unsigned int jMax = jAtomST + MIN(jAtom + BlockSize2D*GridSizeExecX, NatomEl[jEl]);//index of the last j-th (column) atom
					unsigned int j0 = jAtomST + jAtom;
					size_t global_work_size[2] = { BlockSize2D*GridSizeExecX, BlockSize2D*GridSizeExecY };
					(i0 == j0) ? diag = 1 : diag = 0;//checking if we are on the diagonal grid or not
					clSetKernelArg(calcHistKernel, 1, sizeof(cl_uint), (void *)&i0);
					clSetKernelArg(calcHistKernel, 2, sizeof(cl_uint), (void *)&j0);
					clSetKernelArg(calcHistKernel, 3, sizeof(cl_uint), (void *)&iMax);
					clSetKernelArg(calcHistKernel, 4, sizeof(cl_uint), (void *)&jMax);
					clSetKernelArg(calcHistKernel, 8, sizeof(cl_uint), (void *)&diag);
					cl_event event;
					clEnqueueNDRangeKernel(queue, calcHistKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
					if (kernelExecTimeoutEnabled) {						
						//clWaitForEvents(1, &event);
						clFlush(queue);
						clFinish(queue);
						//cl_ulong time_start, time_end;
						//double time;
						//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
						//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
						//time = time_end - time_start;
						//cout << "calcHistKernel execution time is: " << time / 1000000.0 << " ms\n" << endl;
					}
				}
			}
		}
	}	
	clFlush(queue);
	clFinish(queue);
	clReleaseCommandQueue(queue);
	clReleaseKernel(zeroHistKernel);
	clReleaseKernel(calcHistKernel);
}

//Organazies the computations of the scattering intensity (powder diffraction pattern) using the histogram of interatomic distances with OpenCL
void calcInt1DHistOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, cl_mem rij_hist, const unsigned int *NatomEl, const config *cfg, const cl_mem * dFF, vector<double> SL, cl_mem dq, unsigned int Ntot){
	unsigned int BlockSize = BlockSize1Dsmall;//setting the size of the thread blocks to 256 (default)
	cl_float *hI = NULL;//host array for scattering intensity
	cl_mem dI; //device array for scattering intensity
	*I = new double[cfg->q.N];
	unsigned int GridSize = MIN(256, cfg->Nhist / BlockSize + BOOL(cfg->Nhist % BlockSize));
	unsigned int MaxBinsPerBlock = cfg->Nhist / GridSize + BOOL(cfg->Nhist % GridSize);
	cl_bool kernelExecTimeoutEnabled = true, UMflag = false;
	unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	char *vendor = NULL;
	size_t info_size;
	cl_device_type device_type;	
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	if (kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 1.5e-5; // t = k * Nq * MaxBinsPerBlock / GFLOPS
		MaxBinsPerBlock = MIN((unsigned int)(tmax * GFLOPS / (k * cfg->q.N)), MaxBinsPerBlock);
	}
	unsigned int Isize = GridSize*cfg->q.N;//each wotk-group writes to it's own copy of scattering intensity array
	dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Isize * sizeof(cl_float), NULL, NULL);//allocating the device memory for the scattering intensity array
	//creating kernels
	cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL);
	clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&Isize);
	cl_kernel addIKernelXray = NULL, addIKernelNeutron = NULL, calcIntHistKernelXray = NULL, calcIntHistKernelNeutron = NULL;
	cl_float hbin = float(cfg->hist_bin);
	if (cfg->source == xray) {
		addIKernelXray = clCreateKernel(OCLprogram, "addIKernelXray", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernelXray, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernelXray, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		calcIntHistKernelXray = clCreateKernel(OCLprogram, "calcIntHistKernelXray", NULL);
		clSetKernelArg(calcIntHistKernelXray, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(calcIntHistKernelXray, 3, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcIntHistKernelXray, 4, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcIntHistKernelXray, 5, sizeof(cl_mem), (void *)&rij_hist);
		clSetKernelArg(calcIntHistKernelXray, 8, sizeof(cl_uint), (void *)&cfg->Nhist);
		clSetKernelArg(calcIntHistKernelXray, 9, sizeof(cl_uint), (void *)&MaxBinsPerBlock);
		clSetKernelArg(calcIntHistKernelXray, 10, sizeof(cl_uint), (void *)&hbin);
	}
	else {
		addIKernelNeutron = clCreateKernel(OCLprogram, "addIKernelNeutron", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernelNeutron, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernelNeutron, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		calcIntHistKernelNeutron = clCreateKernel(OCLprogram, "calcIntHistKernelNeutron", NULL);
		clSetKernelArg(calcIntHistKernelNeutron, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(calcIntHistKernelNeutron, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcIntHistKernelNeutron, 3, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcIntHistKernelNeutron, 4, sizeof(cl_mem), (void *)&rij_hist);
		clSetKernelArg(calcIntHistKernelNeutron, 7, sizeof(cl_uint), (void *)&cfg->Nhist);
		clSetKernelArg(calcIntHistKernelNeutron, 8, sizeof(cl_uint), (void *)&MaxBinsPerBlock);
		clSetKernelArg(calcIntHistKernelNeutron, 9, sizeof(cl_float), (void *)&hbin);
	}
	unsigned int GSzero = Isize / BlockSize + BOOL(Isize % BlockSize);//grid size for zero1DFloatArrayKernel
	size_t local_work_size = BlockSize;
	size_t global_work_size_zero = GSzero * local_work_size;
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);//Replace 0 with CL_QUEUE_PROFILING_ENABLE for profiling
	clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size, 0, NULL, NULL);//reseting intensity array
	unsigned int Nstart = 0, GSadd = cfg->q.N / BlockSize + BOOL(cfg->q.N % BlockSize);//grid size for addIKernelXray/addIKernelNeutron
	size_t global_work_size_add = GSadd * local_work_size;
	size_t global_work_size = GridSize * local_work_size;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
		if (cfg->source == xray) {
			clSetKernelArg(addIKernelXray, 2, sizeof(cl_mem), (void *)&dFF[iEl]);
			clSetKernelArg(calcIntHistKernelXray, 1, sizeof(cl_mem), (void *)&dFF[iEl]);
			clSetKernelArg(addIKernelXray, 3, sizeof(cl_uint), (void *)&NatomEl[iEl]);
			clEnqueueNDRangeKernel(queue, addIKernelXray, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		else {
			cl_float mult = float(SQR(SL[iEl]) * NatomEl[iEl]);
			clSetKernelArg(addIKernelNeutron, 2, sizeof(cl_float), (void *)&mult);
			clEnqueueNDRangeKernel(queue, addIKernelNeutron, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
			if (cfg->source == xray) clSetKernelArg(calcIntHistKernelXray, 2, sizeof(cl_mem), (void *)&dFF[jEl]);
			else {
				cl_float SLij = (cl_float) (SL[iEl] * SL[jEl]);
				clSetKernelArg(calcIntHistKernelNeutron, 1, sizeof(cl_float), (void *)&SLij);
			}
			for (unsigned int iBin = 0; iBin < cfg->Nhist; iBin += GridSize*MaxBinsPerBlock) {//iterations to avoid killswitch triggering
				cl_event event;
				if (cfg->source == xray) {					
					clSetKernelArg(calcIntHistKernelXray, 6, sizeof(cl_uint), (void *)&Nstart);
					clSetKernelArg(calcIntHistKernelXray, 7, sizeof(cl_uint), (void *)&iBin);
					clEnqueueNDRangeKernel(queue, calcIntHistKernelXray, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event);
				}
				else {//neutron
					clSetKernelArg(calcIntHistKernelNeutron, 5, sizeof(cl_uint), (void *)&Nstart);
					clSetKernelArg(calcIntHistKernelNeutron, 6, sizeof(cl_uint), (void *)&iBin);
					clEnqueueNDRangeKernel(queue, calcIntHistKernelNeutron, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event);
				}
				//clWaitForEvents(1, &event);
				clFlush(queue);
				clFinish(queue);
				//cl_ulong time_start, time_end;
				//double time;
				//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
				//clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
				//time = time_end - time_start;
				//cout << "calcIntHistKernel execution time is: " << time / 1000000.0 << " ms\n" << endl;
			}
		}
	}
	cl_kernel sumIKernel = clCreateKernel(OCLprogram, "sumIKernel", NULL); //summing intensity copies
	clSetKernelArg(sumIKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(sumIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(sumIKernel, 2, sizeof(cl_uint), (void *)&GridSize);
	clEnqueueNDRangeKernel(queue, sumIKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);//summing intensity copies
	cl_kernel PolarFactor1DKernel = NULL;
	if (cfg->PolarFactor) {
		PolarFactor1DKernel = clCreateKernel(OCLprogram, "PolarFactor1DKernel", NULL);
		cl_float lambdaf = float(cfg->lambda);
		clSetKernelArg(PolarFactor1DKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(PolarFactor1DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor1DKernel, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(PolarFactor1DKernel, 3, sizeof(cl_float), (void *)&lambdaf);
		clEnqueueNDRangeKernel(queue, PolarFactor1DKernel, 1, NULL, &global_work_size_add, &local_work_size, 0, NULL, NULL);
	}	
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dI, true, CL_MAP_READ, 0, cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
	else {
		hI = new cl_float[cfg->q.N];
		clEnqueueReadBuffer(queue, dI, true, 0, cfg->q.N * sizeof(cl_float), (void *)hI, 0, NULL, NULL);
	}
	for (unsigned int iq = 0; iq < cfg->q.N; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing	
	if (UMflag) clEnqueueUnmapMemObject(queue, dI, (void *)hI, 0, NULL, NULL);
	else delete[] hI;
	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(zero1DFloatArrayKernel);
	clReleaseKernel(sumIKernel);
	if (cfg->source == xray) {
		clReleaseKernel(addIKernelXray);
		clReleaseKernel(calcIntHistKernelXray);
	}
	else {
		clReleaseKernel(addIKernelNeutron);
		clReleaseKernel(calcIntHistKernelNeutron);
	}
	if (cfg->PolarFactor) clReleaseKernel(PolarFactor1DKernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(dI);	
}

//Depending on the computational scenario organazies the computations of the scattering intensity(powder diffraction pattern) or PDF using the histogram of interatomic distances with OpenCL
void calcPDFandDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, double **PDF, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq) {
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	cl_mem rij_hist;//array for pair-distribution histogram (device only)
	calcHistOCL(OCLcontext, OCLdevice, OCLprogram, &rij_hist, ra, NatomEl, cfg->Nel, cfg->Nhist, float(cfg->hist_bin));//calculating the histogram
	t2 = chrono::steady_clock::now();
	cout << "Histogram calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl];//calculating the total number of atoms
	if (cfg->scenario > Debye_hist) {//calculating the PDFs
		t1 = chrono::steady_clock::now();
		unsigned int BlockSize = BlockSize1Dsmall;
		unsigned int NPDF = (1 + (cfg->Nel * (cfg->Nel + 1)) / 2) * cfg->Nhist, NPDFh = NPDF;//total PDF array size (full (cfg->Nhist) + partial (cfg->Nhist*(cfg->Nel*(cfg->Nel + 1)) / 2) )
		if (!cfg->PrintPartialPDF) NPDFh = cfg->Nhist;//if the partial PDFs are not needed, we are not copying them to the host
		*PDF = new double[NPDFh];//resulting array of doubles for PDF
		cl_float *hPDF = NULL;
		cl_mem dPDF;
		dPDF = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, NPDF * sizeof(cl_float), NULL, NULL);//allocating the device memory for PDF array
		float Faverage2 = 0;
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			Faverage2 += float(SL[iEl] * NatomEl[iEl]); //calculating the average form-factor
		}
		Faverage2 /= Ntot;
		Faverage2 *= Faverage2;//and squaring it
		//the size of the histogram array may exceed the maximum number of wotk-groups in the grid multiplied by the work-group size 
		//so any operations on histogram array should be performed iteratively
		unsigned int GSzero = NPDF / BlockSize + BOOL(NPDF % BlockSize);
		unsigned int GridSize = cfg->Nhist / BlockSize + BOOL(cfg->Nhist % BlockSize);//grid size for zero1DFloatArrayKernel and other kernels
		cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL); 
		clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dPDF);
		clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&NPDF);
		cl_uint Nstart = 0, iEnd = cfg->Nhist;
		cl_float hbin = (cl_float)cfg->hist_bin;
		cl_kernel calcPartialRDFkernel = NULL, calcPartialPDFkernel = NULL, calcPartialRPDFkernel = NULL;
		switch (cfg->PDFtype){
		case typeRDF://calculating partial RDFs
			calcPartialRDFkernel = clCreateKernel(OCLprogram, "calcPartialRDFkernel", NULL);
			clSetKernelArg(calcPartialRDFkernel, 0, sizeof(cl_mem), (void *)&dPDF);
			clSetKernelArg(calcPartialRDFkernel, 1, sizeof(cl_mem), (void *)&rij_hist);
			clSetKernelArg(calcPartialRDFkernel, 3, sizeof(cl_uint), (void *)&iEnd);
			break;
		case typePDF://calculating partial PDFs
			calcPartialPDFkernel = clCreateKernel(OCLprogram, "calcPartialPDFkernel", NULL);
			clSetKernelArg(calcPartialPDFkernel, 0, sizeof(cl_mem), (void *)&dPDF);
			clSetKernelArg(calcPartialPDFkernel, 1, sizeof(cl_mem), (void *)&rij_hist);
			clSetKernelArg(calcPartialPDFkernel, 3, sizeof(cl_uint), (void *)&iEnd);
			clSetKernelArg(calcPartialPDFkernel, 5, sizeof(cl_float), (void *)&hbin);
			break;
		case typeRPDF://calculating partial rPDFs
			calcPartialRPDFkernel = clCreateKernel(OCLprogram, "calcPartialRPDFkernel", NULL);
			clSetKernelArg(calcPartialRPDFkernel, 0, sizeof(cl_mem), (void *)&dPDF);
			clSetKernelArg(calcPartialRPDFkernel, 1, sizeof(cl_mem), (void *)&rij_hist);
			clSetKernelArg(calcPartialRPDFkernel, 3, sizeof(cl_uint), (void *)&iEnd);
			clSetKernelArg(calcPartialRPDFkernel, 6, sizeof(cl_float), (void *)&hbin);
			break;
		}
		cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
		size_t local_work_size = BlockSize;
		size_t global_work_size_zero = GSzero*local_work_size;
		size_t global_work_size = GridSize*local_work_size;
		clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size, 0, NULL, NULL);//reseting the PDF array
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
				cl_float mult, sub;
				switch (cfg->PDFtype){
				case typeRDF://calculating partial RDFs
					mult = 2.f / (float(cfg->hist_bin)*Ntot);
					clSetKernelArg(calcPartialRDFkernel, 2, sizeof(cl_uint), (void *)&Nstart);
					clSetKernelArg(calcPartialRDFkernel, 4, sizeof(cl_float), (void *)&mult);
					clEnqueueNDRangeKernel(queue, calcPartialRDFkernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
					break;
				case typePDF://calculating partial PDFs
					mult = 0.5f / (PIf*float(cfg->hist_bin*cfg->p0)*Ntot);
					clSetKernelArg(calcPartialPDFkernel, 2, sizeof(cl_uint), (void *)&Nstart);
					clSetKernelArg(calcPartialPDFkernel, 4, sizeof(cl_float), (void *)&mult);
					clEnqueueNDRangeKernel(queue, calcPartialPDFkernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
					break;
				case typeRPDF://calculating partial rPDFs
					mult = 2.f / (float(cfg->hist_bin)*Ntot);
					(jEl > iEl) ? sub = 8.f*PIf*float(cfg->p0)*float(NatomEl[iEl])* float(NatomEl[jEl]) / SQR(float(Ntot)) : sub = 4.f*PIf*float(cfg->p0)*SQR(float(NatomEl[iEl])) / SQR(float(Ntot));
					clSetKernelArg(calcPartialRPDFkernel, 2, sizeof(cl_uint), (void *)&Nstart);
					clSetKernelArg(calcPartialRPDFkernel, 4, sizeof(cl_float), (void *)&mult);
					clSetKernelArg(calcPartialRPDFkernel, 5, sizeof(cl_float), (void *)&sub);
					clEnqueueNDRangeKernel(queue, calcPartialRPDFkernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
					break;
				}
			}
		}
		Nstart = cfg->Nhist;
		cl_kernel calcPDFkernel = clCreateKernel(OCLprogram, "calcPDFkernel", NULL);
		clSetKernelArg(calcPDFkernel, 0, sizeof(cl_mem), (void *)&dPDF);
		clSetKernelArg(calcPDFkernel, 2, sizeof(cl_uint), (void *)&iEnd);
		for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) {//calculating full PDF by summing partial PDFs
			for (unsigned int jEl = iEl; jEl < cfg->Nel; jEl++, Nstart += cfg->Nhist){
				cl_float multIJ = (cl_float)(SL[iEl] * SL[jEl]) / Faverage2;
				clSetKernelArg(calcPDFkernel, 1, sizeof(cl_uint), (void *)&Nstart);
				clSetKernelArg(calcPDFkernel, 3, sizeof(cl_float), (void *)&multIJ);
				clEnqueueNDRangeKernel(queue, calcPDFkernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
			}
		}
		cl_bool UMflag = false; 
		char *vendor = NULL;
		size_t info_size;
		clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
		vendor = new char[info_size / sizeof(char)];
		clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
		string vendor_str(vendor);
		delete[] vendor;
		transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
		if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
		if (UMflag) hPDF = (cl_float *)clEnqueueMapBuffer(queue, dPDF, true, CL_MAP_READ, 0, NPDFh * sizeof(cl_float), 0, NULL, NULL, NULL);
		else {//copying the PDF from the device to the host
			hPDF = new cl_float[NPDFh];
			clEnqueueReadBuffer(queue, dPDF, true, 0, NPDFh * sizeof(cl_float), (void *)hPDF, 0, NULL, NULL);
		}
		for (unsigned int i = 0; i < NPDFh; i++) (*PDF)[i] = double(hPDF[i]);//converting into double
		if (UMflag) clEnqueueUnmapMemObject(queue, dPDF, (void *)hPDF, 0, NULL, NULL);
		else delete[] hPDF;
		clFlush(queue);
		clFinish(queue);
		clReleaseKernel(zero1DFloatArrayKernel);
		clReleaseKernel(calcPDFkernel);
		switch (cfg->PDFtype){
		case typeRDF:
			clReleaseKernel(calcPartialRDFkernel);
			break;
		case typePDF:
			clReleaseKernel(calcPartialPDFkernel);
			break;
		case typeRPDF:
			clReleaseKernel(calcPartialRPDFkernel);
			break;
		}
		clReleaseCommandQueue(queue);
		clReleaseMemObject(dPDF);
		t2 = chrono::steady_clock::now();
		cout << "PDF calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	if ((cfg->scenario == Debye_hist) || (cfg->scenario == DebyePDF)) {
		t1 = chrono::steady_clock::now();
		calcInt1DHistOCL(OCLcontext, OCLdevice, OCLprogram, I, rij_hist, NatomEl, cfg, dFF, SL, dq, Ntot);//calculating the scattering intensity using the pair-distribution histogram
		t2 = chrono::steady_clock::now();
		cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
	}
	clReleaseMemObject(rij_hist);//deallocating memory for pair distribution histogram
}

//Organazies the computations of the scattering intensity (powder diffraction pattern) using the original Debye equation (without the histogram approximation) with OpenCL
void calcIntDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq){
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	unsigned int BlockSize2D = BlockSize2Dsmall;//setting block size to 16x16 (default)
	cl_mem dI;
	float *hI = NULL; //host and device arrays for scattering intensity
	unsigned int Ntot = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //calculating total number of atoms	
	unsigned int GridSizeExecMax = 64;
	unsigned int BlockSize = SQR(BlockSize2D);//total number of threads per block
	cl_bool kernelExecTimeoutEnabled = true, UMflag = false;
	unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	char *vendor = NULL;
	size_t info_size;
	cl_device_type device_type;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	if (kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 * cfg->q.N / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}
	unsigned int Isize = SQR(GridSizeExecMax)*cfg->q.N;//total size of the intensity array
	dI = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, Isize * sizeof(cl_float), NULL, NULL);//allocating the device memory for the scattering intensity array
	cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL);
	clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&Isize);
	cl_kernel addIKernelXray = NULL, addIKernelNeutron = NULL, calcIntDebyeKernelXray = NULL, calcIntDebyeKernelNeutron = NULL;
	if (cfg->source == xray) {
		addIKernelXray = clCreateKernel(OCLprogram, "addIKernelXray", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernelXray, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernelXray, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		calcIntDebyeKernelXray = clCreateKernel(OCLprogram, "calcIntDebyeKernelXray", NULL);
		clSetKernelArg(calcIntDebyeKernelXray, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(calcIntDebyeKernelXray, 3, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcIntDebyeKernelXray, 4, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcIntDebyeKernelXray, 5, sizeof(cl_mem), (void *)&ra);
	}
	else {
		addIKernelNeutron = clCreateKernel(OCLprogram, "addIKernelNeutron", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernelNeutron, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(addIKernelNeutron, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		calcIntDebyeKernelNeutron = clCreateKernel(OCLprogram, "calcIntDebyeKernelNeutron", NULL);
		clSetKernelArg(calcIntDebyeKernelNeutron, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(calcIntDebyeKernelNeutron, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcIntDebyeKernelNeutron, 3, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcIntDebyeKernelNeutron, 4, sizeof(cl_mem), (void *)&ra);
	}
	unsigned int GSzero = Isize / BlockSize + BOOL(Isize % BlockSize);//grid size for zero1DFloatArrayKernel
	size_t local_work_size_zero = BlockSize, local_work_size[2] = { BlockSize2D, BlockSize2D};
	size_t global_work_size_zero = GSzero * local_work_size_zero;
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
	clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size_zero, 0, NULL, NULL);//reseting intensity array
	unsigned int GSadd = cfg->q.N / BlockSize + BOOL(cfg->q.N % BlockSize);//grid size for addIKernelXray/addIKernelNeutron
	size_t global_work_size_add = GSadd * local_work_size_zero;
	unsigned int iAtomST = 0, jAtom0;//grid size for addIKernelXray/addIKernelNeutron
	cl_uint diag = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iAtomST += NatomEl[iEl], iEl++) {
		if(cfg->source == xray) {
			clSetKernelArg(addIKernelXray, 2, sizeof(cl_mem), (void *)&dFF[iEl]);
			clSetKernelArg(calcIntDebyeKernelXray, 1, sizeof(cl_mem), (void *)&dFF[iEl]);
			clSetKernelArg(addIKernelXray, 3, sizeof(cl_uint), (void *)&NatomEl[iEl]);
			clEnqueueNDRangeKernel(queue, addIKernelXray, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		else {
			cl_float mult = float(SQR(SL[iEl]) * NatomEl[iEl]);
			clSetKernelArg(addIKernelNeutron, 2, sizeof(cl_float), (void *)&mult);
			clEnqueueNDRangeKernel(queue, addIKernelNeutron, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
		}
		unsigned int jAtomST = iAtomST;
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jAtomST += NatomEl[jEl], jEl++) {
			if (cfg->source == xray) clSetKernelArg(calcIntDebyeKernelXray, 2, sizeof(cl_mem), (void *)&dFF[jEl]);
			else {
				cl_float SLij = (cl_float)(SL[iEl] * SL[jEl]);
				clSetKernelArg(calcIntDebyeKernelNeutron, 1, sizeof(cl_float), (void *)&SLij);
			}
			for (unsigned int iAtom = 0; iAtom < NatomEl[iEl]; iAtom += BlockSize2D * GridSizeExecMax){
				unsigned int GridSizeExecY = MIN((NatomEl[iEl] - iAtom) / BlockSize2D + BOOL((NatomEl[iEl] - iAtom) % BlockSize2D), GridSizeExecMax);//Y-size of the grid on the current step
				unsigned int iMax = iAtomST + MIN(iAtom + BlockSize2D * GridSizeExecY, NatomEl[iEl]);//index of the last i-th (row) atom
				unsigned int i0 = iAtomST + iAtom;
				(iEl == jEl) ? jAtom0 = iAtom : jAtom0 = 0;//loop should exclude subdiagonal grids
				for (unsigned int jAtom = jAtom0; jAtom < NatomEl[jEl]; jAtom += BlockSize2D * GridSizeExecMax){
					unsigned int GridSizeExecX = MIN((NatomEl[jEl] - jAtom) / BlockSize2D + BOOL((NatomEl[jEl] - jAtom) % BlockSize2D), GridSizeExecMax);//X-size of the grid on the current step
					unsigned int jMax = jAtomST + MIN(jAtom + BlockSize2D * GridSizeExecX, NatomEl[jEl]);//index of the last j-th (column) atom
					unsigned int j0 = jAtomST + jAtom;
					size_t global_work_size[2] = { BlockSize2D * GridSizeExecX, BlockSize2D * GridSizeExecY };
					(i0 == j0) ? diag = 1 : diag = 0;//checking if we are on the diagonal grid or not
					if (cfg->source == xray) {
						clSetKernelArg(calcIntDebyeKernelXray, 6, sizeof(cl_uint), (void *)&i0);
						clSetKernelArg(calcIntDebyeKernelXray, 7, sizeof(cl_uint), (void *)&j0);
						clSetKernelArg(calcIntDebyeKernelXray, 8, sizeof(cl_uint), (void *)&iMax);
						clSetKernelArg(calcIntDebyeKernelXray, 9, sizeof(cl_uint), (void *)&jMax);
						clSetKernelArg(calcIntDebyeKernelXray, 10, sizeof(cl_uint), (void *)&diag);
						clEnqueueNDRangeKernel(queue, calcIntDebyeKernelXray, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
					}
					else {//neutron
						clSetKernelArg(calcIntDebyeKernelNeutron, 5, sizeof(cl_uint), (void *)&i0);
						clSetKernelArg(calcIntDebyeKernelNeutron, 6, sizeof(cl_uint), (void *)&j0);
						clSetKernelArg(calcIntDebyeKernelNeutron, 7, sizeof(cl_uint), (void *)&iMax);
						clSetKernelArg(calcIntDebyeKernelNeutron, 8, sizeof(cl_uint), (void *)&jMax);
						clSetKernelArg(calcIntDebyeKernelNeutron, 9, sizeof(cl_uint), (void *)&diag);
						clEnqueueNDRangeKernel(queue, calcIntDebyeKernelNeutron, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
					}
					if (kernelExecTimeoutEnabled) {
						clFlush(queue);
						clFinish(queue);
					}
				}
			}
		}
	}
	cl_uint Ncopies = SQR(GridSizeExecMax);
	cl_kernel sumIKernel = clCreateKernel(OCLprogram, "sumIKernel", NULL); //summing intensity copies
	clSetKernelArg(sumIKernel, 0, sizeof(cl_mem), (void *)&dI);
	clSetKernelArg(sumIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(sumIKernel, 2, sizeof(cl_uint), (void *)&Ncopies);
	clEnqueueNDRangeKernel(queue, sumIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//summing intensity copies
	cl_kernel PolarFactor1DKernel = NULL;
	if (cfg->PolarFactor) {
		PolarFactor1DKernel = clCreateKernel(OCLprogram, "PolarFactor1DKernel", NULL);
		cl_float lambdaf = float(cfg->lambda);
		clSetKernelArg(PolarFactor1DKernel, 0, sizeof(cl_mem), (void *)&dI);
		clSetKernelArg(PolarFactor1DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor1DKernel, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(PolarFactor1DKernel, 3, sizeof(cl_float), (void *)&lambdaf);
		clEnqueueNDRangeKernel(queue, PolarFactor1DKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);
	}
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dI, true, CL_MAP_READ, 0, cfg->q.N * sizeof(cl_float), 0, NULL, NULL, NULL);
	else {
		hI = new cl_float[cfg->q.N];
		clEnqueueReadBuffer(queue, dI, true, 0, cfg->q.N * sizeof(cl_float), (void *)hI, 0, NULL, NULL);
	}
	*I = new double[cfg->q.N];
	for (unsigned int iq = 0; iq<cfg->q.N; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing	
	if (UMflag) clEnqueueUnmapMemObject(queue, dI, (void *)hI, 0, NULL, NULL);
	else delete[] hI;
	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(zero1DFloatArrayKernel);
	clReleaseKernel(sumIKernel);
	if (cfg->source == xray) {
		clReleaseKernel(addIKernelXray);
		clReleaseKernel(calcIntDebyeKernelXray);
	}
	else {
		clReleaseKernel(addIKernelNeutron);
		clReleaseKernel(calcIntDebyeKernelNeutron);
	}
	if (cfg->PolarFactor) clReleaseKernel(PolarFactor1DKernel);
	clReleaseCommandQueue(queue);
	clReleaseMemObject(dI);
	t2 = chrono::steady_clock::now();
	cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}

//Organazies the computations of the scattering intensity(powder diffraction pattern) using the original Debye equation(without the histogram approximation) with OpenCL
void calcIntPartialDebyeOCL(cl_context OCLcontext, cl_device_id OCLdevice, cl_program OCLprogram, double **I, const config *cfg, const unsigned int *NatomEl, cl_mem ra, const cl_mem * dFF, vector<double> SL, cl_mem dq, const block *Block){
	chrono::steady_clock::time_point t1, t2;
	t1 = chrono::steady_clock::now();
	unsigned int BlockSize2D = BlockSize2Dsmall, BlockSize=SQR(BlockSize2Dsmall);//setting block size to 16x16 (default)
	unsigned int Nparts = (cfg->Nblocks * (cfg->Nblocks + 1)) / 2;
	cl_mem *dI = NULL, dIpart;
	float *hI = NULL; //host and device arrays for scattering intensity
	dI = new cl_mem[Nparts];
	unsigned int Ntot = 0, *NatomElBlock;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iEl++) Ntot += NatomEl[iEl]; //calculating total number of atoms
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
	unsigned int GridSizeExecMax = 64;
	cl_bool kernelExecTimeoutEnabled = true, UMflag = false;
	unsigned int GFLOPS = GetGFLOPS(OCLdevice); //theoretical peak GPU performance
	char *vendor = NULL;
	size_t info_size;
	cl_device_type device_type;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);//checking if device is dedicated accelerator
	if (device_type == CL_DEVICE_TYPE_ACCELERATOR) kernelExecTimeoutEnabled = false;
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, 0, NULL, &info_size);
	vendor = new char[info_size / sizeof(char)];
	clGetDeviceInfo(OCLdevice, CL_DEVICE_VENDOR, info_size, vendor, NULL);
	string vendor_str(vendor);
	delete[] vendor;
	transform(vendor_str.begin(), vendor_str.end(), vendor_str.begin(), ::tolower);
	if (vendor_str.find("intel") != string::npos) clGetDeviceInfo(OCLdevice, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &UMflag, NULL);//checking if GPU is integrated or not
	if (vendor_str.find("nvidia") != string::npos) {
		clGetDeviceInfo(OCLdevice, CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV, sizeof(cl_bool), &kernelExecTimeoutEnabled, NULL);
	}
	if (kernelExecTimeoutEnabled)	{//killswitch is enabled, so the time limit should not be exceeded
		const double tmax = 0.02; //maximum kernel time execution in seconds
		const double k = 5.e-8; // t = k * GridSizeExecMax^2 * BlockSize2D^2 * cfg->q.N / GFLOPS
		GridSizeExecMax = MIN((unsigned int)(sqrt(tmax * GFLOPS / (k * cfg->q.N)) / BlockSize2D), GridSizeExecMax);
	}
	cl_command_queue queue = clCreateCommandQueue(OCLcontext, OCLdevice, 0, NULL);
	unsigned int IsizeBlock = SQR(GridSizeExecMax)*cfg->q.N;//total size of the intensity array
	cl_kernel zero1DFloatArrayKernel = clCreateKernel(OCLprogram, "zero1DFloatArrayKernel", NULL);
	clSetKernelArg(zero1DFloatArrayKernel, 1, sizeof(cl_uint), (void *)&IsizeBlock);
	unsigned int GSzero = IsizeBlock / BlockSize + BOOL(IsizeBlock % BlockSize);//grid size for zero1DFloatArrayKernel
	size_t local_work_size_zero = BlockSize, local_work_size[2] = { BlockSize2D, BlockSize2D};
	size_t global_work_size_zero = GSzero*local_work_size_zero;
	for (unsigned int ipart=0; ipart<Nparts; ipart++) {
		dI[ipart] = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, IsizeBlock * sizeof(cl_float), NULL, NULL);//allocating the device memory for the scattering intensity array
		clSetKernelArg(zero1DFloatArrayKernel, 0, sizeof(cl_mem), (void *)&dI[ipart]);
		clEnqueueNDRangeKernel(queue, zero1DFloatArrayKernel, 1, NULL, &global_work_size_zero, &local_work_size_zero, 0, NULL, NULL);//reseting intensity array
	}
	cl_kernel addIKernelXray = NULL, addIKernelNeutron = NULL, calcIntDebyeKernelXray = NULL, calcIntDebyeKernelNeutron = NULL;
	if (cfg->source == xray) {
		addIKernelXray = clCreateKernel(OCLprogram, "addIKernelXray", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernelXray, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		calcIntDebyeKernelXray = clCreateKernel(OCLprogram, "calcIntDebyeKernelXray", NULL);
		clSetKernelArg(calcIntDebyeKernelXray, 3, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcIntDebyeKernelXray, 4, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcIntDebyeKernelXray, 5, sizeof(cl_mem), (void *)&ra);
	}
	else {
		addIKernelNeutron = clCreateKernel(OCLprogram, "addIKernelNeutron", NULL); //add contribution form diagonal (i==j) elements in Debye sum
		clSetKernelArg(addIKernelNeutron, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		calcIntDebyeKernelNeutron = clCreateKernel(OCLprogram, "calcIntDebyeKernelNeutron", NULL);
		clSetKernelArg(calcIntDebyeKernelNeutron, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(calcIntDebyeKernelNeutron, 3, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(calcIntDebyeKernelNeutron, 4, sizeof(cl_mem), (void *)&ra);
	}
	unsigned int GSadd = cfg->q.N / BlockSize + BOOL(cfg->q.N % BlockSize);//grid size for addIKernelXray/addIKernelNeutron
	size_t global_work_size_add = GSadd*local_work_size_zero;
	unsigned int iAtomST = 0, Ipart;
	cl_uint diag = 0;
	for (unsigned int iEl = 0; iEl < cfg->Nel; iAtomST += NatomEl[iEl], iEl++) {
		if(cfg->source == xray) {
			clSetKernelArg(addIKernelXray, 2, sizeof(cl_mem), (void *)&dFF[iEl]);
			clSetKernelArg(calcIntDebyeKernelXray, 1, sizeof(cl_mem), (void *)&dFF[iEl]);
		}
		for (unsigned int iB = 0; iB < cfg->Nblocks; iB++){
			Ipart = cfg->Nblocks * iB - (iB * (iB + 1)) / 2 +iB;
			if(cfg->source == xray) {
				clSetKernelArg(addIKernelXray, 0, sizeof(cl_mem), (void *)&dI[Ipart]);
				clSetKernelArg(addIKernelXray, 3, sizeof(cl_uint), (void *)&NatomElBlock[iEl * cfg->Nblocks + iB]);
				clEnqueueNDRangeKernel(queue, addIKernelXray, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
			}
			else {
				cl_float mult = float(SQR(SL[iEl]) * NatomEl[iEl]);
				clSetKernelArg(addIKernelNeutron, 0, sizeof(cl_mem), (void *)&dI[Ipart]);
				clSetKernelArg(addIKernelNeutron, 2, sizeof(cl_float), (void *)&mult);
				clEnqueueNDRangeKernel(queue, addIKernelNeutron, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//add contribution form diagonal (i==j) elements in Debye sum
			}
		}
		unsigned int jAtomST = iAtomST;
		for (unsigned int jEl = iEl; jEl < cfg->Nel; jAtomST += NatomEl[jEl], jEl++) {
			if (cfg->source == xray) clSetKernelArg(calcIntDebyeKernelXray, 2, sizeof(cl_mem), (void *)&dFF[jEl]);
			else {
				cl_float SLij = (cl_float)(SL[iEl] * SL[jEl]);
				clSetKernelArg(calcIntDebyeKernelNeutron, 1, sizeof(cl_float), (void *)&SLij);
			}
			unsigned int iAtomSB = 0;
			for (unsigned int iB = 0; iB < cfg->Nblocks; iAtomSB += NatomElBlock[iEl * cfg->Nblocks + iB], iB++) {
				for (unsigned int iAtom = 0; iAtom < NatomElBlock[iEl * cfg->Nblocks + iB]; iAtom += BlockSize2D * GridSizeExecMax){
					unsigned int GridSizeExecY = MIN((NatomElBlock[iEl * cfg->Nblocks + iB] - iAtom) / BlockSize2D + BOOL((NatomElBlock[iEl * cfg->Nblocks + iB] - iAtom) % BlockSize2D), GridSizeExecMax);
					unsigned int iMax = iAtomST + iAtomSB + MIN(iAtom + BlockSize2D * GridSizeExecY, NatomEl[iEl]);
					unsigned int i0 = iAtomST + iAtomSB + iAtom;
					unsigned int jAtomSB = 0;
					for (unsigned int jB = 0; jB < cfg->Nblocks; jAtomSB += NatomElBlock[jEl * cfg->Nblocks + jB], jB++) {
						(jB>iB) ? Ipart = cfg->Nblocks * iB - (iB * (iB + 1)) / 2 + jB : Ipart = cfg->Nblocks * jB - (jB * (jB + 1)) / 2 + iB;
						if (cfg->source == xray) clSetKernelArg(calcIntDebyeKernelXray, 0, sizeof(cl_mem), (void *)&dI[Ipart]);
						else clSetKernelArg(calcIntDebyeKernelNeutron, 0, sizeof(cl_mem), (void *)&dI[Ipart]);
						for (unsigned int jAtom = 0; jAtom < NatomElBlock[jEl * cfg->Nblocks + jB]; jAtom += BlockSize2D * GridSizeExecMax){
							unsigned int j0 = jAtomST + jAtomSB + jAtom;
							if (j0 >= i0) {
								unsigned int GridSizeExecX = MIN((NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom) / BlockSize2D + BOOL((NatomElBlock[jEl * cfg->Nblocks + jB] - jAtom) % BlockSize2D), GridSizeExecMax);
								unsigned int jMax = jAtomST + jAtomSB + MIN(jAtom + BlockSize2D*GridSizeExecX, NatomElBlock[jEl * cfg->Nblocks + jB]);
								size_t global_work_size[2] = { BlockSize2D*GridSizeExecX, BlockSize2D * GridSizeExecY };
								(i0 == j0) ? diag = 1 : diag = 0;//checking if we are on the diagonal grid or not
								if (cfg->source == xray) {
									clSetKernelArg(calcIntDebyeKernelXray, 6, sizeof(cl_uint), (void *)&i0);
									clSetKernelArg(calcIntDebyeKernelXray, 7, sizeof(cl_uint), (void *)&j0);
									clSetKernelArg(calcIntDebyeKernelXray, 8, sizeof(cl_uint), (void *)&iMax);
									clSetKernelArg(calcIntDebyeKernelXray, 9, sizeof(cl_uint), (void *)&jMax);
									clSetKernelArg(calcIntDebyeKernelXray, 10, sizeof(cl_uint), (void *)&diag);
									clEnqueueNDRangeKernel(queue, calcIntDebyeKernelXray, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
								}
								else {//neutron
									clSetKernelArg(calcIntDebyeKernelNeutron, 5, sizeof(cl_uint), (void *)&i0);
									clSetKernelArg(calcIntDebyeKernelNeutron, 6, sizeof(cl_uint), (void *)&j0);
									clSetKernelArg(calcIntDebyeKernelNeutron, 7, sizeof(cl_uint), (void *)&iMax);
									clSetKernelArg(calcIntDebyeKernelNeutron, 8, sizeof(cl_uint), (void *)&jMax);
									clSetKernelArg(calcIntDebyeKernelNeutron, 9, sizeof(cl_uint), (void *)&diag);
									clEnqueueNDRangeKernel(queue, calcIntDebyeKernelNeutron, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
								}
								if (kernelExecTimeoutEnabled) {
									clFlush(queue);
									clFinish(queue);
								}
							}
						}
					}
				}
			}
		}
	}
	clFlush(queue);
	clFinish(queue);
	delete[] NatomElBlock;
	unsigned int IpartialSize = (Nparts + 1) * cfg->q.N;
	dIpart = clCreateBuffer(OCLcontext, CL_MEM_READ_WRITE, IpartialSize * sizeof(cl_float), NULL, NULL);
	cl_uint Ncopies = SQR(GridSizeExecMax);
	cl_kernel sumIpartialKernel = clCreateKernel(OCLprogram, "sumIpartialKernel", NULL); //summing intensity copies
	clSetKernelArg(sumIpartialKernel, 1, sizeof(cl_mem), (void *)&dIpart);
	clSetKernelArg(sumIpartialKernel, 3, sizeof(cl_uint), (void *)&cfg->q.N);
	clSetKernelArg(sumIpartialKernel, 4, sizeof(cl_uint), (void *)&Ncopies);
	for (unsigned int ipart=0; ipart<Nparts; ipart++) {
		clSetKernelArg(sumIpartialKernel, 0, sizeof(cl_mem), (void *)&dI[ipart]);
		clSetKernelArg(sumIpartialKernel, 2, sizeof(cl_uint), (void *)&ipart);
		clEnqueueNDRangeKernel(queue, sumIpartialKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//summing intensity copies
	}
	cl_kernel PolarFactor1DKernel = NULL;
	if (cfg->PolarFactor) {
		size_t global_work_size_polar[2] = {GSadd*local_work_size_zero,Nparts+1};
		size_t local_work_size_polar[2] = {local_work_size_zero,1};
		PolarFactor1DKernel = clCreateKernel(OCLprogram, "PolarFactor1DKernel", NULL);
		cl_float lambdaf = float(cfg->lambda);
		clSetKernelArg(PolarFactor1DKernel, 0, sizeof(cl_mem), (void *)&dIpart);
		clSetKernelArg(PolarFactor1DKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
		clSetKernelArg(PolarFactor1DKernel, 2, sizeof(cl_mem), (void *)&dq);
		clSetKernelArg(PolarFactor1DKernel, 3, sizeof(cl_float), (void *)&lambdaf);
		clEnqueueNDRangeKernel(queue, PolarFactor1DKernel, 2, NULL, global_work_size_polar, local_work_size_polar, 0, NULL, NULL);
	}
	cl_kernel sumIKernel = clCreateKernel(OCLprogram, "sumIKernel", NULL); //summing partial intensities
	clSetKernelArg(sumIKernel, 0, sizeof(cl_mem), (void *)&dIpart);
	clSetKernelArg(sumIKernel, 1, sizeof(cl_uint), (void *)&cfg->q.N);
	cl_uint Nsum = Nparts + 1;
	clSetKernelArg(sumIKernel, 2, sizeof(cl_uint), (void *)&Nsum);
	clEnqueueNDRangeKernel(queue, sumIKernel, 1, NULL, &global_work_size_add, &local_work_size_zero, 0, NULL, NULL);//summing intensity copies
	if (UMflag) hI = (cl_float *)clEnqueueMapBuffer(queue, dIpart, true, CL_MAP_READ, 0, IpartialSize * sizeof(cl_float), 0, NULL, NULL, NULL);
	else {
		hI = new cl_float[IpartialSize];
		clEnqueueReadBuffer(queue, dIpart, true, 0, IpartialSize * sizeof(cl_float), (void *)hI, 0, NULL, NULL);
	}
	*I = new double[IpartialSize];
	for (unsigned int iq = 0; iq < IpartialSize; iq++) (*I)[iq] = double(hI[iq]) / Ntot;//normalizing
	if (UMflag) clEnqueueUnmapMemObject(queue, dIpart, (void *)hI, 0, NULL, NULL);
	else delete[] hI;
	clFlush(queue);
	clFinish(queue);
	clReleaseKernel(zero1DFloatArrayKernel);
	clReleaseKernel(sumIKernel);
	clReleaseKernel(sumIpartialKernel);
	if (cfg->source == xray) {
		clReleaseKernel(addIKernelXray);
		clReleaseKernel(calcIntDebyeKernelXray);
	}
	else {
		clReleaseKernel(addIKernelNeutron);
		clReleaseKernel(calcIntDebyeKernelNeutron);
	}
	if (cfg->PolarFactor) clReleaseKernel(PolarFactor1DKernel);
	clReleaseCommandQueue(queue);
	for (unsigned int ipart=0; ipart<Nparts; ipart++) clReleaseMemObject(dI[ipart]);
	delete[] dI;
	clReleaseMemObject(dIpart);
	t2 = chrono::steady_clock::now();
	cout << "1D pattern calculation time: " << chrono::duration_cast< chrono::duration < float > >(t2 - t1).count() << " s" << endl;
}
#endif
