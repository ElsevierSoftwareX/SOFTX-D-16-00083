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

//Contains OpenCL kernels used to compute the powder diffraction patterns and PDFs with the hostogram approximation

//Some macros
#define SQR(x) ((x)*(x))
#define BlockSize2D 16
#define PIf 3.14159265f
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define BOOL(x) ((x) ? 1 : 0)

//use built-in 64-bit atomic add if the GPU supports it, define custom 64-bit atomic add if not
#ifndef CustomInt64atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#else
void atomInc64 (__global uint *counter) {
	uint old, carry;

	old = atomic_inc(&counter[0]);
	carry = old == 0xFFFFFFFF;
	atomic_add(&counter[1], carry);
}
#endif

/**
	Computes polarization factor and multiplies scattering intensity by this factor

	@param *I     Scattering intensity array
	@param Nq     Size of the scattering intensity array
	@param *q     Scattering vector magnitude array
	@param lambda Wavelength of the source
*/
__kernel void PolarFactor1DKernel(__global float *I, unsigned int Nq, const __global float *q, float lambda){
	unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq)	{
		float sintheta = q[iq] * (lambda * 0.25f / PIf);
		float cos2theta = 1.f - 2.f * SQR(sintheta);
		float factor = 0.5f * (1.f + SQR(cos2theta));
		I[iq] *= factor;
	}
}

/**
	Resets 1D float array of size N

	@param *A  Array
	@param N   Size of the array
*/
__kernel void zero1DFloatArrayKernel(__global float *A, unsigned int N){
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i<N) A[i] = 0;
}

/**
	Adds the diagonal elements (j==i) of the Debye double sum to the x-ray scattering intensity

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param *FF   X-ray atomic form-factor array (for one kernel call the computations are done only for the atoms of the same chemical element)	
	@param N     Total number of atoms of the chemical element for whcich the computations are done
*/
__kernel void addIKernelXray(__global float *I, unsigned int Nq, const __global float *FFi, unsigned int N) {
	unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq)	{
		float lFF = FFi[iq];
		I[iq] += SQR(lFF) * N;
	}
}

/**
	Adds the diagonal elements (j==i) of the Debye double sum to the neutron scattering intensity

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param Add   The value to add to the intensity (the result of multiplying the square of the scattering length
	to the total number of atoms of the chemical element for whcich the computations are done)
*/
__kernel void addIKernelNeutron(__global float *I, unsigned int Nq, float Add) {
	unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq < Nq)	I[iq] += Add;
}

/**
	Computes the total scattering intensity (first Nq elements) from the partials sums computed by different work-groups

	@param *I    Scattering intensity array
	@param Nq    Resolution of the total scattering intensity (powder diffraction pattern)
	@param Nsum  Number of parts to sum (equalt to the total number of work-groups)
*/
__kernel void sumIKernel(__global float *I, unsigned int Nq, unsigned int Nsum){
	unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (iq<Nq) {
		for (unsigned int j = 1; j < Nsum; j++)	I[iq] += I[j * Nq + iq];
	}
}

/**
	Resets the histogram array (ulong)

	@param *rij_hist  Histogram of interatomic distances
	@param N          Size of the array
*/
__kernel void zeroHistKernel(__global ulong *rij_hist, unsigned int N){
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i<N) rij_hist[i] = 0;
}

/**
	Computes the partial radial distribution function (RDF)

	@param *dPDF     Partial PDF array (contains all partial PDFs)
	@param *rij_hist Histogram of interatomic distances (device)
	@param iSt       Index of the first element of the current (for this kernel call) partial histogram/PDF
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (Ntot * bin_width)
*/
__kernel void calcPartialRDFkernel(__global float *dPDF, const __global ulong *rij_hist, unsigned int iSt, unsigned int Nhist, float mult) {
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * mult;
}

/**
	Computes the partial pair distribution function (PDF)

	@param *dPDF     Prtial PDF array (contains all partial PDFs)
	@param *rij_hist Histogram of interatomic distances (device)
	@param iSt       Index of the first element of the current (for this kernel call) partial histogram/PDF
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (4 * PI * rho * Ntot * bin_width)
	@param bin       Width of the histogram bin
*/
__kernel void calcPartialPDFkernel(__global float *dPDF, const __global ulong *rij_hist, unsigned int iSt, unsigned int Nhist, float mult, float bin) {
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) {
		float r = (i + 0.5f) * bin;
		dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * (mult / SQR(r));
	}
}

/**
	Computes the partial reduced pair distribution function (rPDF)

	@param *dPDF     Partial PDF array (contains all partial PDFs)
	@param *rij_hist Histogram of interatomic distances (device)
	@param iSt       Index of the first element of the current (for this kernel call) partial histogram/PDF
	@param Nhist     Size of the partial histogram of interatomic distances
	@param mult      1 / (Ntot * bin_width)
	@param submult   4 * PI * rho * NatomEl_i * NatomEl_j / SQR(Ntot)
	@param bin       Width of the histogram bin
*/
__kernel void calcPartialRPDFkernel(__global float *dPDF, const __global ulong *rij_hist, unsigned int iSt, unsigned int Nhist, float mult, float submult, float bin) {
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) {
		float r = (i + 0.5f) * bin;
		dPDF[Nhist + iSt + i] = rij_hist[iSt + i] * (mult / r) - submult * r;
	}
}

/**
	Computes the total PDF using the partial PDFs

	@param *dPDF   Total (first Nhist elements) + partial PDF array. The memory is allocated inside the function.
	@param iSt     Index of the first element of the partial PDF whcih will be added to the total PDF in this kernel call
	@param Nhist   Size of the partial histogram of interatomic distances
	@param multIJ  FF_i(q0) * FF_j(q0) / <FF> (for x-ray) and SL_i * SL_j / <SL> (for neutron)
*/
__kernel void calcPDFkernel(__global float *dPDF, unsigned int iSt, unsigned int Nhist, float multIJ) {
	unsigned int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if (i < Nhist) 	dPDF[i] += dPDF[iSt + i] * multIJ;
}

/**
	Computes the histogram of interatomic distances

	@param *ra        Atomic coordinate array 
	@param i0         Index of the 1st i-th atom in ra array for this kernel call
	@param j0         Index of the 1st j-th atom in ra array for this kernel call
	@param iMax       Total number of i-th atoms for this kernel call
	@param jMax       Total number of j-th atoms for this kernel call
	@param *rij_hist  Histogram of interatomic distances
	@param iSt        Index of the first element of the partial histogram corresponding to the i-th and j-th atoms for this kernel call
	@param bin        Width of the histogram bin
	@param diag       True if the j-th atoms and the i-th atoms are the same (diagonal) for this kernel call
*/
__kernel void calcHistKernel(const __global float4 *ra, unsigned int i0, unsigned int j0, unsigned int iMax, unsigned int jMax, __global ulong *rij_hist, unsigned int iSt, float bin, unsigned int diag){
	if ((diag) && (get_group_id(0) < get_group_id(1))) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal work-groups (for which j < i for all work-items) do nothing and return
	unsigned int jt = get_local_id(0), it = get_local_id(1);
	unsigned int j = get_group_id(0) * BlockSize2D + jt;
	unsigned int iCopy = get_group_id(1) * BlockSize2D + jt; //jt!!! memory transaction are performed by the work-item of the same wavefront to coalesce them
	unsigned int i = get_group_id(1) * BlockSize2D + it;
	__local float4 ris[BlockSize2D], rjs[BlockSize2D];
	if ((it == 0) && (j + j0 < jMax)) { //copying atomic coordinates for j-th (column) atoms
		rjs[jt] = ra[j0 + j];
	}
	if ((it == 4) && (iCopy + i0 < iMax)) { //the same for i-th (row) atoms
		ris[jt] = ra[i0 + iCopy];
	}
	barrier(CLK_LOCAL_MEM_FENCE);//sync to ensure that copying is complete
	if (!diag){
		if ((j + j0 < jMax) && (i + i0 < iMax)) {
			float rij = sqrt(SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z)); //calculate distance
			unsigned int index = (unsigned int)(rij / bin); //get the index of histogram bin
#ifndef CustomInt64atomics
			atom_inc(&rij_hist[iSt + index]); //add +1 to histogram bin
#else
			atomInc64(&rij_hist[iSt + index]);
#endif
		}
	}
	else{//we are in diagonal grid
		if ((j + j0 < jMax) && (i + i0 < iMax) && (j > i)) {//all the subdiagonal work-groups already quit, but we have diagonal work-groups  (get_group_id(0) == get_group_id(1)), so we should check if j > i
			float rij = sqrt(SQR(ris[it].x - rjs[jt].x) + SQR(ris[it].y - rjs[jt].y) + SQR(ris[it].z - rjs[jt].z));
			unsigned int index = (unsigned int)(rij / bin);
#ifndef CustomInt64atomics
			atom_inc(&rij_hist[iSt + index]);
#else
			atomInc64(&rij_hist[iSt + index]);
#endif
		}
	}
}

/**
	Computes the x-ray scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

	@param *I              Scattering intensity array
	@param *FFi            X-ray atomic form factor for the i-th atoms (all the i-th atoms are of the same chemical element for one kernel call)
	@param *FFj            X-ray atomic form factor for the j-th atoms (all the j-th atoms are of the same chemical element for one kernel call)
	@param *q              Scattering vector magnitude array
	@param Nq              Size of the scattering intensity array
	@param **rij_hist      Histogram of interatomic distances (device). The memory is allocated inside the function
	@param iSt             Index of the first element of the partial histogram corresponding to the i-th and j-th atoms for this kernel call
	@param iBinSt          Starting index of the partial histogram bin for this kernel call (the kernel is called iteratively in a loop)
	@param Nhist           Size of the partial histogram of interatomic distances
	@param MaxBinsPerBlock Maximum number of histogram bins used by a single work-group
	@param bin             Width of the histogram bin
*/
__kernel void calcIntHistKernelXray(__global float *I, const __global float *FFi, const __global float *FFj, const  __global float *q, unsigned int Nq, const __global ulong *rij_hist, unsigned int iSt, unsigned int iBinSt, unsigned int Nhist, unsigned int MaxBinsPerBlock, float bin){
	unsigned int iBegin = iBinSt + get_group_id(0) * MaxBinsPerBlock;//first index for histogram bin to process
	unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);//last index for histogram bin to process
	for (unsigned int iterq = 0; iterq < (Nq / get_local_size(0)) + BOOL(Nq % get_local_size(0)); iterq++) {//if Nq > get_local_size(0 there will be work-items that compute more than one element of the intensity array
		unsigned int iq = iterq * get_local_size(0) + get_local_id(0);//index of the intensity array element
		if (iq < Nq) {//checking for the array margin
			float lI = 0, qrij;
			float lq = q[iq];//copying the scattering vector modulus to the private memory
			for (unsigned int i = iBegin; i < iEnd; i++) {//looping over the histogram bins
				ulong Nrij = rij_hist[iSt + i];
				if (Nrij){
					qrij = lq * (i + 0.5f) * bin;//distance that corresponds to the current histogram bin
					lI += (Nrij * native_sin(qrij)) / (qrij + 0.000001f);//scattering intensity without form factors
				}
			}
			float lFFij = 2.f * FFi[iq] * FFj[iq];
			I[get_group_id(0)*Nq + iq] += lI * lFFij;//multiplying intensity by form-factors and storing the results in global memory
		}
	}
}

/**
	Computes the neutron scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

	@param *I              Scattering intensity array
	@param SLij            Product of the scattering lenghts of i-th j-th atoms
	@param *q              Scattering vector magnitude array
	@param Nq              Size of the scattering intensity array
	@param **rij_hist      Histogram of interatomic distances (device). The memory is allocated inside the function
	@param iSt             Index of the first element of the partial histogram corresponding to the i-th and j-th atoms for this kernel call
	@param iBinSt          Starting index of the histogram bin for this kernel call (the kernel is called iteratively in a loop)
	@param Nhist           Size of the partial histogram of interatomic distances
	@param MaxBinsPerBlock Maximum number of histogram bins used by a single work-group
	@param bin             Width of the histogram bin
*/
__kernel void calcIntHistKernelNeutron(__global float *I, float SLij, const  __global float *q, unsigned int Nq, const __global ulong *rij_hist, unsigned int iSt, unsigned int iBinSt, unsigned int Nhist, unsigned int MaxBinsPerBlock, float bin){
	//see comments in the calcIntHistKernelXray() kernel
	unsigned int iBegin = iBinSt + get_group_id(0) * MaxBinsPerBlock;
	unsigned int iEnd = MIN(Nhist, iBegin + MaxBinsPerBlock);
	for (unsigned int iterq = 0; iterq < (Nq / get_local_size(0)) + BOOL(Nq % get_local_size(0)); iterq++) {
		unsigned int iq = iterq * get_local_size(0) + get_local_id(0);
		if (iq < Nq) {
			float lI = 0, qrij;
			float lq = q[iq];
			for (unsigned int i = iBegin; i < iEnd; i++) {
				ulong Nrij = rij_hist[iSt + i];
				if (Nrij){
					qrij = lq * (i + 0.5f) * bin;
					lI += (Nrij * native_sin(qrij)) / (qrij + 0.000001f);
				}
			}
			I[get_group_id(0) * Nq + iq] += 2.f * lI * SLij;
		}
	}
}