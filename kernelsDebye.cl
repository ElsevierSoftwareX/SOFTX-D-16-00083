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

//Contains OpenCL kernels used to compute the powder diffraction patterns using the original Debye equation (without the histogram approximation)

//some macros
#define SQR(x) ((x)*(x))
#define BlockSize2D 16
#define PIf 3.14159265f
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define BOOL(x) ((x) ? 1 : 0)

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
		I[get_group_id(1) * Nq + iq] *= factor;
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
        float lIsum=0;
		for (unsigned int j = 1; j < Nsum; j++)	lIsum += I[j * Nq + iq];
        I[iq] += lIsum;
	}
}

/**
	Computes the x-ray scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

	@param *I    Scattering intensity array
	@param *FFi  X-ray atomic form factor for the i-th atoms (all the i-th atoms are of the same chemical element for one kernel call)
	@param *FFj  X-ray atomic form factor for the j-th atoms (all the j-th atoms are of the same chemical element for one kernel call)
	@param *q    Scattering vector magnitude array
	@param Nq    Size of the scattering intensity array
	@param *ra   Atomic coordinate array
	@param i0    Index of the 1st i-th atom in ra array for this kernel call
	@param j0    Index of the 1st j-th atom in ra array for this kernel call
	@param iMax  Total number of i-th atoms for this kernel call
	@param jMax  Total number of j-th atoms for this kernel call
	@param diag  True if the j-th atoms and the i-th atoms are the same (diagonal) for this kernel call
*/
__kernel void calcIntDebyeKernelXray(__global float *I, const __global float *FFi, const __global float *FFj, const __global float *q, unsigned int Nq, const __global float4 *ra, unsigned int i0, unsigned int j0, unsigned int iMax, unsigned int jMax, unsigned int diag){
	if ((diag) && (get_group_id(0) < get_group_id(1))) return; //we need to calculate inter-atomic distances only for j > i, so if we are in the diagonal grid, all the subdiagonal work-groups (for which j < i for all work-items) do nothing and return
	unsigned int jt = get_local_id(0), it = get_local_id(1);
	unsigned int j = get_group_id(0) * BlockSize2D + jt;
	unsigned int iCopy = get_group_id(1) * BlockSize2D + jt; //jt!!! memory transaction are performed by the work-items of the same wavefront/warp to coalesce them
	unsigned int i = get_group_id(1) * BlockSize2D + it;
	__local float xis[BlockSize2D], yis[BlockSize2D], zis[BlockSize2D], xjs[BlockSize2D], yjs[BlockSize2D], zjs[BlockSize2D]; //cache arrays for atomic coordinates (we use separate x,y,z arrays here to avoid bank conflicts)
	__local float rij[BlockSize2D][BlockSize2D]; //cache array for inter-atomic distances
	rij[it][jt] = 0;
	if ((it == 0) && (j + j0 < jMax)) { //copying atomic coordinates for j-th (column) atoms
		float4 rt = ra[j0 + j];
		xjs[jt] = rt.x, yjs[jt] = rt.y, zjs[jt] = rt.z;
	}
	if ((it == 4) && (iCopy + i0 < iMax)) { //the same for i-th (row) atoms
		float4 rt = ra[i0 + iCopy];
		xis[jt] = rt.x, yis[jt] = rt.y, zis[jt] = rt.z;
	}
	barrier(CLK_LOCAL_MEM_FENCE);//sync to ensure that copying is complete
	if (!diag){
		if ((j + j0 < jMax) && (i + i0 < iMax)) rij[it][jt] = sqrt(SQR(xis[it] - xjs[jt]) + SQR(yis[it] - yjs[jt]) + SQR(zis[it] - zjs[jt])); //calculate distance
	}
	else {
		if ((j + j0 < jMax) && (i + i0 < iMax) && (j > i)) rij[it][jt] = sqrt(SQR(xis[it] - xjs[jt]) + SQR(yis[it] - yjs[jt]) + SQR(zis[it] - zjs[jt]));
	}
	barrier(CLK_LOCAL_MEM_FENCE);//synchronizing work-items to ensure that the calculation of the distances is complete
	iMax = MIN(BlockSize2D, iMax - i0 - get_group_id(1) * BlockSize2D); //last i-th (row) atom index for the current work-group
	jMax = MIN(BlockSize2D, jMax - j0 - get_group_id(0) * BlockSize2D); //last j-th (column) atom index for the current work-group
	for (unsigned int iterq = 0; iterq < Nq; iterq += SQR(BlockSize2D)) {//if Nq > SQR(BlockSize2D) there will be work-items that compute more than one element of the intensity array
		unsigned int iq = iterq + it*BlockSize2D + jt;
		if (iq < Nq) {//checking for array margin
			float lI = 0, qrij;
			float lq = q[iq];//copying the scattering vector magnitude to the private memory
			if ((diag) && (get_group_id(0) == get_group_id(1))) {//diagonal work-group, j starts from i + 1
				for (i = 0; i < iMax; i++) {
					for (j = i + 1; j < jMax; j++) {
						qrij = lq * rij[i][j];
						lI += native_sin(qrij) / (qrij + 0.000001f); //scattering intensity without form-factors
					}
				}
			}
			else {//j starts from 0
				for (i = 0; i < iMax; i++) {
					for (j = 0; j < jMax; j += 8) {//unrolling to speed up the performance
						qrij = lq * rij[i][j];
						lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+1];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+2];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+3];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+4];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+5];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+6];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+7];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
					}
				}
			}
			I[Nq * (get_num_groups(0) * get_group_id(1) + get_group_id(0)) + iq] += 2.f * lI * FFi[iq] * FFj[iq]; //multiplying the intensity by form-factors and storing the results in the global memory (2.f is for j < i part)
		}
	}
}

/**
	Computes the x-ray scattering intensity (powder diffraction pattern) using the histogram of interatomic distances

	@param *I    Scattering intensity array
	@param SLij  Product of the scattering lenghts of i-th j-th atoms
	@param *q    Scattering vector magnitude array
	@param Nq    Size of the scattering intensity array
	@param *ra   Atomic coordinate array
	@param i0    Index of the 1st i-th atom in ra array for this kernel call
	@param j0    Index of the 1st j-th atom in ra array for this kernel call
	@param iMax  Total number of i-th atoms for this kernel call
	@param jMax  Total number of j-th atoms for this kernel call
	@param diag  True if the j-th atoms and the i-th atoms are the same (diagonal) for this kernel call
*/
__kernel void calcIntDebyeKernelNeutron(__global float *I, float SLij, const __global float *q, unsigned int Nq, const __global float4 *ra, unsigned int i0, unsigned int j0, unsigned int iMax, unsigned int jMax, unsigned int diag){
	//see comments in the calcIntDebyeKernelXray kernel
	if ((diag) && (get_group_id(0) < get_group_id(1))) return;
	unsigned int jt = get_local_id(0), it = get_local_id(1);
	unsigned int j = get_group_id(0) * BlockSize2D + jt;
	unsigned int iCopy = get_group_id(1) * BlockSize2D + jt;
	unsigned int i = get_group_id(1) * BlockSize2D + it;
	__local float xis[BlockSize2D], yis[BlockSize2D], zis[BlockSize2D], xjs[BlockSize2D], yjs[BlockSize2D], zjs[BlockSize2D];
	__local float rij[BlockSize2D][BlockSize2D];
	rij[it][jt] = 0;
	if ((it == 0) && (j + j0 < jMax)) {
		float4 rt = ra[j0 + j];
		xjs[jt] = rt.x, yjs[jt] = rt.y, zjs[jt] = rt.z;
	}
	if ((it == 4) && (iCopy + i0 < iMax)) {
		float4 rt = ra[i0 + iCopy];
		xis[jt] = rt.x, yis[jt] = rt.y, zis[jt] = rt.z;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (!diag){
		if ((j + j0 < jMax) && (i + i0 < iMax)) rij[it][jt] = sqrt(SQR(xis[it] - xjs[jt]) + SQR(yis[it] - yjs[jt]) + SQR(zis[it] - zjs[jt]));
	}
	else {
		if ((j + j0 < jMax) && (i + i0 < iMax) && (j > i)) rij[it][jt] = sqrt(SQR(xis[it] - xjs[jt]) + SQR(yis[it] - yjs[jt]) + SQR(zis[it] - zjs[jt]));
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	iMax = MIN(BlockSize2D, iMax - i0 - get_group_id(1) * BlockSize2D);
	jMax = MIN(BlockSize2D, jMax - j0 - get_group_id(0) * BlockSize2D);
	for (unsigned int iterq = 0; iterq < Nq; iterq += SQR(BlockSize2D)) {
		unsigned int iq = iterq + it * BlockSize2D + jt;
		if (iq < Nq) {
			float lI = 0, qrij;
			float lq = q[iq];
			if ((diag) && (get_group_id(0) == get_group_id(1))) {
				for (i = 0; i < iMax; i++) {
					for (j = i + 1; j < jMax; j++) {
						qrij = lq * rij[i][j];
						lI += native_sin(qrij) / (qrij + 0.000001f);
					}
				}
			}
			else {
				for (i = 0; i < iMax; i++) {
					for (j = 0; j < jMax; j += 8) {
                        qrij = lq * rij[i][j];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+1];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+2];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+3];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+4];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+5];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+6];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
                        qrij = lq * rij[i][j+7];
                        lI += native_sin(qrij) / (qrij + 0.000001f);
					}
				}
			}
			I[Nq * (get_num_groups(0) * get_group_id(1) + get_group_id(0)) + iq] += 2.f * lI * SLij;
		}
	}
}

/**
	Computes the partial scattering intensity (*Ipart) from the partials sums (*I) computed by different thread blocks

	@param *I     Scattering intensity array (partials sums as computed by thread blocks)
	@param *Ipart Array with partial scattering intensities
	@param ipart  Index of the current partial scattering intensity for this kernel call (the kernel is called iteratively in the loop)
	@param Nq     Resolution of the total scattering intensity (powder diffraction pattern)
	@param Nsum   Number of parts to sum (equalt to the total number of thread blocks in the grid)
*/
__kernel void sumIpartialKernel(__global float *I,__global float *Ipart, unsigned int ipart,  unsigned int Nq, unsigned int Nsum){
    unsigned int iq = get_local_size(0) * get_group_id(0) + get_local_id(0);
    if (iq<Nq) {
        float lIsum=0;
        for (unsigned int j = 0; j < Nsum; j++)	lIsum += I[j * Nq + iq];
        Ipart[(ipart + 1) * Nq + iq] = lIsum;
        Ipart[iq] = 0;
    }
}