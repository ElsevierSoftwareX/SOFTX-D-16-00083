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

//Contains OpenCL kernels used to compute the 2D (monocrystal) diffraction patterns

//Some macros
#define SQR(x) ((x)*(x))
#define BlockSize2D 16
#define SizeR 128
#define PIf 3.14159265f
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define BOOL(x) ((x) ? 1 : 0)

/**
	Resets the 2D scattering intensity array

	@param *I   Intensity array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
*/
__kernel void zeroInt2DKernel(__global float *I, unsigned int Nq, unsigned int Nfi){
	unsigned int iq = get_local_size(1) * get_group_id(1) + get_local_id(1), ifi = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi))	I[iq * Nfi + ifi] = 0;
}

/**
	Resets the 2D scattering amplitude arrays (real and imaginary parts)

	@param *Ar  Real part of the 2D scattering amplitude array
	@param *Ai  Imaginary part of the 2D scattering amplitude array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
*/
__kernel void zeroAmp2DKernel(__global float *Ar, __global float *Ai, unsigned int Nq, unsigned int Nfi){
	unsigned int iq = get_local_size(1) * get_group_id(1) + get_local_id(1), ifi = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi)){
		Ar[iq * Nfi + ifi] = 0;
		Ai[iq * Nfi + ifi] = 0;
	}
}

/**
	Computes the 2D scattering intensity using the scattering amplitude

	@param *I   Intensity array
	@param *Ar  Real part of the 2D scattering amplitude array
	@param *Ai  Imaginary part of the 2D scattering amplitude array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D amplitude array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D amplitude array)
*/
__kernel void Sum2DKernel(__global float *I, const __global float *Ar, const __global float *Ai, unsigned int Nq, unsigned int Nfi){
	unsigned int iq = get_local_size(1) * get_group_id(1) + get_local_id(1), ifi = get_local_size(0) * get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi))	I[iq * Nfi + ifi] += SQR(Ar[iq * Nfi + ifi]) + SQR(Ai[iq * Nfi + ifi]);
}

/**
	Multiplies the 2D scattering intensity by a normalizing factor

	@param *I   Intensity array
	@param Nq   Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi  Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
	@param norm Normalizing factor
*/
__kernel void Norm2DKernel(__global float *I, unsigned int Nq, unsigned int Nfi, float norm){
	unsigned int iq = get_local_size(1) * get_group_id(1) + get_local_id(1), ifi = get_local_size(0) *get_group_id(0) + get_local_id(0);
	if ((iq < Nq) && (ifi < Nfi))	I[iq * Nfi + ifi] *= norm;
}

/**
	Computes the polarization factor and multiplies the 2D scattering intensity by this factor

	@param *I     Intensity array
	@param Nq     Size of the scattering vector magnitude mesh (number of rows in the 2D intensity array)
	@param Nfi    Size of the scattering vector polar angle mesh (number of columns in the 2D intensity array)
	@param *q     Scattering vector magnitude array
	@param lambda Wavelength of the source
*/
__kernel void PolarFactor2DKernel(__global float *I, unsigned int Nq, unsigned int Nfi,const __global float *q, float lambda){
	unsigned int iq = get_local_size(1) * get_group_id(1) + get_local_id(1), ifi = get_local_size(0) * get_group_id(0) + get_local_id(0);
	unsigned int iqCopy = get_local_size(1) * get_group_id(1) + get_local_id(0);
	__local float factor[BlockSize2D];
	if ((get_local_id(1) == 0) && (iqCopy < Nq)) {
		//polarization factor is computed only by the first BlockSize2D work-items of the first wavefront/warp and stored in the shared memory
		float sintheta = q[iqCopy] * (lambda * 0.25f / PIf);
		float cos2theta = 1.f - 2.f * SQR(sintheta);
		factor[get_local_id(0)] = 0.5f * (1.f + SQR(cos2theta));
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((iq < Nq) && (ifi < Nfi)) I[iq * Nfi + ifi] *= factor[get_local_id(1)];
}

/**
	Computes the real and imaginary parts of the 2D x-ray scattering amplitude in the polar coordinates (q,q_fi) of the reciprocal space

	@param *Ar    Real part of the 2D scattering amplitude array
	@param *Ai	  Imaginary part of the 2D scattering amplitude array
	@param *q     Scattering vector magnitude array
	@param Nq     Size of the scattering vector magnitude mesh (number of rows in the 2D amplitude array)
	@param Nfi    Size of the scattering vector polar angle mesh (number of columns in the 2D amplitude array)
	@param CS[]   Transposed rotational matrix. Defines the orientation of the nanoparticle in the 3D space.
	@param lambda Wavelength of the source
	@param *ra    Atomic coordinate array
	@param Nst    Index of the 1st atom in ra array for this kernel call (the kernel is called iteratively in the loop)
	@param Nfin   Number of atoms to compute for in this kernel call (less or equal to the total number of atoms)
	@param *FF    X-ray atomic form-factor array (for one kernel call the computations are done only for the atoms of the same chemical element)
*/
__kernel void calcInt2DKernelXray(__global float *Ar, __global float *Ai, const __global float *q, unsigned int Nq, unsigned int Nfi, const __global float4 *CS, float lambda, const __global float4 *ra, unsigned int Nst, unsigned int Nfin, const __global float *FF){
	//to avoid bank conflicts for local memory operations BlockSize2D should be equal to the size of the wavefront/warp
	//SizeR should be a multiple of BlockSize2D
	unsigned int iq = get_local_size(1) * get_group_id(1) + get_local_id(1), ifi = get_local_size(0) * get_group_id(0) + get_local_id(0); //each work-unit computes only one element of 2D amplitude matrix
	unsigned int iqCopy = get_local_size(1) * get_group_id(1) + get_local_id(0);//copying of the scattering vector magnitude array to the local memory performed by the work-items of the same wavefront/warp
	__local float lFF[BlockSize2D]; //cache array for the atomic from-factors
	__local float qi[BlockSize2D]; //cache array for the scattering vector modulus
	__local float4 r[SizeR]; //cache array for the atomic coordinates
	unsigned int Niter = Nfin / SizeR + BOOL(Nfin % SizeR);//we don't have enough local memory to load the array of atomic coordinates as a whole, so we do it with iterations
	float4 qv; //scattering vector
	float lAr = 0, lAi = 0, cosfi = 0, sinfi = 0, sintheta = 0, costheta = 0;
	if ((get_local_id(1) == 0) && (iqCopy < Nq)) lFF[get_local_id(0)] = FF[iqCopy]; //loading Xray form-factors to the local memory
	if ((get_local_id(1) == 4) && (iqCopy < Nq)) qi[get_local_id(0)] = q[iqCopy]; //loading scattering vector magnitude to the local memory
	barrier(CLK_LOCAL_MEM_FENCE); //synchronizing after loading to the local memory
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		float arg = ifi * 2.f * PIf / Nfi;
		sinfi = native_sin(arg);
		cosfi = native_cos(arg);
		sintheta = 0.25f * lambda * qi[get_local_id(1)] / PIf; //q = 4pi/lambda*sin(theta)
		costheta = 1.f - SQR(sintheta); //theta in [0, pi/2];
		qv = (float4)(costheta * cosfi, costheta * sinfi, -sintheta, 0) * qi[get_local_id(1)];//computing the scattering vector
		//instead of pre-multiplying the atomic coordinates by the rotational matrix we are pre-multiplying the scattering vector by the transposed rotational matrix (dot(qv,r) will be the same)
		qv = (float4)(dot(qv, CS[0]),dot(qv, CS[1]),dot(qv, CS[2]),0);
	}
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(Nfin - iter * SizeR, SizeR); //checking for the margins of the atomic coordinates array
		if (get_local_id(1) < SizeR / BlockSize2D) {
			unsigned int iAtom = get_local_id(1) * BlockSize2D + get_local_id(0);
			if (iAtom < NiterFin) r[iAtom] = ra[Nst + iter * SizeR + iAtom]; //loading the atomic coordinates to the shared memory
		}
		barrier(CLK_LOCAL_MEM_FENCE); //synchronizing after loading to shared memory
		if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
			for (unsigned int iAtom = 0; iAtom < NiterFin; iAtom++){
				float arg = dot(qv, r[iAtom]);
				sinfi = native_sin(arg);
				cosfi = native_cos(arg);
				lAr += cosfi; //real part of the amplitute
				lAi += sinfi; //imaginary part of the amplitute
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE); //synchronizing before the next loading starts
	}
	if ((iq < Nq) && (ifi < Nfi)){//checking the 2d array margins
		Ar[iq * Nfi + ifi] += lFF[get_local_id(1)] * lAr; //multiplying the real part of the amplitude by the form-factor and writing the results to the global memory
		Ai[iq * Nfi + ifi] += lFF[get_local_id(1)] * lAi; //doing the same for the imaginary part of the amplitude
	}
}

/**
	Computes the real and imaginary parts of the 2D x-ray scattering amplitude in the polar coordinates (q,q_fi) of the reciprocal space

	@param *Ar    Real part of the 2D scattering amplitude array
	@param *Ai	  Imaginary part of the 2D scattering amplitude array
	@param *q     Scattering vector magnitude array
	@param Nq     Size of the scattering vector magnitude mesh (number of rows in the 2D amplitude array)
	@param Nfi    Size of the scattering vector polar angle mesh (number of columns in the 2D amplitude array)
	@param CS[]   Transposed rotational matrix. Defines the orientation of the nanoparticle in the 3D space.
	@param lambda Wavelength of the source
	@param *ra    Atomic coordinate array
	@param Nst    Index of the 1st atom in ra array for this kernel call (the kernel is called iteratively in the loop)
	@param Nfin   Number of atoms to compute for in this kernel call (less or equal to the total number of atoms)
	@param SL     Neutron scattering length of the current chemical element (for one kernel call the computations are done only for the atoms of the same chemical element)
*/
__kernel void calcInt2DKernelNeutron(__global float *Ar, __global float *Ai, const __global float *q, unsigned int Nq, unsigned int Nfi, const __global float4 *CS, float lambda, const __global float4 *ra, unsigned int Nst, unsigned int Nfin, float SL){
	//see comments in the calcInt2DKernelXray() kernel
	unsigned int iq = get_local_size(1) * get_group_id(1) + get_local_id(1), ifi = get_local_size(0) * get_group_id(0) + get_local_id(0);
	unsigned int iqCopy = get_local_size(1) * get_group_id(1) + get_local_id(0);
	__local float qi[BlockSize2D];
	__local float4 r[SizeR];
	unsigned int Niter = Nfin / SizeR + BOOL(Nfin % SizeR);
	float4 qv;
	float lAr = 0, lAi = 0, cosfi = 0, sinfi = 0, sintheta = 0, costheta = 0;
	if ((get_local_id(1) == 4) && (iqCopy < Nq)) qi[get_local_id(0)] = q[iqCopy];
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((iq < Nq) && (ifi < Nfi)){
		float arg = ifi * 2.f * PIf / Nfi;
		sinfi = native_sin(arg);
		cosfi = native_cos(arg);
		sintheta = 0.25f * lambda * qi[get_local_id(1)] / PIf;
		costheta = 1.f - SQR(sintheta);
		qv = (float4)(costheta * cosfi, costheta * sinfi, -sintheta,0) * qi[get_local_id(1)];
		qv = (float4)(dot(qv, CS[0]), dot(qv, CS[1]), dot(qv, CS[2]),0);
	}
	for (unsigned int iter = 0; iter < Niter; iter++){
		unsigned int NiterFin = MIN(Nfin - iter * SizeR, SizeR);
		if (get_local_id(1) < SizeR / BlockSize2D) {
			unsigned int iAtom = get_local_id(1) * BlockSize2D + get_local_id(0);
			if (iAtom < NiterFin) r[iAtom] = ra[Nst + iter * SizeR + iAtom];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if ((iq < Nq) && (ifi < Nfi)){
			for (unsigned int iAtom = 0; iAtom < NiterFin; iAtom++){
				float arg = dot(qv, r[iAtom]);
				sinfi = native_sin(arg);
				cosfi = native_cos(arg);
				lAr += cosfi;
				lAi += sinfi;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if ((iq < Nq) && (ifi < Nfi)){
		Ar[iq * Nfi + ifi] += SL * lAr;
		Ai[iq * Nfi + ifi] += SL * lAi;
	}
}