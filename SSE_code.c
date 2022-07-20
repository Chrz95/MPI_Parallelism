#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#define MINSNPS_B 5
#define MAXSNPS_E 20
#define UNROLL 4
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MAX4(a,b,c,d) MAX(d, MAX(a,MAX(b,c)))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN4(a,b,c,d) MIN(d, MIN(a,MIN(b,c)))

double gettime(void);
float randpval (void);

double gettime(void)
{
	struct timeval ttime;
	gettimeofday(&ttime , NULL);
	return ttime.tv_sec + ttime.tv_usec * 0.000001;
}

float randpval (void) // Generate a value from 0 to 1.00001
{
	int vr = rand();
	int vm = rand()%vr;
	float r = ((float)vm)/(float)vr;
	assert(r>=0.0f && r<=1.00001f);
	return r;
}

int main(int argc, char ** argv)
{
	// A) Memory Allocation
	assert(argc==2); // There must be two arguments for the program to run
	double timeTotalMainStart = gettime();
	unsigned int N = (unsigned int)atoi(argv[1]);
	unsigned int iters = 10;
	srand(1);

	// Defines the 6 N-Size float arrays
	float * mVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(mVec!=NULL);
	float * nVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(nVec!=NULL);
	float * LVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(LVec!=NULL);
	float * RVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(RVec!=NULL);
	float * CVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(CVec!=NULL);
	float * FVec = (float*)_mm_malloc(sizeof(float)*N,16);
	assert(FVec!=NULL);

	// B) Initialization

	for(unsigned int i=0;i<N;i++)
	{
		// Fills the two arrays with random floats in [5,20 + 5]
		mVec[i] = (float)(MINSNPS_B+rand()%MAXSNPS_E);
		nVec[i] = (float)(MINSNPS_B+rand()%MAXSNPS_E);

		LVec[i] = randpval()*mVec[i];
		RVec[i] = randpval()*nVec[i];
		CVec[i] = randpval()*mVec[i]*nVec[i];
		FVec[i] = 0.0;

		assert(mVec[i]>=MINSNPS_B && mVec[i]<=(MINSNPS_B+MAXSNPS_E));
		assert(nVec[i]>=MINSNPS_B && nVec[i]<=(MINSNPS_B+MAXSNPS_E));
		assert(LVec[i]>=0.0f && LVec[i]<=1.0f*mVec[i]);
		assert(RVec[i]>=0.0f && RVec[i]<=1.0f*nVec[i]);
		assert(CVec[i]>=0.0f && CVec[i]<=1.0f*mVec[i]*nVec[i]);
	}

	__m128 avgF = _mm_setzero_ps() ;
	__m128 maxF = _mm_setzero_ps() ;
	__m128 minF = _mm_set_ps1(FLT_MAX) ;

	double timeOmegaTotalStart = gettime();

	// C) Calculate w

	__m128* mVec_vec_ptr = (__m128*) mVec;
	__m128* nVec_vec_ptr = (__m128*) nVec;
	__m128* LVec_vec_ptr = (__m128*) LVec;
	__m128* RVec_vec_ptr = (__m128*) RVec;
	__m128* CVec_vec_ptr = (__m128*) CVec;
	__m128* FVec_vec_ptr = (__m128*) FVec;

	for(unsigned int j=0;j<iters;j++)
	{
		avgF = _mm_setzero_ps() ;
		maxF = _mm_setzero_ps() ;
		minF = _mm_set_ps1(FLT_MAX) ;

		for(unsigned int i=0;i<(N/UNROLL);i++)
		{
			FVec_vec_ptr[i] = _mm_div_ps (_mm_div_ps (_mm_add_ps (LVec_vec_ptr[i], RVec_vec_ptr[i]), _mm_add_ps (_mm_mul_ps (mVec_vec_ptr[i], (mVec_vec_ptr[i] - 1.0f)/2.0f), _mm_mul_ps (nVec_vec_ptr[i], (nVec_vec_ptr[i] - 1.0f)/2.0f))), _mm_div_ps (_mm_sub_ps (CVec_vec_ptr[i], _mm_add_ps (LVec_vec_ptr[i], RVec_vec_ptr[i])), _mm_mul_ps (mVec_vec_ptr[i], nVec_vec_ptr[i])) + 0.01f);

			maxF = _mm_max_ps(maxF,FVec_vec_ptr[i]);
			minF = _mm_min_ps(minF,FVec_vec_ptr[i]);
			avgF = _mm_add_ps(avgF,FVec_vec_ptr[i]);
		}
	}

	float final_max[4] , final_min[4] , final_avg[4] ;
	_mm_store_ps(final_max,maxF);
	_mm_store_ps(final_min,minF);
	_mm_store_ps(final_avg,avgF);

	float min = MIN4(final_min[0],final_min[1],final_min[2],final_min[3]) ;
	float max = MAX4(final_max[0],final_max[1],final_max[2],final_max[3]) ;
	float avg = (float)(final_avg[0] + final_avg[1] + final_avg[2] + final_avg[3])/(float)N ;

	double timeOmegaTotal = gettime()-timeOmegaTotalStart;
	double timeTotalMainStop = gettime();
	//printf("Omega time %fs - Total time %fs\n",timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart);
	printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n\n",timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart,(double)min,(double)max,(double)avg);
	_mm_free(mVec);
	_mm_free(nVec);
	_mm_free(LVec);
	_mm_free(RVec);
	_mm_free(CVec);
	_mm_free(FVec);
}
