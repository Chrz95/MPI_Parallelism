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
	float * Data = (float*)_mm_malloc(sizeof(float)*6*N,16);
	assert(Data!=NULL);

	// B) Initialization

	for(unsigned int i=0;i<N/4;i++)
	{
		// 1
		Data[24*i + 8] = (float)(MINSNPS_B+rand()%MAXSNPS_E); 

		Data[24*i + 12] = (float)(MINSNPS_B+rand()%MAXSNPS_E); 

		Data[24*i + 0] = randpval()*Data[24*i + 8];

		Data[24*i + 4] = randpval()*Data[24*i + 12];

		Data[24*i + 16] = randpval()*Data[24*i + 8]*Data[24*i + 12];

		Data[24*i + 20] = 0.0;

		// 2

		Data[24*i + 9] = (float)(MINSNPS_B+rand()%MAXSNPS_E);

		Data[24*i + 13] = (float)(MINSNPS_B+rand()%MAXSNPS_E); 

		Data[24*i + 1] = randpval()*Data[24*i + 9];

		Data[24*i + 5] = randpval()*Data[24*i + 13];

		Data[24*i + 17] = randpval()*Data[24*i + 9]*Data[24*i + 13];

		Data[24*i + 21] = 0.0;

		// 3

		Data[24*i + 10] = (float)(MINSNPS_B+rand()%MAXSNPS_E); 

		Data[24*i + 14] = (float)(MINSNPS_B+rand()%MAXSNPS_E); 

		Data[24*i + 2] = randpval()*Data[24*i + 10];

		Data[24*i + 6] = randpval()*Data[24*i + 14];

		Data[24*i + 18] = randpval()*Data[24*i + 10]*Data[24*i + 14];

		Data[24*i + 22] = 0.0;

		// 4		

		Data[24*i + 11] = (float)(MINSNPS_B+rand()%MAXSNPS_E); 

		Data[24*i + 15] = (float)(MINSNPS_B+rand()%MAXSNPS_E); 

		Data[24*i + 3] = randpval()*Data[24*i + 11];

		Data[24*i + 7] = randpval()*Data[24*i + 15];

		Data[24*i + 19] = randpval()*Data[24*i + 11]*Data[24*i + 15];
		
		Data[24*i + 23] = 0.0;
	}

	register __m128 avgF = _mm_setzero_ps() ;
	register __m128 maxF = _mm_setzero_ps() ;
	register __m128 minF = _mm_set_ps1(FLT_MAX) ;

	double timeOmegaTotalStart = gettime();

	register __m128 L_vec ;
	register __m128 R_vec ;  
	register __m128 m_vec ;
	register __m128 n_vec ; 
	register __m128 F_vec ; 

	// C) Calculate w

	__m128* Data_ptr = (__m128*) Data;

	for(unsigned int j=0;j<iters;j++)
	{
		avgF = _mm_setzero_ps() ;
		maxF = _mm_setzero_ps() ;
		minF = _mm_set_ps1(FLT_MAX) ;

		for(unsigned int i=0;i<(N/UNROLL);i++)
		{
  			L_vec = Data_ptr[6*i + 0] ;
			R_vec = Data_ptr[6*i + 1] ;  
			m_vec = Data_ptr[6*i + 2] ;
			n_vec = Data_ptr[6*i + 3] ; 
			F_vec = Data_ptr[6*i + 5] ; 

			F_vec  = _mm_div_ps (_mm_div_ps (_mm_add_ps (L_vec, R_vec), _mm_add_ps (_mm_mul_ps (m_vec, (m_vec - 1.0f)/2.0f), _mm_mul_ps (n_vec, (n_vec - 1.0f)/2.0f))), _mm_div_ps (_mm_sub_ps (Data_ptr[6*i + 4], _mm_add_ps (L_vec, R_vec)), _mm_mul_ps (m_vec, n_vec)) + 0.01f);

			maxF = _mm_max_ps(maxF,F_vec);
			minF = _mm_min_ps(minF,F_vec);
			avgF = _mm_add_ps(avgF,F_vec);
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
	_mm_free(Data);
}
