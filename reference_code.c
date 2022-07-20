#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>

#define MINSNPS_B 5
#define MAXSNPS_E 20

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
	float avgF = 0.0f;
	float maxF = 0.0f;
	float minF = FLT_MAX;
	unsigned int N = (unsigned int)atoi(argv[1]);
	unsigned int iters = 10;
	srand(1); // The srand() function sets the starting point for producing
	//a series of pseudo-random integers. If srand() is not called, the rand()
	// seed is set as if srand(1) were called at program start. Any other value
	//for seed sets the generator to a different starting point.

	// Defines the 6 N-Size float arrays
	float * mVec = (float*)malloc(sizeof(float)*N);
	assert(mVec!=NULL);
	float * nVec = (float*)malloc(sizeof(float)*N);
	assert(nVec!=NULL);
	float * LVec = (float*)malloc(sizeof(float)*N);
	assert(LVec!=NULL);
	float * RVec = (float*)malloc(sizeof(float)*N);
	assert(RVec!=NULL);
	float * CVec = (float*)malloc(sizeof(float)*N);
	assert(CVec!=NULL);
	float * FVec = (float*)malloc(sizeof(float)*N);
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

	double timeOmegaTotalStart = gettime();
	// C) Calculate w

	for(unsigned int j=0;j<iters;j++) // This is the one we will have to change
	{
		avgF = 0.0f;
		maxF = 0.0f;
		minF = FLT_MAX;

		for(unsigned int i=0;i<N;i++)
		{
			// Calculate num
			float num_0 = LVec[i]+RVec[i];
			float num_1 = mVec[i]*(mVec[i]-1.0f)/2.0f;
			float num_2 = nVec[i]*(nVec[i]-1.0f)/2.0f;
			float num = num_0/(num_1+num_2);

			// Calculate den
			float den_0 = CVec[i]-LVec[i]-RVec[i];
			float den_1 = mVec[i]*nVec[i];
			float den = den_0/den_1;

			//Calculate w
			FVec[i] = num/(den+0.01f);

			// Find the max , min and avg w
			maxF = FVec[i]>maxF?FVec[i]:maxF;
			minF = FVec[i]<minF?FVec[i]:minF;
			avgF += FVec[i];
		}
	}

	double timeOmegaTotal = gettime()-timeOmegaTotalStart;
	double timeTotalMainStop = gettime();
	printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n",timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart, (double)minF, (double)maxF,	(double)avgF/N);
	free(mVec);
	free(nVec);
	free(LVec);
	free(RVec);
	free(CVec);
	free(FVec);
}
