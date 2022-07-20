#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <float.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pthread.h>
#include <stdbool.h>

#define MINSNPS_B 5
#define MAXSNPS_E 20
#define UNROLL 4
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MAX4(a,b,c,d) MAX(d, MAX(a,MAX(b,c)))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN4(a,b,c,d) MIN(d, MIN(a,MIN(b,c)))

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

#define EXIT 127
#define BUSYWAIT 0
#define STARTITERATIONS 1

double gettime(void);
float randpval (void);
void sense_reversal_barrier_init (int num_threads);
void sense_reversal_barrier (int tid, int num_threads);
void sense_reversal_barrier_destroy (void);

static int count;
static bool sense;
static pthread_t * workerThread;
int iterationsPerThread  ;
int extraIterations  ;
int totalIterations ;

typedef struct localsense_t
{
	bool lsense;
} localsense_t;

typedef struct
{
 	__m128* mVec_vec_ptr ;
	__m128* nVec_vec_ptr ;
	__m128* LVec_vec_ptr ;
	__m128* RVec_vec_ptr ;
	__m128* CVec_vec_ptr ;
	__m128* FVec_vec_ptr ;
	float * maxF ;
	float * minF ;
	float * avgF ;

} iterData_t;

typedef struct
{
	int threadID;
	int threadTOTAL;
	int threadOPERATION;
	iterData_t iterData;

} threadData_t;

static inline void startIterations (threadData_t * threadData);

static localsense_t *localsense_list=NULL;

void sense_reversal_barrier_init (int num_threads)
{
	int i;
	sense = true;
	count = num_threads;
	if (localsense_list==NULL)
	{
		localsense_list = (localsense_t *)malloc(sizeof(localsense_t)*(unsigned long)num_threads);
		assert(localsense_list!=NULL);
	}
	for (i=0; i<num_threads; i++)
	{
		localsense_list[i].lsense=true;
	}
}

void sense_reversal_barrier (int tid, int num_threads)
{
	int threadno = tid;
	localsense_list[threadno].lsense = !localsense_list[threadno].lsense;

	if (__sync_fetch_and_sub (&count, 1) == 1) {
		count = num_threads;
		sense = localsense_list[threadno].lsense;
	}
	else
	{
		while(sense != localsense_list[threadno].lsense) __sync_synchronize();
	}
}

void sense_reversal_barrier_destroy (void)
{
	if(localsense_list!=NULL)
		free(localsense_list);

	localsense_list=NULL;
}

void initializeThreadData(threadData_t * cur, int i, int threads, __m128 * mVec_vec_ptr,__m128 * nVec_vec_ptr, __m128 * LVec_vec_ptr,__m128 * RVec_vec_ptr,__m128 * CVec_vec_ptr,__m128 * FVec_vec_ptr)
{
	cur->threadID=i;
	cur->threadTOTAL=threads;
	cur->threadOPERATION=BUSYWAIT;

	cur->iterData.mVec_vec_ptr = mVec_vec_ptr ;
	cur->iterData.nVec_vec_ptr = nVec_vec_ptr;
	cur->iterData.LVec_vec_ptr = LVec_vec_ptr;
	cur->iterData.RVec_vec_ptr = RVec_vec_ptr;
	cur->iterData.CVec_vec_ptr = CVec_vec_ptr ;
	cur->iterData.FVec_vec_ptr = FVec_vec_ptr;

	cur->iterData.maxF = (float*)_mm_malloc(sizeof(float)*4,16); 
	cur->iterData.minF = (float*)_mm_malloc(sizeof(float)*4,16); 
	cur->iterData.avgF = (float*)_mm_malloc(sizeof(float)*4,16); 
}

static inline void syncThreadsBARRIER(threadData_t * threadData)
{
	int i, threads = threadData[0].threadTOTAL ;
	threadData[0].threadOPERATION=BUSYWAIT;
	sense_reversal_barrier(0, threads);
}

static inline void setThreadOperation(threadData_t * threadData, int operation)
{
	int i, threads=threadData[0].threadTOTAL;

	for(i=0;i<threads;i++) // We unstuck the threads
	{
		threadData[i].threadOPERATION = operation;
	}
}

void startThreadOperations(threadData_t * threadData, int operation)
{
	setThreadOperation(threadData, operation);
	startIterations(&threadData[0]);
	syncThreadsBARRIER(threadData);
}

void * thread (void * x)
{
	threadData_t * currentThread = (threadData_t *) x;
	int tid = currentThread->threadID;
	int threads = currentThread->threadTOTAL;

	while(1)
	{
		__sync_synchronize();

		if(currentThread->threadOPERATION==EXIT)
			return NULL;

		if(currentThread->threadOPERATION==STARTITERATIONS)
		{
			startIterations (currentThread);
			currentThread->threadOPERATION=BUSYWAIT;
			sense_reversal_barrier(tid, threads);
		}
	}

	return NULL;
}

void terminateWorkerThreads(pthread_t * workerThreadL, threadData_t * threadData)
{
	int i, threads=threadData[0].threadTOTAL;

	for(i=0;i<threads;i++)
	{
		threadData[i].threadOPERATION = EXIT;
		_mm_free(threadData[i].iterData.maxF);
		_mm_free(threadData[i].iterData.minF);
		_mm_free(threadData[i].iterData.avgF);
	}		

	for(i=1;i<threads;i++)
		pthread_join(workerThreadL[i-1],NULL);
}

static inline void startIterations (threadData_t * threadData)
{
	int threadID = threadData->threadID;
	int totalThreads = threadData->threadTOTAL;
	float final_min[4] , final_max[4] , final_avg[4] ;

	__m128* mVec_vec_ptr = threadData->iterData.mVec_vec_ptr ;
	__m128* nVec_vec_ptr = threadData->iterData.nVec_vec_ptr ;
	__m128* LVec_vec_ptr = threadData->iterData.LVec_vec_ptr ;
	__m128* RVec_vec_ptr = threadData->iterData.RVec_vec_ptr ;
	__m128* CVec_vec_ptr = threadData->iterData.CVec_vec_ptr ;
	__m128* FVec_vec_ptr = threadData->iterData.FVec_vec_ptr ;

	__m128* avg_vec_ptr = (__m128*) threadData->iterData.avgF;
	__m128* max_vec_ptr = (__m128*) threadData->iterData.maxF;
	__m128* min_vec_ptr = (__m128*) threadData->iterData.minF;

	avg_vec_ptr[0] = _mm_setzero_ps() ;
	max_vec_ptr[0] = _mm_setzero_ps() ;
	min_vec_ptr[0] = _mm_set_ps1(FLT_MAX) ;

	int start ;
	int stop ;

	if (threadID == 0) // (iterationsPerThread  + extraIterations) iterations
	{
		start = 0 ;
		stop = iterationsPerThread - 1  + extraIterations ;
	}
	else // (iterationsPerThread) iterations
	{
		start  = (threadID * iterationsPerThread) + extraIterations ;
		stop = start + iterationsPerThread - 1 ;
	}

	//printf("The thread %d works in [%d,%d] \n",threadID,start,stop);

	for(unsigned int i=start;i<stop;i++)
	{
		FVec_vec_ptr[i] = _mm_div_ps (_mm_div_ps (_mm_add_ps (LVec_vec_ptr[i], RVec_vec_ptr[i]), _mm_add_ps (_mm_mul_ps (mVec_vec_ptr[i], (mVec_vec_ptr[i] - 1.0f)/2.0f), _mm_mul_ps (nVec_vec_ptr[i], (nVec_vec_ptr[i] - 1.0f)/2.0f))), _mm_div_ps (_mm_sub_ps (CVec_vec_ptr[i], _mm_add_ps (LVec_vec_ptr[i], RVec_vec_ptr[i])), _mm_mul_ps (mVec_vec_ptr[i], nVec_vec_ptr[i])) + 0.01f);
		max_vec_ptr[0] = _mm_max_ps(max_vec_ptr[0],FVec_vec_ptr[i]);
		min_vec_ptr[0] = _mm_min_ps(min_vec_ptr[0],FVec_vec_ptr[i]);
		avg_vec_ptr[0] = _mm_add_ps(avg_vec_ptr[0],FVec_vec_ptr[i]);
	}
}

int main(int argc, char ** argv)
{
	// A) Memory Allocation

	//assert(argc==3);
	double timeTotalMainStart = gettime();
	unsigned int N = (unsigned int)atoi(argv[1]);
	unsigned int NumOfThreads = (unsigned int)atoi(argv[2]);
	unsigned int iters = 10;
	srand(1);

	totalIterations = N/UNROLL ;
	iterationsPerThread = totalIterations / NumOfThreads ;
	extraIterations = totalIterations % NumOfThreads  ;

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

	__m128* mVec_vec_ptr = (__m128*) mVec;
	__m128* nVec_vec_ptr = (__m128*) nVec;
	__m128* LVec_vec_ptr = (__m128*) LVec;
	__m128* RVec_vec_ptr = (__m128*) RVec;
	__m128* CVec_vec_ptr = (__m128*) CVec;
	__m128* FVec_vec_ptr = (__m128*) FVec;

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

	sense_reversal_barrier_init (NumOfThreads);

	workerThread = (pthread_t *) malloc (sizeof(pthread_t)*((unsigned long)(NumOfThreads-1)));
	assert(workerThread!=NULL);
	threadData_t * threadData = (threadData_t *) malloc (sizeof(threadData_t)*((unsigned long)NumOfThreads));
	assert(threadData!=NULL);

	for(int i=0;i<NumOfThreads;i++)
		initializeThreadData(&threadData[i],i,NumOfThreads,mVec_vec_ptr,nVec_vec_ptr,LVec_vec_ptr,RVec_vec_ptr,CVec_vec_ptr,FVec_vec_ptr);

	for(int i=1;i<NumOfThreads;i++)
		pthread_create(&workerThread[i-1], NULL, thread, (void *) (&threadData[i]));

	for(unsigned int j=0;j<iters;j++)
		startThreadOperations(threadData,STARTITERATIONS);

	float min = FLT_MAX ;
	float max = 0.0f ;
	float avg = 0.0f ;

	for (int i = 0 ; i< NumOfThreads ; i++)
	{
		min = MIN(MIN4(threadData[i].iterData.minF[0],threadData[i].iterData.minF[1],threadData[i].iterData.minF[2],threadData[i].iterData.minF[3]),min) ;
		max = MAX(MAX4(threadData[i].iterData.maxF[0],threadData[i].iterData.maxF[1],threadData[i].iterData.maxF[2],threadData[i].iterData.maxF[3]),max) ;
		avg = ((float)(threadData[i].iterData.avgF[0] + threadData[i].iterData.avgF[1] + threadData[i].iterData.avgF[2] + threadData[i].iterData.avgF[3])) + avg;
	}

	double timeOmegaTotal = gettime()-timeOmegaTotalStart;
	double timeTotalMainStop = gettime();
	printf("Omega time %fs - Total time %fs - Min %e - Max %e - Avg %e\n",timeOmegaTotal/iters, timeTotalMainStop-timeTotalMainStart,(double)min,(double)max,(double)avg/N);

	// Release the unneeded resources

	terminateWorkerThreads(workerThread,threadData);
	sense_reversal_barrier_destroy();

	free(workerThread) ;
	free(threadData) ;
	_mm_free(mVec);
	_mm_free(nVec);
	_mm_free(LVec);
	_mm_free(RVec);
	_mm_free(CVec);
	_mm_free(FVec);
}
