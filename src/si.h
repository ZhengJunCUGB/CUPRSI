// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "su.h"
#include "segy.h"
#include "header.h"


typedef float2 Complex;
typedef struct _sipar {
/* su header */
	unsigned long ns = 30001;
	unsigned int ntr = 201;
	unsigned int dt = 4000;
	unsigned int outns = 1001;

/* si par*/
	unsigned int file_mode;
	unsigned int ns_win = 1001;
	unsigned int StartFldr = 1;
	unsigned int EndFldr = 201;
	unsigned int num_ns_win = 0;
	unsigned int Stack_flag;

/* Computing resource parameters */
	unsigned int RAMSizeMB;//MB
	unsigned int VRAMSizeMB;
	unsigned int Device;//GPU 
	dim3 threads;
	dim3 blocks;
} sipar;

void siCUFFT(sipar par, FILE* in);
