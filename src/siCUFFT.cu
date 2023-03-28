#include "si.h"
#include "siCUFFT.h"
void siCUFFT(sipar par, FILE* in)
{
	/* Calculate new size of signals */
	unsigned int new_size = par.ns_win * 2; /* Fourier transform length */

	/* Declare parameters */
	unsigned int k; // Loop variable
	int ifopen = 0; // Judge whether the reading is successful, ifopen=0 is successful.
	unsigned int num_fldr = par.EndFldr - par.StartFldr + 1; // The number of fldr

	/* Calculate the number of time windows which stay in RAM and allocate memory */
	/* The signal in the disk is read into RAM in blocks and the block size is equal to variable num_time_windows_on_RAM multiply Variable ns_win */
	unsigned int num_time_windows_on_RAM; // The number of time windows which stay in RAM
	float** h_signal_all = (float**) malloc (sizeof(float*) * par.ntr); // RAM space of input signals
	float* h_out_all; // RAM space of cross correlated signals
	unsigned short* h_signal_all_value; // Determine whether the input signals are in RAM , h_signal_all_value[i]=1 means the signal is in RAM
	unsigned long size = 2 * new_size * sizeof(Complex) + par.ntr * num_fldr * par.outns * sizeof(float); // RAM space of calculation and cross correlated signals
	unsigned long long remaining_size = par.RAMSizeMB - size / (1024 * 1024); // Remaining RAM space of input signals
	num_time_windows_on_RAM = remaining_size * 1024 * 1024 / par.ntr / (par.ns_win * sizeof(float));
	num_time_windows_on_RAM = (num_time_windows_on_RAM > par.num_ns_win) ? par.num_ns_win : num_time_windows_on_RAM;
	if(num_time_windows_on_RAM <= 0)
	{
		warn("SI    <<< wrong number of time windows on RAM , =%u>>>", num_time_windows_on_RAM);
		return;
	}
	while( par.num_ns_win % num_time_windows_on_RAM != 0)
	{
		num_time_windows_on_RAM --;
	}
	
	checkCudaErrors(cudaHostAlloc(reinterpret_cast<void **>(&h_signal_all_value), sizeof(unsigned short) * par.ntr, cudaHostAllocDefault));
	for(unsigned int tracl = 0; tracl < par.ntr; tracl++)
	{
		h_signal_all[tracl] = (float*) malloc (sizeof(float) * num_time_windows_on_RAM * par.ns_win);
		memset(h_signal_all[tracl], 0, sizeof(float) * num_time_windows_on_RAM * par.ns_win);
		checkCudaErrors(cudaHostRegister(h_signal_all[tracl],sizeof(float) * num_time_windows_on_RAM * par.ns_win, 0));
	}
	checkCudaErrors(cudaHostAlloc(reinterpret_cast<void **>(&h_out_all), sizeof(float) * (long long)par.ntr * num_fldr * par.outns, cudaHostAllocDefault));
	
	//getLastCudaError("Allocation failed [ h_signal_all_value ]");
	cudaMemset(h_signal_all_value, 0, sizeof(unsigned short) * par.ntr);
	cudaMemset(h_out_all, 0, sizeof(float) * (long long)par.ntr * num_fldr * par.outns);
	//getLastCudaError("Memset failed [ h_signal_all_value ]");

	/* Calculate the number of time windows which stay in VRAM all the time */
	unsigned int num_time_windows_on_VRAM; // The number of time windows which stay in VRAM
	size = 2 * sizeof(Complex) * new_size;
	remaining_size = par.VRAMSizeMB - size / (1024 * 1024) - 135 * 2; //VRAM for CUFFT is 135MB 
	num_time_windows_on_VRAM = remaining_size * 1024 * 1024 / ((2 * sizeof(float) * par.ns_win) + (5 * sizeof(Complex) * new_size));
	num_time_windows_on_VRAM = (num_time_windows_on_VRAM > par.num_ns_win) ? par.num_ns_win : num_time_windows_on_VRAM;
	if(num_time_windows_on_VRAM <= 0)
	{
		warn("SI    <<< wrong num of time windows on VRAM , =%u>>>", num_time_windows_on_VRAM);
		return;
	}
	while(num_time_windows_on_RAM % num_time_windows_on_VRAM != 0)
	{
		num_time_windows_on_VRAM --;
	}
	num_time_windows_on_RAM = (num_time_windows_on_RAM > num_time_windows_on_VRAM) ? num_time_windows_on_VRAM : num_time_windows_on_RAM;
	warn("SI    <<< %u time windows stay in RAM >>>", num_time_windows_on_RAM);
	warn("SI    <<< %u time windows stay in VRAM >>>", num_time_windows_on_VRAM);

	/* Read file list if mode = 1 */
	char **file_name; // RAM space of all file names
	char *line; // RAM space of a file name
	unsigned int count_file = 0; // Variable to count
	if(par.file_mode == 1)
	{
		file_name = (char **) malloc (par.ntr * sizeof (char*));
		line = (char *) malloc (500 * sizeof (char));
		while (fgets (line, 500, in))
		{
			file_name[count_file] = (char *) malloc (500 * sizeof (char));
			sscanf (line, "%s", file_name[count_file]);
			count_file++;
		}
		warn("SI    <<< count_file =%u >>>", count_file);
		free(line);
	}

	/* Allocate host space for cross correlated signals */
	Complex* h_sumdata0; // RAM space of cross correlated signals (stream0)
	Complex* h_sumdata1; // RAM space of cross correlated signals (stream1)
	checkCudaErrors(cudaHostAlloc(reinterpret_cast<void **>(&h_sumdata0), sizeof(Complex) * new_size, cudaHostAllocDefault));
	checkCudaErrors(cudaHostAlloc(reinterpret_cast<void **>(&h_sumdata1), sizeof(Complex) * new_size, cudaHostAllocDefault));
	//getLastCudaError("Allocation failed [ h_sumdata0 ]");
	
	/* Allocate device space */
	float* d_signal0; // VRAM space of the input signals (stream0)
	Complex* d_shot0; // VRAM space of the input signal 1 (stream0)
	Complex* d_tracl0; // VRAM space of the input signal 2 (stream0)
	Complex* d_sumdata0; // VRAM space of cross correlated signals (stream0)
	//s1(w)*s2*(w)
	float* d_signal1; // VRAM space of the input signals (stream1)
	Complex* d_tracl1; // VRAM space of the input signal 2 (stream1)
	Complex* d_sumdata1; // VRAM space of cross correlated signals (stream1)
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal0), sizeof(float) * (long long)par.ns_win * num_time_windows_on_RAM));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_shot0), sizeof(Complex) * (long long)new_size * num_time_windows_on_RAM));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_tracl0), sizeof(Complex) * (long long)new_size * num_time_windows_on_RAM));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sumdata0), sizeof(Complex) * new_size));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_signal1), sizeof(float) * (long long)par.ns_win * num_time_windows_on_RAM));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_tracl1), sizeof(Complex) * (long long)new_size * num_time_windows_on_RAM));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sumdata1), sizeof(Complex) * new_size));
	//getLastCudaError("Allocation failed [ d_signal0 ]");

	/* Initialize stream */
	cudaStream_t stream0;
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	checkCudaErrors(cudaStreamCreate(&stream0));
	checkCudaErrors(cudaStreamCreate(&stream1));
	checkCudaErrors(cudaStreamCreate(&stream2));
	checkCudaErrors(cudaStreamCreate(&stream3));
	//getLastCudaError("Stream creation failed [ stream0 ]");

	/* Create CUFFT plan */
	cufftHandle plan0;
	cufftHandle plan1;
	checkCudaErrors(cufftPlan1d(&plan0, new_size, CUFFT_C2C, num_time_windows_on_RAM));
	checkCudaErrors(cufftPlan1d(&plan1, new_size, CUFFT_C2C, num_time_windows_on_RAM));
	cufftSetStream(plan0, stream0);
	cufftSetStream(plan1, stream1);
	//getLastCudaError("CUFFT plan failed [ cufftPlan1d ]");

	/* Cross correlation */
	unsigned int fldr;
	unsigned int tracl;
	long ns = par.ns;
	unsigned int block_number;
	unsigned int block_offset = num_time_windows_on_RAM * par.ns_win;
	for(block_number = 0; block_number < par.num_ns_win / num_time_windows_on_RAM; block_number++)
	{
		cudaMemset(h_signal_all_value, 0, sizeof(unsigned short) * par.ntr);
		for(fldr = par.StartFldr - 1; fldr < par.EndFldr; ++fldr)
		{
			/* Determine whether to read from disk */
			if(h_signal_all_value[fldr] == 0 )
			{
				if(par.file_mode == 0)
				{
					ReadDataFromDiskModeOne(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, fldr, ns, in);
				}
				else
				{
					ifopen = ReadDataFromDiskModeTwo(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, fldr, file_name);
					if(ifopen != 0)
					{
						return;
					}
				}
			}

			ReadDataFromMem(h_signal_all, num_time_windows_on_RAM, fldr, block_offset, block_number, d_signal0, stream0);
			/* Convert float to complex and pad data */
			cudaMemsetAsync(d_shot0, 0, sizeof(Complex) * new_size * num_time_windows_on_RAM, stream0);
			f2c<<<par.blocks, par.threads, 0, stream0>>>(d_signal0, d_shot0, par.ns_win, new_size, num_time_windows_on_RAM);
			//getLastCudaError("Kernel execution failed [ f2c ]");
			/* Convert time domain data to frequency domain on device */
			checkCudaErrors(cufftExecC2C(plan0, reinterpret_cast<cufftComplex *>(d_shot0), reinterpret_cast<cufftComplex *>(d_shot0), CUFFT_FORWARD));
	
			/* Determine whether to read from disk */
			if(h_signal_all_value[0] == 0)
			{
			
				if(par.file_mode == 0)
				{
					ReadDataFromDiskModeOne(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, 0, ns, in);
				}
				else
				{
					ifopen = ReadDataFromDiskModeTwo(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, 0, file_name);
					if(ifopen != 0)
					{
						return;
					}
				}
			}
			if(h_signal_all_value[1] == 0)
			{
				if(par.file_mode == 0)
				{
					ReadDataFromDiskModeOne(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, 1, ns, in);
				}
				else
				{
					ifopen = ReadDataFromDiskModeTwo(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, 1, file_name);
					if(ifopen != 0)
					{
						return;
					}
				}
			}
			
			ReadDataFromMem(h_signal_all, num_time_windows_on_RAM, 1, block_offset, block_number, d_signal1, stream1);
			cudaStreamSynchronize(stream0);
			ReadDataFromMem(h_signal_all, num_time_windows_on_RAM, 0, block_offset, block_number, d_signal0, stream0);
			for(tracl = 0; tracl < par.ntr - 1; tracl = tracl + 2)
			{
				if(tracl != 0)
				{
					cudaStreamSynchronize(stream2);
				}
				/* Convert float to complex and pad data */
				cudaMemsetAsync(d_tracl0, 0, sizeof(Complex) * new_size * num_time_windows_on_RAM, stream0);
				f2c_flip<<<par.blocks, par.threads, 0, stream0>>>(d_signal0, d_tracl0, par.ns_win, new_size, num_time_windows_on_RAM);

				if(tracl != 0)
				{
					cudaStreamSynchronize(stream3);
				}
				
				cudaMemsetAsync(d_tracl1, 0, sizeof(Complex) * new_size * num_time_windows_on_RAM, stream1);
				f2c_flip<<<par.blocks, par.threads, 0, stream1>>>(d_signal1, d_tracl1, par.ns_win, new_size, num_time_windows_on_RAM);
				/* Read next tracl data */
				if(tracl + 3 < par.ntr)
				{
					if(h_signal_all_value[tracl + 2] == 0)
					{
						if(par.file_mode == 0)
						{
							ReadDataFromDiskModeOne(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, tracl + 2, ns, in);
						}
						else
						{
							ifopen = ReadDataFromDiskModeTwo(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, tracl + 2, file_name);
							if(ifopen != 0)
							{
								return;
							}
						}
					}
					if(h_signal_all_value[tracl + 3] == 0)
					{
						if(par.file_mode == 0)
						{
							ReadDataFromDiskModeOne(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, tracl + 3, ns, in);
						}
						else
						{
							ifopen = ReadDataFromDiskModeTwo(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, tracl + 3, file_name);
							if(ifopen != 0)
							{
								return;
							}
						}
					}
					
					cudaStreamSynchronize(stream0);
					ReadDataFromMem(h_signal_all, num_time_windows_on_RAM, tracl + 2, block_offset, block_number, d_signal0, stream2);
					
					cudaStreamSynchronize(stream1);
					ReadDataFromMem(h_signal_all, num_time_windows_on_RAM, tracl + 3, block_offset, block_number, d_signal1, stream3);
				}

				/* Convert time domain data to frequency domain on device */
				checkCudaErrors(cufftExecC2C(plan0, reinterpret_cast<cufftComplex *>(d_tracl0), reinterpret_cast<cufftComplex *>(d_tracl0), CUFFT_FORWARD));
				checkCudaErrors(cufftExecC2C(plan1, reinterpret_cast<cufftComplex *>(d_tracl1), reinterpret_cast<cufftComplex *>(d_tracl1), CUFFT_FORWARD));

				/* Complex pointwise multiplication and complex scale */
				ComplexPointwiseMulAndScale<<<par.blocks, par.threads, 0, stream0>>>(d_shot0, d_tracl0, d_tracl0, num_time_windows_on_RAM, new_size);//Single shot cross correlation with all time windows
				ComplexPointwiseMulAndScale<<<par.blocks, par.threads, 0, stream1>>>(d_shot0, d_tracl1, d_tracl1, num_time_windows_on_RAM, new_size);
				
				/* Check if kernel execution generated and error */
				//getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
				
				/* ICUFFT */
				checkCudaErrors(cufftExecC2C(plan0, reinterpret_cast<cufftComplex *>(d_tracl0), reinterpret_cast<cufftComplex *>(d_tracl0), CUFFT_INVERSE));
				checkCudaErrors(cufftExecC2C(plan1, reinterpret_cast<cufftComplex *>(d_tracl1), reinterpret_cast<cufftComplex *>(d_tracl1), CUFFT_INVERSE));
				/* Add all time windows */
				cudaMemsetAsync(d_sumdata0, 0, sizeof(Complex) * new_size, stream0);
				Add<<<par.blocks, par.threads, 0, stream0>>>(d_sumdata0, d_tracl0, new_size, num_time_windows_on_RAM);//add all time windows of single tracl to a time windows
				cudaMemsetAsync(d_sumdata1, 0, sizeof(Complex) * new_size, stream1);
				Add<<<par.blocks, par.threads, 0, stream1>>>(d_sumdata1, d_tracl1, new_size, num_time_windows_on_RAM);
				checkCudaErrors(cudaMemcpyAsync(h_sumdata0, d_sumdata0, new_size * sizeof(Complex), cudaMemcpyDeviceToHost, stream0));
				checkCudaErrors(cudaMemcpyAsync(h_sumdata1, d_sumdata1, new_size * sizeof(Complex), cudaMemcpyDeviceToHost, stream1));
				/* Write result */
				cudaStreamSynchronize(stream0);
				if(par.Stack_flag == 0)
				{
					for(k = 0; k < par.outns; k++)
					{
						h_out_all[(fldr - par.StartFldr + 1) * par.ntr * par.outns + tracl * par.outns + k] += h_sumdata0[k].x;
					}
				}
				else
				{
					for(k = 0; k < par.outns; k++)
					{
						h_out_all[(fldr - par.StartFldr + 1) * par.ntr * par.outns + tracl * par.outns + k] += h_sumdata0[par.ns_win -1 + k].x + h_sumdata0[par.ns_win - 1 - k].x;
					}
				}
				cudaStreamSynchronize(stream1);
				if(par.Stack_flag == 0)
				{
					for(k = 0; k < par.outns; k++)
					{
						h_out_all[(fldr - par.StartFldr + 1) * par.ntr * par.outns + (tracl + 1) * par.outns + k] += h_sumdata1[k].x;
					}
				}
				else
				{
					for(k = 0; k < par.outns; k++)
					{
						h_out_all[(fldr - par.StartFldr + 1) * par.ntr * par.outns + (tracl + 1) * par.outns + k] += h_sumdata1[par.ns_win - 1 + k].x + h_sumdata1[par.ns_win - 1 - k].x;
					}
				}
			}

			if(par.ntr % 2 == 1)
			{
				tracl = par.ntr - 1;
				if(h_signal_all_value[tracl] == 0)
				{
					if(par.file_mode == 0)
					{
						ReadDataFromDiskModeOne(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, tracl, ns, in);
					}
					else
					{
						ifopen = ReadDataFromDiskModeTwo(h_signal_all_value, h_signal_all, num_time_windows_on_RAM, block_offset, block_number, tracl, file_name);
						if(ifopen != 0)
						{
							return;
						}
					}
				}
				ReadDataFromMem(h_signal_all, num_time_windows_on_RAM, tracl, block_offset, block_number, d_signal0, stream0);

				/* Convert float to complex and pad data */
				cudaMemsetAsync(d_tracl0, 0, sizeof(Complex) * new_size * num_time_windows_on_RAM, stream0);
				f2c_flip<<<par.blocks, par.threads, 0, stream0>>>(d_signal0, d_tracl0, par.ns_win, new_size, num_time_windows_on_RAM);
				/* Convert time domain data to frequency domain on device */
				checkCudaErrors(cufftExecC2C(plan0, reinterpret_cast<cufftComplex *>(d_tracl0), reinterpret_cast<cufftComplex *>(d_tracl0), CUFFT_FORWARD));

				/* Complex pointwise multiplication and complex scale */
				ComplexPointwiseMulAndScale<<<par.blocks, par.threads, 0, stream0>>>(d_shot0, d_tracl0, d_tracl0, num_time_windows_on_RAM, new_size);//Single shot cross correlation with all time windows

				/* Check if kernel execution generated and error */
				//getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
				checkCudaErrors(cufftExecC2C(plan0, reinterpret_cast<cufftComplex *>(d_tracl0), reinterpret_cast<cufftComplex *>(d_tracl0), CUFFT_INVERSE));

				/* Add all time windows */
				cudaMemsetAsync(d_sumdata0, 0, sizeof(Complex) * new_size, stream0);
				Add<<<par.blocks, par.threads, 0, stream0>>>(d_sumdata0, d_tracl0, new_size, num_time_windows_on_RAM);//add all time windows of single tracl to a time windows
				checkCudaErrors(cudaMemcpyAsync(h_sumdata0, d_sumdata0, new_size * sizeof(Complex), cudaMemcpyDeviceToHost, stream0));

				/* Write result */
				cudaStreamSynchronize(stream0);
				if(par.Stack_flag == 0)
				{
					for(k = 0; k < par.outns; k++)
					{
						h_out_all[(fldr - par.StartFldr + 1) * par.ntr * par.outns + tracl * par.outns + k] += h_sumdata0[k].x;
					}
				}
				else
				{
					for(k = 0; k < par.outns; k++)
					{
						h_out_all[(fldr - par.StartFldr + 1) * par.ntr * par.outns + tracl * par.outns + k] += h_sumdata0[par.ns_win - 1 + k].x + h_sumdata0[par.ns_win - 1 - k].x;
					}
				}
			}
		}
	}

	/* Write su file */
	segy outtrace;
	outtrace.ntr = (par.EndFldr - par.StartFldr + 1) * par.ntr;
	outtrace.ns = par.outns;
	outtrace.dt = par.dt;
	for(fldr = par.StartFldr - 1; fldr < par.EndFldr; ++fldr)
	{
		outtrace.fldr = fldr + 1;
		for(tracl = 0; tracl < par.ntr; tracl = tracl + 1)
		{
			outtrace.tracl = tracl + 1;
			for(k = 0; k < par.outns; k++)
			{
				outtrace.data[k] = h_out_all[(fldr - par.StartFldr + 1) * par.ntr * par.outns + tracl * par.outns + k];
			}
			puttr(&outtrace);
		}
	}
	
	/* Free space */
	checkCudaErrors(cufftDestroy(plan0));
	checkCudaErrors(cudaFree(d_signal0));
	checkCudaErrors(cudaFree(d_shot0));
	checkCudaErrors(cudaFree(d_tracl0));
	checkCudaErrors(cudaFree(d_sumdata0));
	checkCudaErrors(cudaFreeHost(h_sumdata0));
	checkCudaErrors(cudaStreamDestroy(stream0));
	checkCudaErrors(cufftDestroy(plan1));
	checkCudaErrors(cudaFree(d_signal1));
	checkCudaErrors(cudaFree(d_tracl1));
	checkCudaErrors(cudaFree(d_sumdata1));
	checkCudaErrors(cudaFreeHost(h_sumdata1));
	checkCudaErrors(cudaStreamDestroy(stream1));
	checkCudaErrors(cudaStreamDestroy(stream2));
	checkCudaErrors(cudaStreamDestroy(stream3));
	checkCudaErrors(cudaFreeHost(h_signal_all_value));
	for(unsigned int tracl = 0; tracl < par.ntr; tracl++)
	{
		free(h_signal_all[tracl]);
	}
	free(h_signal_all);
	checkCudaErrors(cudaFreeHost(h_out_all));
	if(par.file_mode == 1)
	{
		while(count_file > 0)
		{
			count_file--;
			free(file_name[count_file]);
		}
		free(file_name);
	}
}

static __global__ void f2c(float *a, Complex *b, unsigned int size, unsigned int new_size, unsigned int num_ns_win)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for(unsigned int win = 0; win < num_ns_win; ++win)
	{
		for (unsigned int i = threadID; i < size; i += numThreads)
		{
			b[win * new_size + i].x = a[win * size + i];
		}
	}
}

static __global__ void f2c_flip(float *a, Complex *b, unsigned int size, unsigned int new_size, unsigned int num_ns_win)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for(unsigned int win = 0; win < num_ns_win; ++win)
	{
		for (unsigned int i = threadID; i < size; i += numThreads)
		{
			b[win * new_size + size - 1 - i].x = a[win * size + i];
		}
	}
}


static __global__ void ComplexPointwiseMulAndScale(Complex *d_shot, Complex *d_signal, Complex *d_out_f, unsigned int num_ns_win, unsigned int new_size)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for(unsigned int win = 0; win < num_ns_win; ++win)
	{
		for (unsigned int i = threadID; i < new_size; i += numThreads)
		{ 
			d_out_f[win * new_size + i] = ComplexScale(ComplexMul(d_shot[win * new_size + i], d_signal[win * new_size + i]), 1.0f / (float)new_size);
		}
	}
}

static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
	Complex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

static __global__ void Add(Complex *a, Complex *b, unsigned int size, unsigned int num_ns_win)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for(unsigned int win = 0; win < num_ns_win; ++win)
	{
		for (unsigned int i = threadID; i < size; i += numThreads)
		{
			a[i].x = a[i].x + b[win * size + i].x;
			a[i].y = a[i].y + b[win * size + i].y;
		}
	}
}

void ReadDataFromDiskModeOne(unsigned short *h_signal_all_value, float **h_signal_all, unsigned int num_time_windows_on_RAM, unsigned int block_offset, unsigned int block_number, unsigned int tracl, long ns, FILE *in)
{
	/* Read data from binary file */

	unsigned int ifseek;
	unsigned int iread;
	long long offset = (long long)tracl * ns * sizeof(float) + (long long)block_number * block_offset * sizeof(float);
	ifseek = fseeko64(in, offset, SEEK_SET);
	if(ifseek != 0)
	{
		warn(" fseek failed tracl = %u, ifseek = %u", tracl + 1, ifseek);
	}
	iread = fread(h_signal_all[tracl], sizeof(float), block_offset, in);
	if(iread != block_offset)
	{
		warn(" iread != ns tracl = %u, iread =%u", tracl + 1, iread);
	}
	h_signal_all_value[tracl] = 1;
}

int ReadDataFromDiskModeTwo(unsigned short *h_signal_all_value, float **h_signal_all, unsigned int num_time_windows_on_RAM, unsigned int block_offset, unsigned int block_number, unsigned int tracl, char** file_name)
{
	/* Read data from file names in list file, Files should be binary */

	unsigned int ifseek;
	unsigned int iread;
	FILE *fin = fopen (file_name[tracl], "rb");
	if(fin == NULL)
	{
		warn(" No such file :%s",file_name[tracl]);
		return -1;
	}
	__off64_t offset = (__off64_t)block_number * block_offset * sizeof(float);
	ifseek = fseeko64(fin, offset, SEEK_SET);
	if(ifseek != 0)
	{
		warn(" fseek failed tracl = %u, ifseek = %u", tracl + 1, ifseek);
	}
	iread = fread(h_signal_all[tracl], sizeof(float), block_offset, fin);
	if(iread != block_offset)
	{
		warn(" iread != ns tracl = %u, iread =%u", tracl + 1, iread);
	}
	fclose (fin);
	h_signal_all_value[tracl] = 1;
	return 0;
}

void ReadDataFromMem(float **h_signal_all, unsigned int num_time_windows_on_RAM, unsigned int tracl, unsigned int block_offset, unsigned int block_number, float* d_signal, cudaStream_t stream)
{
	/* Read the specified data from RAM */
	checkCudaErrors(cudaMemcpyAsync(d_signal, h_signal_all[tracl], block_offset * sizeof(float), cudaMemcpyHostToDevice, stream));
}

