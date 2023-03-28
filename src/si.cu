#include "si.h"

char *sdoc[] = {
" ns=                number of points of input traces ",
" ntr=               number of input traces ",
" dt=                sample interval; in micro-seconds ",
" ns_win=            number of points of time windows in correlation ",
" outns=             number of points in output traces ",
" StartFldr=         start number of original field record number ",
" EndFldr=           end number of original field record number ",
" Stack_flag=        0 Not stack ",
"                    1 Stack causal and noncausal part ",
" Device=            Device number ",
" threads=           number of threads ",
" blocks=            number of blocks ",
" RAMSizeMB=         RAM size (MB) ",
" VRAMSizeMB=        VRAM size (MB)",
" file_mode=         0 A binary file ",
"                    1 A file list of binary files ",
NULL};

/******************************************************************************/
/*Author  : JunZheng                                                          */
/*Mail    : zhengjun@email.cugb.edu.cn                                        */
/*Inst    : CUGB                                                              */
/*Time    : 2023-03-27                                                        */
/*============================================================================*/

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	sipar par;
	unsigned int threads;
	unsigned int blocks;

	/* Initialize */
	initargs(argc, argv);
	requestdoc(0);
	/* Get parameters */
	if (!getparulong("ns", &par.ns))   par.ns = 50001;
	if (!getparuint("ntr", &par.ntr))   par.ntr = 701;
	if (!getparuint("dt", &par.dt)) par.dt = 4000;
	if (!getparuint("ns_win", &par.ns_win)) par.ns_win = par.ns;
	if (!getparuint("outns", &par.outns)) par.outns = 1001;
	if (!getparuint("StartFldr", &par.StartFldr)) par.StartFldr = 1;
	if (!getparuint("EndFldr", &par.EndFldr)) par.EndFldr = 701;
	if (!getparuint("Stack_flag", &par.Stack_flag)) par.Stack_flag = 1;
	if (!getparuint("Device", &par.Device)) par.Device = 0;
	if (!getparuint("threads", &threads)) threads = 512;
	if (!getparuint("blocks", &blocks)) blocks = 512;
	if (!getparuint("RAMSizeMB", &par.RAMSizeMB)) par.RAMSizeMB = 8192;
	if (!getparuint("VRAMSizeMB", &par.VRAMSizeMB)) par.VRAMSizeMB = 4096;
	if (!getparuint("file_mode", &par.file_mode)) par.file_mode = 0;
	
	/* Judge whether to obtain appropriate parameters */
	par.num_ns_win = par.ns / par.ns_win;
	warn("SI is starting ");
	if(!(par.EndFldr >= par.StartFldr))
	{
		warn("SI    <<< cant get right parameters : parameter EndFldr is less than parameter StartFldr >>>");
		return 0;
	}
	
	if(!((par.file_mode == 1)||(par.file_mode == 0)))
	{
		warn("SI    <<< cant get right parameters (file_mode) >>>");
		return 0;
	}

	if(!((par.Stack_flag == 1)||(par.Stack_flag == 0)))
	{
		warn("SI    <<< cant get right parameters (Stack_flag) >>>");
		return 0;
	}

	if(par.Stack_flag == 1)
	{
		if(par.outns > par.ns_win)
		{
			warn("SI    <<< cant get right parameters : parameter outns should be less than ns_win >>>");
			return 0;
		}
	}
	else
	{
		if(par.outns > 2 * par.ns_win - 1)
		{
			warn("SI    <<< cant get right parameters : parameter outns should be less than 2 * ns_win - 1 >>>");
			return 0;
		}
	}

	if(par.ntr <= 1)
	{
		warn("SI    <<< cant get right parameters : parameter ntr should be great than 1 >>>");
		return 0;
	}

	if(par.ns <= 1)
	{
		warn("SI    <<< cant get right parameters : parameter ns should be great than 1 >>>");
		return 0;
	}
	if(par.ns < par.ns_win)
	{
		warn("SI    <<< cant get right parameters : parameter ns should be great than ns_win >>>");
	}

	/* Print parameters */
	warn("SI    <<< ns = %u >>>", par.ns);
	warn("SI    <<< ntr = %u >>>", par.ntr);
	warn("SI    <<< dt = %u >>>", par.dt);
	warn("SI    <<< ns_win = %u >>>", par.ns_win);
	warn("SI    <<< outns = %u >>>", par.outns);
	warn("SI    <<< StartFldr = %u >>>", par.StartFldr);
	warn("SI    <<< EndFldr = %u >>>", par.EndFldr);
	warn("SI    <<< Stack_flag = %u >>>", par.Stack_flag);
	warn("SI    <<< threads = %u >>>", threads);
	warn("SI    <<< blocks = %u >>>", blocks);
	warn("SI    <<< RAMSizeMB = %u >>>", par.RAMSizeMB);
	warn("SI    <<< VRAMSizeMB = %u >>>", par.VRAMSizeMB);
	warn("SI    <<< file_mode = %u >>>", par.file_mode);
	warn("SI    <<< number of windows in a tracl = %u >>>", par.num_ns_win);

	/* Set device and record time */
	cudaSetDevice((int)par.Device);
	StopWatchInterface *timer;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	par.threads = dim3((int)threads, 1);
	par.blocks = dim3((int)blocks, 1);

	/* Get GPU information */
	cudaDeviceProp props;
	unsigned int DeviceMem;
	checkCudaErrors(cudaGetDeviceProperties(&props, par.Device));
	DeviceMem = props.totalGlobalMem / 1024 / 1024;
	warn("SI    <<< Device %d: \"%s\" with Compute %d.%d capability >>>", par.Device, props.name, props.major, props.minor);
	warn("SI    <<< Device memory = %u MB >>>", DeviceMem);
	if(par.VRAMSizeMB > DeviceMem)
	{
		warn("SI    <<< cant get right parameters (VRAMSizeMBGpu) >>>");
		return 0;
	}

	/* Cross correlation function */
	siCUFFT(par, stdin);
	
	/* Record time and print */
	sdkStopTimer(&timer);
	warn("Processing time: %f (s)", sdkGetTimerValue(&timer) / 1000);
	sdkDeleteTimer(&timer);
	return 0;
}
