static __global__ void f2c(float *a, Complex *b, unsigned int size, unsigned int new_size, unsigned int num_ns_win);
static __global__ void f2c_flip(float *a, Complex *b, unsigned int size, unsigned int new_size, unsigned int num_ns_win);
static __global__ void ComplexPointwiseMulAndScale(Complex *d_shot, Complex *d_signal, Complex *d_out_f, unsigned int num_ns_win, unsigned int new_size);
static __device__ __host__ inline Complex ComplexScale(Complex a, float s);
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b);
static __global__ void Add(Complex *a, Complex *b, unsigned int size, unsigned int num_ns_win);
void ReadDataFromDiskModeOne(unsigned short *h_signal_all_value, float **h_signal_all, unsigned int num_time_windows_on_RAM, unsigned int block_offset, unsigned int block_number, unsigned int tracl, long ns, FILE *in);
int ReadDataFromDiskModeTwo(unsigned short *h_signal_all_value, float **h_signal_all, unsigned int num_time_windows_on_RAM, unsigned int block_offset, unsigned int block_number, unsigned int tracl, char** file_name);
void ReadDataFromMem(float **h_signal_all, unsigned int num_time_windows_on_RAM, unsigned int tracl, unsigned int block_offset, unsigned int block_number, float* d_signal, cudaStream_t stream);
