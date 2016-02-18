#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>

#define CHECK_CUDA( fn ) {						\
    CUresult status = (fn);						\
    if ( CUDA_SUCCESS != status ) {					\
      const char* errstr;						\
      cuGetErrorString(status, &errstr);				\
      printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
      if (hContext) cuCtxDestroy(hContext);				\
      exit(EXIT_FAILURE);						\
    }									\
  }


// select which cubin to use

//#define CONV_NAME "sconv_fprop_K128_N128"
//#define CONV_NAME "sconv_fprop_K128_N64"
//#define CONV_NAME "sconv_fprop_K32_N128"
//#define CONV_NAME "sconv_fprop_K64_N128"
#define CONV_NAME "sconv_fprop_K64_N64"

CUcontext hContext = 0;

int main(){
  CHECK_CUDA( cuInit(0) );
  CHECK_CUDA( cuCtxCreate(&hContext, 0, 0) );
  CUmodule hModule;
  CUfunction hKernel;
  CUdeviceptr d_Sum, d_O, d_I, d_F;

  int N = 1; // n image
  int C = 1; // input feature map
  int K = 1; // output feature map

  int H = 5; // input height
  int W = 1; // input width

  int S = 1; // filter height
  int R = 1; // filter width

  int P = 5; // output height
  int Q = 1; // output width

  int size_input  = N * C * H * W;
  int size_filter = C * K * S * R;
  int size_output = N * K * P * Q;

  int D = 1, T = 1, M = 1;
  float alpha = 1.0f, beta = 0.0f;
  int flags = 0, offset_K = 0;

  int WN = W*N;
  int HWN = H*WN;
  int DHWN = D*HWN;
  int RS = R*S;
  int RST = RS*T;
  int CRST = C*RST;
  int PQ = P*Q;
  int QN = Q*N;
  int PQN = P*QN;
  int MPQN = M*PQN;

  int magic_RS = 1, magic_S = 1, magic_Q = 1, magic_PQ = 1;
  int shift_RS = 0, shift_S = 0, shift_Q = 0, shift_PQ = 0;
  int pad_d = 0, pad_h = 0, pad_w = 0;
  int str_d = 1, str_h = 1, str_w = 1;

  float *h_I = (float*) malloc(size_input  * sizeof(float));
  float *h_F = (float*) malloc(size_filter * sizeof(float));
  float *h_O = (float*) malloc(size_output * sizeof(float));

  for(int i = 0; i < size_input; i++)   h_I[i] = 0.1 * (i + 1);  // fill input with  .1, .2, .3, ...
  for(int i = 0; i < size_filter; i++)  h_F[i] = 0.2 * (i + 1);  // fill filter with .2, .4, .6, ...
  memset(h_O, 0.0f, size_output * sizeof(float));                // fill output with 0

  // allocate device memory
  CHECK_CUDA( cuMemAlloc(&d_I, size_input  * sizeof(float)) );
  CHECK_CUDA( cuMemAlloc(&d_F, size_filter * sizeof(float)) );
  CHECK_CUDA( cuMemAlloc(&d_O, size_output * sizeof(float)) );
  
  // copy host input/filter data to device; set device output to 0
  CHECK_CUDA( cuMemcpyHtoD(d_I, h_I, size_input  * sizeof(float)) );
  CHECK_CUDA( cuMemcpyHtoD(d_F, h_F, size_filter * sizeof(float)) );
  CHECK_CUDA( cuMemcpyHtoD(d_O, h_O, size_output * sizeof(float)) );

  // load cubin and kernel
  CHECK_CUDA( cuModuleLoad(&hModule, CONV_NAME ".cubin") );
  CHECK_CUDA( cuModuleGetFunction(&hKernel, hModule, CONV_NAME) );


  void* params[] ={&d_Sum, &d_O, &d_I, &d_F,
		   &alpha, &beta, &flags, &offset_K,
		   &N, &K, &D, &H,
		   &W, &WN, &HWN, &DHWN,
		   &C, &CRST, &RST, &RS, &magic_RS, &shift_RS,
  		   &S,&magic_S,&shift_S,
		   &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
		   &Q, &PQ, &QN, &PQN, &MPQN,
		   &magic_Q, &shift_Q, &magic_PQ, &shift_PQ };

  int block_x  = 1;
  int block_y  = (N-1) / 64 + 1;
  int block_z  = 1;
  int thread_x = 1;
  int thread_y = 1;
  int thread_z = 1;
  int shared_mem = 2056;

  // launch conv
  CHECK_CUDA( cuLaunchKernel(hKernel,
			     block_x, block_y, block_z,
			     thread_x, thread_y, thread_z,
			     shared_mem, NULL, params, 0) );


  // copy device output data to host
  CHECK_CUDA( cuMemcpyDtoH(h_O, d_O, size_output * sizeof(float)) );
  
  // print results
  for(int i = 0; i < size_output; i++) printf("out[%d] = %.2f\n", i, h_O[i]);
  return 0;
}
