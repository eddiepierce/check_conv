#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>

#define CHECK_CUDNN(status) {						\
    if(status != CUDNN_STATUS_SUCCESS) {				\
      printf("CUDNN failure; Error: %s\n", cudnnGetErrorString(status)); \
    }									\
  }
#define CHECK_CUDA(status) {						\
    if(status != 0) {							\
      printf("Cuda failure\nError: %s", cudaGetErrorString(status));	\
    }									\
  }

int main(){
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle));
  cudnnTensorDescriptor_t descI, descO;
  cudnnFilterDescriptor_t descF;
  cudnnConvolutionDescriptor_t descConv;

  // create Descriptors
  CHECK_CUDNN( cudnnCreateTensorDescriptor(&descI) );
  CHECK_CUDNN( cudnnCreateTensorDescriptor(&descO) );
  CHECK_CUDNN( cudnnCreateFilterDescriptor(&descF) );
  CHECK_CUDNN( cudnnCreateConvolutionDescriptor(&descConv) );


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

  float alpha = 1.0f, beta = 0.0f;


  float *d_I, *d_F, *d_O;
  // allocate device memory
  CHECK_CUDA( cudaMalloc((void**) &d_I, size_input  * sizeof(float)) );
  CHECK_CUDA( cudaMalloc((void**) &d_F, size_filter * sizeof(float)) );
  CHECK_CUDA( cudaMalloc((void**) &d_O, size_output * sizeof(float)) );

  int pad[] = {0,0}, str[] = {1,1}, upscale[] = {1,1};

  // set decriptors
  cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
  CHECK_CUDNN( cudnnSetTensor4dDescriptor(descI, CUDNN_TENSOR_NCHW, dataType, N, C, H, W) );
  CHECK_CUDNN( cudnnSetFilter4dDescriptor(descF, dataType,                    C, K, S, R) );
  CHECK_CUDNN( cudnnSetTensor4dDescriptor(descO, CUDNN_TENSOR_NCHW, dataType, N, K, P, Q) );
  CHECK_CUDNN( cudnnSetConvolutionNdDescriptor(descConv, 2, pad, str, upscale, CUDNN_CONVOLUTION, dataType) );

  int outputDim[] = {N, K, P, Q};
  CHECK_CUDNN( cudnnGetConvolutionNdForwardOutputDim(descConv, descI, descF, 4, outputDim) );


  float *h_I = (float*) malloc(size_input  * sizeof(float));
  float *h_F = (float*) malloc(size_filter * sizeof(float));
  float *h_O = (float*) malloc(size_output * sizeof(float));

  for(int i = 0; i < size_input; i++)  h_I[i] = .1 * (i + 1); // fill input with  .1, .2, .3, ...
  for(int i = 0; i < size_filter; i++) h_F[i] = .2 * (i + 1); // fill filter with .2, .4, .6, ...

  // copy host input/filter data to device
  CHECK_CUDA( cudaMemcpy(d_I, h_I, size_input  * sizeof(float), cudaMemcpyHostToDevice) );
  CHECK_CUDA( cudaMemcpy(d_F, h_F, size_filter * sizeof(float), cudaMemcpyHostToDevice) );

  // launch conv
  CHECK_CUDNN( cudnnConvolutionForward(handle, &alpha, descI, d_I, descF, d_F, descConv,
				      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 1<<20, &beta, descO, d_O) );
  // copy device output data to host
  CHECK_CUDA( cudaMemcpy(h_O, d_O, size_output * sizeof(float), cudaMemcpyDeviceToHost) );

  // print results
  for(int i = 0; i < size_output; i++) printf("out[%d] = %.2f\n", i, h_O[i]);
  return 0;
}
