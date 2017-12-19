
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float kernel[1250];

__global__ void forward_kernel(float *y, const float *x, const float *k, int xBound, int yBound) {
    /*
      A few constants are used in this kernel to reduce computation
    */
    __shared__ float x_shared[10192];
    int block_num = blockIdx.x;
    //int local_h = threadIdx.y;
    //int local_w = threadIdx.x;
    int local_index = (threadIdx.y << 6) + threadIdx.x; //(local_h << 6) + local_w;
    int read_base = (block_num << 4) * 1225 + local_index;   //784 * 25
    int write_base = (block_num << 7) * 5625 + local_index;//28800 * 25
    int x_pos, y_pos;
    float sum;
    // load 28 * 28 input
    for (int index = 0; index < 17; index++) {
      x_pos = read_base + (index<<6)*9;
      if(x_pos < xBound){
        x_shared[(index<<6)*9 + local_index] = x[x_pos];
      }
    }
    if(local_index < 400 && read_base + 9792 < xBound){
        x_shared[9792 + local_index] = x[read_base + 9792];
    }
    __syncthreads();

    // compute the convolution result
      for (int index = 0; index < 13; index++) {
        x_pos = (index<<4)*49 + (((local_index >> 3)/3) << 2) * 7 + local_index%24;
        y_pos = write_base + (index << 7)*225;
        if(y_pos < yBound){
        #pragma unroll 15
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[x_pos + (i << 2)*7  + j] *
                  kernel[kernel_index * 25 + i * 5 + j];
              }
          }
          y[y_pos + (kernel_index << 6) * 9] = sum;
        }
        }
      }
    __syncthreads();

    // ------------------- second batch --------------------------
    read_base += 10192;   //7840;
    write_base += 374400; //288000;
    for (int index = 0; index < 16; index++) {
      x_pos = read_base + (index<<6)*9;
      if(x_pos < xBound){
        x_shared[(index<<6)*9 + local_index] = x[x_pos];
      }
    }
    if(local_index < 192 && read_base + 9214 < xBound) x_shared[9216 + local_index] = x[read_base + 9216];
    __syncthreads();

    // compute the convolution result
      for (int index = 0; index < 12; index++) {
        x_pos = (index<<4)*49 + (((local_index>>3)/3) << 2) * 7 + local_index%24;
        y_pos = write_base + (index << 7)*225;
        if(y_pos < yBound){
        #pragma unroll 15
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[x_pos + (i<<2)*7 + j] *
                  kernel[kernel_index * 25 + i * 5 + j];
              }
          }
          y[y_pos + (kernel_index<<6) * 9] = sum;
        }
        }
      }
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    const int B = x.shape_[0]; // input batch

    dim3 blockDim(64, 9, 1);
    dim3 gridDim((B - 1) / 25 + 1, 1, 1);

    // allocate constant_kernel
    cudaMemcpyToSymbol(kernel, w.dptr_, 5000, 0, cudaMemcpyDeviceToDevice);
    // Call the kernel                                0 is sharemem s is stream
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, (B<<4)*49, (B<<6)*9);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}
    

}
}

#endif
