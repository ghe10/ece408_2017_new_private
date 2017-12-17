
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 28
#define PIC_PER_BLOCK 10

namespace mxnet
{
namespace op
{

__constant__ float kernel[1250];

__global__ void forward_kernel(float *y, const float *x, const float *k) {
    /*
      A few constants are used in this kernel to reduce computation
    */
    __shared__ float x_shared[7840];
    int block_num = blockIdx.x << 1;
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    int local_index = (local_h << 2)*7 + local_w;
    int read_base = (block_num << 5)*245 + local_index;
    int write_base;
    float sum;

    // load 28 * 28 input
    for (int index = 0; index < PIC_PER_BLOCK; index++) {
      x_shared[(index<<4)*49 + local_index] = x[read_base + (index<<4)*49];
    }
    __syncthreads();

    // compute the convolution result

    if (local_index < 576) {
      write_base = (block_num << 8) * 1125 + local_index;
      for (int index = 0; index < PIC_PER_BLOCK; index++) {
        #pragma unroll 14
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[(index<<4)*49 + (local_index/24 + i) * 28 + local_index%24 + j] *
                  kernel[kernel_index * 25 + i * 5 + j];
              }
          }
          y[write_base + (index << 7)*225 + (kernel_index << 6)*9] = sum;
        }
      }
    }
    __syncthreads();

    // ------------------- second batch --------------------------
    read_base += 7840;
    write_base += 288000;
    for (int index = 0; index < PIC_PER_BLOCK; index++) {
      x_shared[(index<<4)*49 + local_index] = x[read_base + (index<<4)*49];
    }
    __syncthreads();

    // compute the convolution result

    if (local_index < 576) {
      for (int index = 0; index < PIC_PER_BLOCK; index++) {
        #pragma unroll 14
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[(index<<4)*49 + (local_index/24 + i) * 28 + local_index%24 + j] *
                  kernel[kernel_index * 25 + i * 5 + j];
              }
          }
          y[write_base + (index<<7)*225 + (kernel_index<<6)*9] = sum;
        }
      }
    }
    //__syncthreads();
    // -----------------  third batch ---------------------
/*    read_base += 7840;
    write_base += 288000;
    for (int index = 0; index < PIC_PER_BLOCK; index++) {
      x_shared[index * 784 + local_index] = x[read_base + index * 784];
    }
    __syncthreads();

    // compute the convolution result

    if (local_index < 576) {
      for (int index = 0; index < PIC_PER_BLOCK; index++) {
        #pragma unroll 15
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[index*784 + (local_index/24 + i) * 28 + local_index%24 + j] *
                  kernel[kernel_index * 25 + i * 5 + j];
              }
          }
          y[write_base + index * 28800 + kernel_index * 576] = sum;
        }
      }
    }
    __syncthreads();
    // -----------------  fourth batch -----------------------
    read_base += 7840;
    write_base += 288000;
    for (int index = 0; index < PIC_PER_BLOCK; index++) {
      x_shared[index * 784 + local_index] = x[read_base + index * 784];
    }
    __syncthreads();

    // compute the convolution result

    if (local_index < 576) {
      for (int index = 0; index < PIC_PER_BLOCK; index++) {
        #pragma unroll 15
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[index*784 + (local_index/24 + i) * 28 + local_index%24 + j] *
                  kernel[kernel_index * 25 + i * 5 + j];
              }
          }
          y[write_base + index * 28800 + kernel_index * 576] = sum;
        }
      }
    }
*/
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    const int B = x.shape_[0] / 20; // input batch

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, 1, 1);

    // allocate constant_kernel
    cudaMemcpyToSymbol(kernel, w.dptr_, 5000, 0, cudaMemcpyDeviceToDevice);
    // Call the kernel                                0 is sharemem s is stream
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_);
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
