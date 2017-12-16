
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 28
#define PIC_PER_BLOCK 10

namespace mxnet
{
namespace op
{

__constant__ float constant_kernel[1250];

template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    A few constants are used in this kernel to reduce computation
    */
    extern __shared__ __align__(sizeof(DType)) unsigned char my_smem[];
    DType *shmem = reinterpret_cast<DType *>(my_smem);
    DType* x_shared = &shmem[0];

    int block_num = blockIdx.x;
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    int input_size = 784;
    int output_size = 576;
    int read_base = block_num * 7840 + local_h * 28 + local_w;
    int write_base = 0;
    int index_base = 0;
    DType sum = 0.0;
    // load 28 * 28 input
    for (int index = 0; index < PIC_PER_BLOCK; index++) {
      x_shared[index * input_size + local_h * 28 + local_w] = x[read_base];
      read_base += input_size; // each block load adjacent 10 images
    }
    __syncthreads();

    // compute the convolution result

    if (local_h < 24 && local_w < 24) {
      write_base = block_num * 288000 + local_h * 24 + local_w;
      for (int index = 0; index < PIC_PER_BLOCK; index++) {
        index_base = index * input_size;
        #pragma unroll 15
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0.0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[index_base + (local_h + i) * 28 + local_w + j] *
                  constant_kernel[kernel_index * 25 + i * 5 + j];
              }
          }
          y[write_base + kernel_index * output_size] = sum;
        }
        write_base += 28800;
      }
    }
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    const int B = x.shape_[0]; // input batch

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((B / PIC_PER_BLOCK), 1, 1);

    // allocate constant_kernel
    cudaMemcpyToSymbol(constant_kernel, w.dptr_, sizeof(float) * 50 * 25, 0, cudaMemcpyDeviceToDevice);
    // Call the kernel                                0 is sharemem s is stream
    forward_kernel<gpu, DType><<<gridDim, blockDim, 7840 * sizeof(DType)>>>(y.dptr_,x.dptr_,w.dptr_, B, 50, 1, 28, 28, 5);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
