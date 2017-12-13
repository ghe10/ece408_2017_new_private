
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
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    extern __shared__ __align__(sizeof(DType)) unsigned char my_smem[];
    DType *shmem = reinterpret_cast<DType *>(my_smem);
    DType* x_shared = &shmem[0];

    int block_num = blockIdx.x;
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    int input_size = 28 * 28;
    int output_size = 24 * 24;
    int read_base = block_num * PIC_PER_BLOCK * 28 * 28 + local_h * 28 + local_w;
    int write_base = 0;
    DType sum = 0.0;
    // load 28 * 28 input
    for (int index = 0; index < PIC_PER_BLOCK; index++) {
      x_shared[index * input_size + local_h * 28 + local_w] = x[read_base];
      read_base += input_size; // each block load adjacent 10 images
    }
    __syncthreads();

    // compute

    if (local_h < 24 && local_w < 24) {
      write_base = block_num * PIC_PER_BLOCK * 50 * output_size + local_h * 24 + local_w;
      for (int index = 0; index < PIC_PER_BLOCK; index++) {
        #pragma unroll 1
        for (int kernel_index = 0; kernel_index < 50; kernel_index++) {
          sum = 0.0;
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                sum += x_shared[index * input_size + (local_h + i) * 28 + local_w + j] *
                  constant_kernel[kernel_index * 25 + i * 5 + j];
              }
          }

          // write_base = (block_num) * PIC_PER_BLOCK * 50 * output_size +
          //   index * (B / PIC_PER_BLOCK) * 50 * output_size
          //   + kernel_index * output_size + local_h * 24 + local_w;
          y[write_base + kernel_index * output_size] = sum; // this should be kernel related
        }
        write_base += 50 * output_size;
      }
    }
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {


    // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) << "Starting a GPU implementation based on share memory!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; // input batch
    const int M = w.shape_[0]; // output channel number
    const int C = x.shape_[1]; // input channel number
    const int H = x.shape_[2]; // input height
    const int W = x.shape_[3]; // input width
    const int K = w.shape_[2]; // kernel size

    // Set the kernel dimensions
    // int Hout = H - K + 1;
    // int Wout = W - K + 1;
    // int W_numOfTiles = (Wout - 1) / TILE_WIDTH + 1;
    // int H_numOfTiles = (Hout - 1) / TILE_WIDTH + 1;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((B / PIC_PER_BLOCK), 1, 1);

    // allocate constant_kernel
    cudaMemcpyToSymbol(constant_kernel, w.dptr_, sizeof(float) * 50 * 25, 0, cudaMemcpyDeviceToDevice);
    // Call the kernel                                0 is sharemem s is stream
    forward_kernel<gpu, DType><<<gridDim, blockDim, PIC_PER_BLOCK * sizeof(DType) * H * W, s>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
