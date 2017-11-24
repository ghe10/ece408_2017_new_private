
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 25
#define TILE_HEIGHT 25
namespace mxnet
{
namespace op
{

  template<typename gpu, typename DType>
  __global__ void forward_kernel(DType *y, const DType *B, const DType *A,  const int C, const int H, const int W, const int K) {

      const int H_out = H - K + 1;
      const int W_out = W - K + 1;
      const int MHW = 50 * H_out * W_out;
      const int HoutWout = H_out * W_out;
      #define y4d(i3,i2,i1,i0) y[(i3) * MHW + (i2)*HoutWout + (i1)*(W_out) + i0]
      __shared__ DType tileA[TILE_WIDTH * TILE_HEIGHT];
      __shared__ DType tileB[TILE_WIDTH * TILE_HEIGHT];
      #define tA2d(i1, i0) tileA[i1 * TILE_WIDTH + i0]
      #define tB2d(i1, i0) tileB[i1 * TILE_WIDTH + i0]
      int b = blockIdx.x; // batch index
      int bx = blockIdx.y; //
      int by = blockIdx.z;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;
      int numARows = 50, numBRows = 25, numAColumns = K * K, numBColumns = H_out * W_out;
      DType sum = 0;
      int aBound = (numAColumns - 1) / TILE_WIDTH + 1; // num of A columns
      int Aindex, Bindex;
      for(int m = 0; m < aBound; ++m){
         Aindex = Row * numAColumns + m * TILE_WIDTH + tx;
         Bindex = b * 25 * 24 * 24 + (m * TILE_WIDTH + ty) * numBColumns + Col;
         if(m * TILE_WIDTH + tx < numAColumns && Row < numARows){
              tA2d(ty, tx) = A[Aindex];
         }else{
              tA2d(ty, tx) = 0;
         }
         if(m * TILE_WIDTH + ty < numBRows && Col < numBColumns){
              tB2d(ty, tx) = B[Bindex];
         }else{
              tB2d(ty, tx) = 0;
         }
         __syncthreads();
         for (int k = 0; k < TILE_WIDTH; ++k){
              sum += tA2d(ty, k) * tB2d(k, tx);
         }
         __syncthreads();
      }
      if(Row < 50 && Col < H_out * W_out){
            //C[Row * numCColumns + Col] = Cvalue;
          y4d(b, Row, Col / W_out, Col % W_out) = sum;
      }
      #undef y4d
      #undef tA2d
      #undef tB2d
  }

template<typename gpu, typename DType>
__global__ void matrix_kernel(DType *y, DType *B, DType *A) {
  int row = blockIdx.z * 25 + blockIdx.y;
  int col = blockIdx.y * 25 + blockIdx.x;
  int b = blockIdx.x;

  if (row < 50 && col < 24 * 24) {
    DType val = 0;
    for (int i = 0; i < 25; i++) {
      val = val +  A[row * 25 + i] * B[b * 25 * 24 *24 + i * 24 * 24 + col];
    }
    int r = col / 24;
    int c = col % 24;
    y[b * 50 * 24 * 24 + row * 24 * 24 + r * 24 + c] = val;
  }
}

template<typename gpu, typename DType>
__global__ void unroll_kernel(DType *k, DType *unroll_kernel,
  int output_channels, int kernel_y, int kernel_x, int input_channels) {
  int bid = blockIdx.x;
  int thx = threadIdx.x;

  //int k_unrolled_size = output_channels * kernel_y * kernel_x * input_channels;
  int row_size = kernel_y * kernel_x * input_channels;
  int stride = kernel_y * kernel_x;
  int kernel_size = kernel_y * kernel_x;
  // one block per kernel
  int base = bid * row_size;
  DType val = k[bid * kernel_size + thx];

  for (int i = thx; i < row_size; i += stride) {
    unroll_kernel[base + i] = val;
  }
}

template<typename gpu, typename DType>
__global__ void unroll_data(DType *x, DType *x_unrolled, int batch_size,
  int input_channels, int input_y_size, int input_x_size, int output_y_size,
  int output_x_size, int kernel_y, int kernel_x) {
    __shared__ DType channel_for_this_block[28 * 28];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // we first compute which input channel we are working on in this batch_size
    int channel_index = bx * input_channels + by;
    int width_base = channel_index * kernel_y * kernel_x; // start y pos in chap 16
    int data_base = channel_index * input_x_size * input_y_size;
    // load data into shared memory
    channel_for_this_block[ty * input_x_size + tx] = x[data_base + ty * input_x_size + tx];
    __syncthreads();
    // mapping
    if (tx < output_x_size && ty < output_y_size) {
      for (int p = 0; p < kernel_y; p++) {
        for (int q = 0; q < kernel_x; q++) {
          int w_unroll = width_base + p * kernel_x + q;
          int h_unroll = ty * output_x_size + tx;
          x_unrolled[h_unroll * output_y_size * output_x_size + w_unroll] =
            channel_for_this_block[(ty + p) * input_x_size + tx + q];
        }
      }
    }
}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &k) {


    // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) << "Starting a GPU implementation based on share memory!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    const int batch_size = x.shape_[0];
    const int input_channels = x.shape_[1]; // c
    const int input_y_size = x.shape_[2]; // h
    const int input_x_size = x.shape_[3]; // w
    const int output_channels = k.shape_[0]; // kernel number
    const int kernel_y = k.shape_[2];
    const int kernel_x = k.shape_[3];
    const int output_y_size = y.shape_[2];
    const int output_x_size = y.shape_[3];

    DType* x_unrolled;
    DType* k_unrolled;

    // unroll k, each block is incharge of a kernel
    int k_unrolled_size = output_channels * kernel_y * kernel_x * input_channels;
    cudaMalloc(&k_unrolled, sizeof(DType) * k_unrolled_size);
    dim3 k_unroll_block_dim(25, 1, 1);
    dim3 k_unroll_grid_dim(output_channels, 1, 1);
    unroll_kernel<gpu, DType><<<k_unroll_grid_dim, k_unroll_block_dim>>>(k.dptr_,
      k_unrolled, output_channels, kernel_y, kernel_x, input_channels);

    // unroll x
    int x_unrolled_size = kernel_y * kernel_x * input_channels * output_y_size
      * output_x_size;
    cudaMalloc(&x_unrolled, sizeof(DType) * x_unrolled_size);
    dim3 x_unroll_block_dim(input_x_size, input_y_size, 1);
    dim3 x_unroll_grid_dim(batch_size, input_channels, 1);
    unroll_data<gpu, DType><<<x_unroll_grid_dim, x_unroll_block_dim>>>(x.dptr_,
      x_unrolled, batch_size, input_channels, input_y_size, input_x_size,
      output_y_size, output_x_size, kernel_y, kernel_x);

      // multi
      dim3 blockDim(25, 25, 1);
      dim3 gridDim(batch_size, 24, 2);
      //forward_kernel<gpu, DType><<<gridDim, blockDim, 2*25*25, s>>>(y.dptr_, x_unrolled, k_unrolled, input_channels, input_y_size, input_x_size, kernel_y);
    matrix_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_, x_unrolled, k_unrolled);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}



}
}

#endif
