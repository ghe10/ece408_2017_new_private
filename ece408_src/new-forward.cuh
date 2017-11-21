
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
namespace mxnet
{
namespace op
{


template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    int X_tile_width = TILE_WIDTH + K - 1;
    int b, m, local_h, local_m, h_base, w_base, global_h, global_w;
    extern __shared__ DType shmem[];
    int W_numOfTiles = (W_out - 1) / TILE_WIDTH + 1;
    int H_numOfTiles = (H_out - 1) / TILE_WIDTH + 1;
    DType* X_shared = &shmem[0];
    DType* W_shared = &shmem[X_tile_width * X_tile_width];
    b = blockIdx.x; // batch index 
    m = blockIdx.y; // output feature map index
    local_w = threadIdx.x;
    local_h = threadIdx.y;
    h_base = (blockIdx.z / W_numOfTiles) * TILE_WIDTH;
    w_base = (blockIdx.z % W_numOfTiles) * TILE_WIDTH;
    global_h = h_base + local_h;
    global_w = w_base + local_w;
    
    #define w2d(i1, i0) W_shared[(i1) * K + i0]
    #define x_shared2d(i1, i0)  X_shared[(i1) * X_tile_width + i0]
    DType sum = 0;
    for(int c = 0; c < C; ++c){         // sum over all input channels
        if(local_h < K && local_w < K){
            w2d(local_h, local_w) = k4d(m, c, local_h, local_w);
        }
        __syncthreads();
        
        for(int i = h; i < h_base + X_tile_width; i += TILE_WIDTH){
            for(int j = w; j < w_base + X_tile_width; j += TILE_WIDTH){
                if(i <  H && j < W){
                    x_shared2d(i - h_base, j - w_base) = x4d[b, c, i, j];
                }
            }
        }
        __syncthreads();
        for(int p = 0; p < K; ++p){
            for(int q = 0; q < K; ++q){
                sum += x_shared2d(local_h + p, local_w + q) * w2d(p, q);
            }
        }
        __syncthreads();
    }
    y4d(b, m, global_h, global_w) = sum;
    #undef y4d
    #undef x4d
    #undef k4d
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    CHECK_EQ(0, 1) << "Starting a GPU implementation based on share memory!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = k.shape_[0];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[2];

    // Set the kernel dimensions
    size_t shmem_size = sizeof(DType)*((TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1) + K * K);
    int Hout = H - K + 1;
    int Wout = W - K + 1;
    int W_numOfTiles = (Wout - 1) / TILE_WIDTH + 1;
    int H_numOfTiles = (Hout - 1) / TILE_WIDTH + 1;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, W_numOfTiles * H_numOfTiles);

    // Call the kernel                                0 is sharemem s is stream
    forward_kernel<gpu, DType><<<gridDim, blockDim, shmem_size, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
