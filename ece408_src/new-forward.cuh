
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
//#include <mxnet/operator.h>
#include "mshadow/tensor.h"
#define TILE_WIDTH 28
#define TILE_HEIGHT 29
namespace mxnet
{
namespace op
{


template<typename gpu, typename DType>
__global__ void forward_mul_kernel(DType *y, DType *B, DType *A, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int MHW = 50 * H_out * W_out;
    const int HoutWout = 24*24; //H_out * W_out;
    #define y4d(i3,i2,i1,i0) y[(i3) * MHW + (i2)*HoutWout + (i1)*(W_out) + i0]
    __shared__ DType tileA[25 * 25];
    __shared__ DType tileB[25 * 25];
    #define tA2d(i1, i0) tileA[i1 * 25 + i0]
    #define tB2d(i1, i0) tileB[i1 * 25 + i0]
    int b = blockIdx.x; // batch index 
    int bx = blockIdx.y; // 
    int by = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * 25 + ty;
    int Col = bx * 25 + tx;
    int numARows = 50, numBRows = 25, numAColumns = 25, numBColumns = 24 * 24;
    DType sum = 0;
    int aBound = (numAColumns - 1) / 25 + 1; // num of A columns
    int Aindex, Bindex;
    for(int m = 0; m < aBound; ++m){
       Aindex = Row * numAColumns + m * 25 + tx;
       Bindex = b * 25 * 24 * 24 + (m * 25 + ty) * numBColumns + Col;
       if(m * 25 + tx < numAColumns && Row < numARows){
            tA2d(ty, tx) = A[Aindex];
       }else{
            tA2d(ty, tx) = 0;
       }
       if(m * 25 + ty < numBRows && Col < numBColumns){
            tB2d(ty, tx) = B[Bindex];
       }else{
            tB2d(ty, tx) = 0;
       }
       __syncthreads();
       for (int k = 0; k < 25; ++k){
            sum += tA2d(ty, k) * tB2d(k, tx);
       }
       __syncthreads();
    }
    if(Row < 50 && Col < H_out * W_out){
        int r = Col / W_out, c = Col % W_out;
        y4d(b, Row, r, c) = sum;
    }
    #undef y4d
    #undef tA2d
    #undef tB2d
}

template<typename gpu, typename DType>
__global__ void matrix_kernel(DType *y, DType *B, DType *A){
    int row = blockIdx.z * 25 + threadIdx.y;
    int col =  blockIdx.y * 25 + threadIdx.x;
    int b = blockIdx.x;

    if(row < 50 && col < 24 * 24){
        DType val = 0;
        for(int i = 0; i < 25; ++i){
            val += A[row * 25 + i] * B[b* 25 * 24 * 24 + i * 24*24 + col];
        }
        int r = col / 24, c = col % 24;
        y[b * 50 * 24 * 24 + row * 24 * 24 + r*24 + c] = val;
    }
}

template<typename gpu, typename DType>
__global__ void unroll_kernel(DType *X_unrolled, DType *W_unrolled, const DType *x, const DType *k, const int H, const int W, const int K){
    const int Wout = W - K + 1;
    const int Hout = H - K + 1;
    int b = blockIdx.x;
    int bid = blockIdx.y;  // id of block
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    DType val = 1;
    #define x4d(i3,i2,i1,i0) x[(i3) * (H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    #define uw2d(i1, i0) W_unrolled[i1 * 25 + i0]
    //#define ux2d(i1, i0) X_unrolled[ + i1 * Wout * Hout + i0]
    if(bid < 2){  // load kernel
        if(b == 0 && local_h < 25 && local_w < 25){
            int m = bid * 25 + local_h;
            int r = local_w / 5, c = local_w % 5;
            uw2d(m, local_w) = k4d(m, 0, r, c);
        }
    }else{ // load input
        val = x4d(b, 0, local_h, local_w);
        for(int i = 0; i < K; ++i){
            for(int j = 0; j < K; ++j){
                if(local_w - j >= 0 && local_w + K - 1 - j < W && local_h - i >= 0 && local_h + K - 1 - i < H){
                   X_unrolled[b * 25 * 24 * 24 + (i * K + j)*24*24 + (local_h - i) * Wout + local_w - j] = val;
                }
            }
        }
    }

    #undef x4d
    #undef k4d
    #undef uw2d
    //#undef ux2d
}


// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    //mshadow::InitTensorEngine<gpu>();
    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;
    //cudaStream_t sx = x.stream_->stream_;
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = w.shape_[0];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[2];
    printf("The filter size is K * K, where K = %d \n", K);
    printf("B = %d, M = %d, C = %d, H = %d, W = %d \n", B, M, C, H, W);
    int Hout = H - K + 1;
    int Wout = W - K + 1;     
    //mshadow::Tensor<gpu, 1, DType> X_unrolled(mshadow::Shape1(B * K * K * Hout * Wout));
    //mshadow::AllocSpace(&X_unrolled);
    //mshadow::Tensor<gpu, 1, DType> W_unrolled(mshadow::Shape1(M * K * K));
    //mshadow::AllocSpace(&W_unrolled);
DType *X_unrolled;
DType *W_unrolled;
    cudaMalloc(&X_unrolled, B * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&W_unrolled, M * K * K * sizeof(DType));
    //
    // Set the kernel dimensions,    
    dim3 blockDim1(28, 28, 1); // 
    dim3 gridDim1(B, 3, 1);    // 
    //unroll_kernel<gpu, DType><<<gridDim1, blockDim1>>>(X_unrolled.dptr_, W_unrolled.dptr_, x.dptr_, w.dptr_, H,W,K);
    unroll_kernel<gpu, DType><<<gridDim1, blockDim1>>>(X_unrolled, W_unrolled, x.dptr_, w.dptr_, H,W,K);

cudaDeviceSynchronize();

    dim3 blockDim(25, 25, 1);  // 25, 25
    dim3 gridDim(B, (Hout * Wout - 1) / 25 + 1, (M - 1) / 25 + 1);
    //dim3 gridDim(B, 24, 2);
    // Call the kernel                                0 is sharemem s is stream
    //matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s>>>(y.dptr_, X_unrolled.dptr_, W_unrolled.dptr_);
matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s>>>(y.dptr_, X_unrolled, W_unrolled);
    //forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,X_unrolled, W_unrolled, C,H,W,K);
//forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,X_unrolled, W_unrolled, C,H,W,K);

    //cudaDeviceSynchronize();
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    cudaFree(X_unrolled);
    cudaFree(W_unrolled);
    //mshadow::FreeSpace(&X_unrolled);
    //mshadow::FreeSpace(&W_unrolled);
}



}
}

#endif
