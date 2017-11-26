
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
__global__ void forward_mul_kernel(DType *y, DType *B, DType *A, const int H, const int W, const int K, int offset) {

    //const int H_out = H - K + 1;
    //const int W_out = W - K + 1;
    //const int MHW = 50 * H_out * W_out;
    //const int HoutWout = 576; //H_out * W_out;
    #define y4d(i3,i2,i1,i0) y[(i3) * 28800 + (i2)*576 + (i1)*(24) + i0]
    __shared__ DType tileA[1250];  //[2 * TILE_WIDTH * TILE_HEIGHT];
    __shared__ DType tileB[25 * 32];   //[TILE_WIDTH * TILE_HEIGHT];
    #define tA2d(i1, i0) tileA[i1 * 50 + i0]
    #define tB2d(i1, i0) tileB[i1 * 32 + i0]
    int b = blockIdx.x;  // batch index 
    int bx = blockIdx.y; // 
    int by = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_HEIGHT + ty;
    int Col = bx * 32 + tx;
    int numAColumns = 25, numBColumns = 576;
    DType sum = 0, sum1 = 0;
    //int aBound = 1; //(numAColumns - 1) / TILE_WIDTH + 1; // num of tiles in A columns
    int Aindex, Bindex, tmp;
    //for(int m = 0; m < aBound; ++m){
       //Aindex = Row * numAColumns + m * TILE_WIDTH + tx;//(tx  + m * TILE_WIDTH) * numAColumns + Row;
       //Bindex = b * 14400 + (m * TILE_WIDTH + ty) * numBColumns + Col;
       //if(m * TILE_WIDTH + tx < numAColumns && Row < numARows){
       if(tx < 25){
            Aindex = Row * numAColumns + tx; //m * TILE_WIDTH + tx;
            tA2d(tx, ty) = A[Aindex];
            tA2d(tx, ty + TILE_WIDTH) = A[Aindex + 625];
       }
       //if(Col < numBColumns){
            Bindex = b * 14400 + ty * numBColumns + Col;
            tB2d(ty, tx) = B[Bindex];
       //}else{
       //     tB2d(ty, tx) = 0;
       //}
       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k){
            tmp = tB2d(k, tx);
            sum += tA2d(k, ty) * tmp; //tB2d(k, tx);
            sum1 += tA2d(k, ty + 25) * tmp; //tB2d(k, tx);
       }
       //__syncthreads();
    //}
    //if(Col < 576){
        tmp = (b + offset) * 28800 + Row *576 + Col;
        y[tmp] = sum;
        y[tmp + 14400] = sum1;
        //y[b * 28800 + Row *576 + Col] = sum;
        //y[b * 28800 + (Row +25)*576 + Col] = sum1;
    //}
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
__global__ void unroll_kernel(DType *X_unrolled, DType *W_unrolled, const DType *x, const DType *k, int offset){
    //const int Wout = W - K + 1;
    //const int Hout = H - K + 1;
    int b = blockIdx.x;
    int bid = blockIdx.y;  // id of block
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    DType val = 0;
    #define x4d(i3,i2,i1,i0) x[(i3) * 784 + (i2)*784 + (i1)*28 + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * 25 + (i2)*25 + (i1)*5 + i0]
    #define uw2d(i1, i0) W_unrolled[i1 * 25 + i0]
    if(bid < 2){  // load kernel
        if(b == 0 && local_h < 25 && local_w < 25){
            int m = bid * 25 + local_h;
            int r = local_w / 5, c = local_w % 5;
            uw2d(m, local_w) = k4d(m, 0, r, c);
        }
    }else{ // load input
        val = x4d(b + offset, 0, local_h, local_w);
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0 && local_h + 5 - 1 - i < 28){
                   X_unrolled[b * 14400 + (i * 5 + j)*576 + (local_h - i) * 24 + local_w - j] = val;
                }
            }
        }
    }

    #undef x4d
    #undef k4d
    #undef uw2d
}


// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s1; //= y.stream_->stream_;
    cudaStream_t s2, s3, s4; //
    cudaStreamCreate(&s1);//
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);
    cudaStreamCreate(&s4);
    
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
    DType *X_unrolled1;
    DType *X_unrolled2;
    DType *X_unrolled3;
    DType *X_unrolled4;
    DType *W_unrolled1;
    DType *W_unrolled2;
    cudaMalloc(&X_unrolled1, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&X_unrolled2, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&X_unrolled3, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&X_unrolled4, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&W_unrolled1, M * K * K * sizeof(DType));
    cudaMalloc(&W_unrolled2, M * K * K * sizeof(DType));
    
    // Set the kernel dimensions,    
    dim3 blockDimU(28, 28, 1); // 
    dim3 gridDimU(B/4, 3, 1);    // 
    //dim3 blockDimU2(28, 28, 1);
    //dim3 gridDimU2(B/4, 3, 1);

    //cudaDeviceSynchronize();

    dim3 blockDim(32, TILE_WIDTH, 1);  // 25, 25
    //dim3 gridDim(B, (Hout * Wout - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    dim3 gridDim(B/4, 18, 1);
    dim3 blockDim2(32, TILE_WIDTH, 1);
    dim3 gridDim2(B/4, 18, 1);
    // Call the kernel                                0 is sharemem s is stream
    //for(int i = 0; i < 1; ++i){
    unroll_kernel<gpu,DType><<<gridDimU, blockDimU, 0, s1>>>(X_unrolled1, W_unrolled1, x.dptr_, w.dptr_,0);
    unroll_kernel<gpu,DType><<<gridDimU, blockDimU, 0, s2>>>(X_unrolled2, W_unrolled2, x.dptr_, w.dptr_,2500);
    unroll_kernel<gpu,DType><<<gridDimU, blockDimU, 0, s3>>>(X_unrolled3, W_unrolled1, x.dptr_, w.dptr_,5000);
    unroll_kernel<gpu,DType><<<gridDimU, blockDimU, 0, s4>>>(X_unrolled4, W_unrolled2, x.dptr_, w.dptr_,7500);
    
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s1>>>(y.dptr_, X_unrolled1, W_unrolled1,H,W,K,0);
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s2>>>(y.dptr_, X_unrolled2, W_unrolled2,H,W,K,2500);
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s3>>>(y.dptr_, X_unrolled3, W_unrolled1,H,W,K,5000);
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s4>>>(y.dptr_, X_unrolled4, W_unrolled2,H,W,K,7500);
    //}
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaStreamSynchronize(s3);
    cudaStreamSynchronize(s4);
    //forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_, X_unrolled, W_unrolled, C,H,W,K,12);
    
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    cudaFree(X_unrolled1);
    cudaFree(X_unrolled2);
    cudaFree(W_unrolled1);
    cudaFree(W_unrolled2);
}



}
}

#endif
