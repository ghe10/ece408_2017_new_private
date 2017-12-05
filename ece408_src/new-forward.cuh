
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 25
#define TILE_HEIGHT 25
namespace mxnet
{
namespace op
{

__global__ void matrix_kernel(float *y, float *x, float *k, int offset){ //gridDim B/4, 3, 1,  blockDim 25 * 32

    __shared__ float tileA[1250];    // 50 * 25
    __shared__ float tileB[7200];    // tile width can be selected from 32, 16, 8, 6, 3, 1
    int b = blockIdx.x;   // batch index
    int bx = blockIdx.y;  // 3
    int tx = threadIdx.x; // 8 (mutable)
    int ty = threadIdx.y; // 25
    int index = ty * 32 + tx;  // tile width = 8 (mutable)
    #define tA2d(i1, i0) tileA[i1 * 50 + i0]
    //------------------------ load filters from global to shared memory -----------------------------        
    if(ty >= 14){
            /*int kIndex = ty * 25 + tx, tAIndex = tx * 50 + ty; //tAIndex2 = (tx);
            tileA[tAIndex] = k[kIndex];
            tileA[tAIndex + TILE_WIDTH] = k[kIndex + 625];*/
            int kIndex = (ty - 14) * 32 + tx;
            int tAIndex = kIndex / 25 + kIndex % 25 * 50;
            tileA[tAIndex] = k[kIndex];
            tileA[tAIndex + TILE_WIDTH] = k[kIndex + 625];
            kIndex += 352;
            if(kIndex < 625){
                tAIndex = kIndex / 25 + kIndex % 25 * 50;
                tileA[tAIndex] = k[kIndex];
                tileA[tAIndex + TILE_WIDTH] = k[kIndex + 625];
            }
    }else{
    //------------------------ load data from global to shared memory ----------------------------
    float val; //[12] = {0};   // 24 * 24 / (3 * tile_width)
    int topR, topC;//tileTmpIndex;
    int local_w, local_h, xIndex;// tileBOffsetFactor;
    //if(index < 448){                                                // 448 = (12 + 4) * 28
            local_w = index % 28;// + blockOffset; //8 * bx;
            local_h = index / 28 + bx * 12;
            xIndex = (b + offset) * 784 + local_h * 28 + local_w;
            val = x[xIndex];
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                    topC = local_w - j;//(local_h - i) * 24 + local_w - j;
                    topR = local_h - i;
                    if(bx*12 <= topR && topR < 12 + bx* 12  && 0 <= topC && topC < 24){
                            tileB[(i * 5 + j) * 288 + (topR - 12*bx) * 24 + topC] = val;
                    }
            }
        }
    //}
    }
    __syncthreads();
    
    //----------------------------------- compute ------------------------------------
    float sum[18] = {0};
    float tmpA1, tmpA2, tmpB;
    int tileBx;
    for(int i = 0; i < 9; ++i){
    for(int k = 0; k < TILE_WIDTH; ++k){
        tmpA1 = tA2d(k, ty); tmpA2 = tA2d(k, ty + 25);
        tileBx = i * 32 + tx;
            tmpB = tileB[tileBx + k * 288]; // 25 * 8
            sum[i * 2] += tmpA1 * tmpB;
            sum[i * 2 + 1] += tmpA2 * tmpB;
    }
    }
    //__syncthreads();

    //---------------------------- write to global memory ----------------------------
        int col = bx * 288 + tx; // 288 = 12 *24 ,   row = ty;
        int tmp = (b + offset) * 28800 + ty *576 + col;
        
        for(int i = 0; i < 9; ++i){
            y[tmp + 32 * i] = sum[i << 1]; // 3 * 8
        }
        tmp += 14400;   // 14400 = 25 * (24 * 24)
        for(int i = 0; i < 9; ++i){
            y[tmp + 32 * i] = sum[i*2 + 1]; // 3 * 8
        }
    
        /*
        for(int i = 0; i < 48; i += 2){
            y[tmp + 48 * (i >> 1)] = sum[i];//y[tmp + 3 * 32 * (i >> 1)] = sum[i];
            y[tmp + 14400 + 48 * (i >> 1)] = sum[i + 1];//y[tmp + 14400 + 3 * 32 * (i >> 1)] = sum[i + 1];
        }*/
        
    #undef tA2d
}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s1; //= y.stream_->stream_;
    cudaStream_t s2, s3, s4;// s5; //
    cudaStreamCreate(&s1);//
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);
    cudaStreamCreate(&s4);
    //cudaStreamCreate(&s5);

    // Extract the tensor dimensions into B,M,C,H,W,K

    dim3 blockDim(32, TILE_WIDTH, 1);  // 25, 25
    //dim3 gridDim(B, (Hout * Wout - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    dim3 gridDim(2500, 2, 1);
    matrix_kernel<<<gridDim, blockDim, 0, s1>>>(y.dptr_, x.dptr_, w.dptr_, 0);
    matrix_kernel<<<gridDim, blockDim, 0, s2>>>(y.dptr_, x.dptr_, w.dptr_, 2500);
    matrix_kernel<<<gridDim, blockDim, 0, s3>>>(y.dptr_, x.dptr_, w.dptr_, 5000);
    matrix_kernel<<<gridDim, blockDim, 0, s4>>>(y.dptr_, x.dptr_, w.dptr_, 7500);
    //matrix_kernel<<<gridDim, blockDim, 0, s5>>>(y.dptr_, x.dptr_, w.dptr_, 8000);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaStreamSynchronize(s3);
    cudaStreamSynchronize(s4);
    //cudaStreamSynchronize(s5);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    //MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}


}
}

#endif
