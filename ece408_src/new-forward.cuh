
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
__global__ void matrix_kernel(DType *y, DType *x, DType *k, int offset){ //gridDim B/4, 3, 1,  blockDim 25 * 32

    __shared__ DType tileA[1250];    // 50 * 25
    //__shared__ DType tileB[25 * 32/2 * (6*2)];
    __shared__ DType tileB[4800];
    int b = blockIdx.x;  // batch index
    int bx = blockIdx.y; // 3
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 25
    //int index = ty * 32 / 2 + tx;
    int index = ty * 16 + tx;
    #define tA2d(i1, i0) tileA[i1 * 50 + i0]
    //if(tx < 25){
            //int tileAIndex = (ty - 5) * 32 + tx;
            //int newTy = tileAIndex / 25;
            //int newTx = tileAIndex % 25;
            int kIndex = ty * 25 + tx, tAIndex = tx * 50 + ty; //tAIndex2 = (tx);
            tileA[tAIndex] = k[kIndex];
            tileA[tAIndex + TILE_WIDTH] = k[kIndex + 625];
            if(tx < 9){
                int tAIndex2 = (tx + 16) * 50 + ty;
                tileA[tAIndex2] = k[kIndex + 16];
                tileA[tAIndex2 + TILE_WIDTH] = k[kIndex + 625 + 16];
            }
            //tA2d(tx, ty) = k[kIndex];                               //A[Aindex];
            //tA2d(tx, ty + TILE_WIDTH) = k[kIndex + 625];
            //tA2d(newTx, newTy) = k[tileAIndex];
            //tA2d(newTx, newTy + TILE_WIDTH) = k[tileAIndex + 625];
    //}
    // load data from global to shared memory
    DType val[12] = {0};
    int tileTmpIndex;
    int local_w, local_h, tileFirstRowIndex, xIndex;

    if(bx == 0 && index < 100){ // 152
        local_w = index % 20;
        local_h = index / 20;
        xIndex = (b + offset) * 784 + local_h * 28 + local_w;
        for(int h = 0; h < 12; ++h){                          // 112
            val[h] = x[xIndex + 56 * h];
            //val[h] = x[(b + offset) * 784 + 0 * 784 + index + 56 * h];//x4d(b + offset, 0, local_h, local_w);
        }
        //local_w = index % 28;
        //local_h = index / 28;
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0){ //&& local_h + 5 - 1 - i < 28)
                    tileFirstRowIndex = (local_h - i) * 24 + local_w - j;
                    if(0 <= tileFirstRowIndex && tileFirstRowIndex < 16 ){ //32
                        tileTmpIndex = (i * 5 + j) * 16 + tileFirstRowIndex;
                        for(int k = 0; k < 12; ++k){
                            tileB[tileTmpIndex + 25 * 16 * k] = val[k];
                        }
                    }
                }
            }
        }
    }

    if(bx == 1 && index < 120){
        if(index < 60){
            local_w = index % 12 + 16;
            local_h = index / 12;
            xIndex = (b + offset) * 784 + local_h * 28 + local_w;
            for(int h = 0; h < 12; ++h){
                val[h] = x[xIndex + 56 * h];
                //val[h] = x[(b + offset) * 784 + 0 * 784 + index + 16 + 56 * h];
            }
            //local_w = (index + 16) % 28;
            //local_h = (index + 16) / 28;
        }else{
            local_w = (index - 60) % 12;
            local_h = (index - 60) / 12 + 1;
            xIndex = (b + offset) * 784 + local_h * 28 + local_w;
            for(int h = 0; h < 12; ++h){
                val[h] = x[xIndex + 56 * h];
            }
        }
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0){// && local_h + 5 - 1 - i<28){
                    tileFirstRowIndex = (local_h - i) * 24 + local_w - j;
                    if(16<= tileFirstRowIndex && tileFirstRowIndex < 32 ){
                        tileTmpIndex = (i * 5 + j) * 16 + tileFirstRowIndex - 16;
                        for(int k = 0; k < 12; ++k){
                            tileB[tileTmpIndex + 25 * 16 * k] = val[k];
                        }
                    }
                }
            }
        }
    }

    if(bx == 2 && index < 100){
        local_w = index % 20 + 8;
        local_h = index / 20 + 1;
        xIndex = (b + offset) * 784 + local_h * 28 + local_w;
        for(int h = 0; h < 12; ++h){
            val[h] = x[xIndex + 56 * h];
            //val[h] = x[(b + offset) * 784 + 0 * 784 + index + 36 + 56 * h];
        }
        //local_w = (index + 36) % 28;
        //local_h = (index + 36) / 28;
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0){// && local_h + 5 - 1 - i<28){
                    tileFirstRowIndex = (local_h - i) * 24 + local_w - j;
                    if(32 <= tileFirstRowIndex && tileFirstRowIndex < 48){
                        tileTmpIndex = (i * 5 + j) * 16 + tileFirstRowIndex - 32;
                        for(int k = 0; k < 12; ++k){
                            tileB[tileTmpIndex + 25 * 16 * k] = val[k];
                        }
                    }
                }
            }
        }
    }

    __syncthreads();
    // compute
    DType sum[24] = {0};
    DType tmpA1, tmpA2, tmpB;
    int tileBIndex;
    for(int k = 0; k < TILE_WIDTH; ++k){
        tmpA1 = tA2d(k, ty); tmpA2 = tA2d(k, ty + 25);
        tileBIndex = k * 16 + tx;
        for(int i = 0; i < 12; ++i){
            tmpB = tileB[tileBIndex + 400 *i];
            sum[i * 2] += tmpA1 * tmpB;
            sum[i * 2 + 1] += tmpA2 * tmpB;
        }
    }
    //__syncthreads();

    //write to global memory
        int col = bx * 16 + tx; // row = ty;
        int tmp = (b + offset) * 28800 + ty *576 + col;
        for(int i = 0; i < 12; ++i){
            y[tmp + 48 * i] = sum[i << 1];
        }
        tmp += 14400;
        for(int i = 0; i < 12; ++i){
            y[tmp + 48 * i] = sum[i*2 + 1];
        }
        /*
        for(int i = 0; i < 24; i += 2){
            y[tmp + 48 * (i >> 1)] = sum[i];//y[tmp + 3 * 32 * (i >> 1)] = sum[i];
            y[tmp + 14400 + 48 * (i >> 1)] = sum[i + 1];//y[tmp + 14400 + 3 * 32 * (i >> 1)] = sum[i + 1];
        }
        */
    #undef tA2d
}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s1; //= y.stream_->stream_;
    cudaStream_t s2, s3, s4,s5; //
    cudaStreamCreate(&s1);//
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);
    cudaStreamCreate(&s4);
    cudaStreamCreate(&s5);

    // Extract the tensor dimensions into B,M,C,H,W,K

    dim3 blockDim(16, TILE_WIDTH, 1);  // 25, 25
    //dim3 gridDim(B, (Hout * Wout - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    dim3 gridDim(2000, 3, 1);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s1>>>(y.dptr_, x.dptr_, w.dptr_, 0);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s2>>>(y.dptr_, x.dptr_, w.dptr_, 2000);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s3>>>(y.dptr_, x.dptr_, w.dptr_, 4000);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s4>>>(y.dptr_, x.dptr_, w.dptr_, 6000);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s5>>>(y.dptr_, x.dptr_, w.dptr_, 8000);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaStreamSynchronize(s3);
    cudaStreamSynchronize(s4);
    cudaStreamSynchronize(s5);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

}
}

#endif
