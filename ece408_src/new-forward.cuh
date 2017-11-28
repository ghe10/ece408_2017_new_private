
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
    __shared__ DType tileB1[25 * 32];   //[TILE_WIDTH * TILE_HEIGHT];
    __shared__ DType tileB2[25 * 32];
    __shared__ DType tileB3[25 * 32];
    //__shared__ DType tileB4[25 * 16];
    #define tA2d(i1, i0) tileA[i1 * 50 + i0]
    #define tB2d1(i1, i0) tileB1[i1 * 32 + i0]
    #define tB2d2(i1, i0) tileB2[i1 * 32 + i0]
    #define tB2d3(i1, i0) tileB3[i1 * 32 + i0]
    //#define tB2d4(i1, i0) tileB4[i1 * 16 + i0]

    int b = blockIdx.x;  // batch index 
    int bx = blockIdx.y; // 
    int by = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = ty; //by * TILE_HEIGHT + ty;
    int Col = bx * 32 + tx;// Col2 = Col + 9 * 32;
    int numAColumns = 25, numBColumns = 576;
    DType sum[6] = {0, 0, 0, 0, 0, 0};
    int Aindex, Bindex;
    DType tmpB, tmpA1, tmpA2;
       if(tx < 25){
            Aindex = Row * numAColumns + tx; //m * TILE_WIDTH + tx;
            tA2d(tx, ty) = A[Aindex];
            tA2d(tx, ty + TILE_WIDTH) = A[Aindex + 625];
       }
       //if(Col < numBColumns){
            Bindex = b * 14400 + ty * numBColumns + Col;
            tB2d1(ty, tx) = B[Bindex];
            tB2d2(ty, tx) = B[Bindex + 6 * 32];
            tB2d3(ty, tx) = B[Bindex + 6 * 32 * 2];
            //tB2d4(ty, tx) = B[Bindex + 9 * 16 * 3];
       //}
       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k){
            tmpA1 = tA2d(k, ty); tmpA2 = tA2d(k, ty + 25);
            tmpB = tB2d1(k, tx); 
            sum[0] += tmpA1 * tmpB; //tB2d(k, tx);
            sum[1] += tmpA2 * tmpB; //tB2d(k, tx);
            tmpB = tB2d2(k, tx);
            sum[2] += tmpA1 * tmpB;
            sum[3] += tmpA2 * tmpB;
            tmpB = tB2d3(k, tx);
            sum[4] += tmpA1 * tmpB;
            sum[5] += tmpA2 * tmpB;
       }
       //__syncthreads();
    //if(Col < 576){
        int tmp = (b + offset) * 28800 + Row *576 + Col;
        for(int i = 0; i < 6; i += 2){
            y[tmp + 6 * 32 * (i >> 1)] = sum[i];
            y[tmp + 14400 + 6 * 32 * (i >> 1)] = sum[i + 1];
        }
        //y[tmp + 9 * 16] = sum2;
        //y[tmp + 14400 + 9 * 32] = sum3;
        //y[b * 28800 + Row *576 + Col] = sum;
        //y[b * 28800 + (Row +25)*576 + Col] = sum1;
    //}
    #undef y4d
    #undef tA2d
    #undef tB2d1
    #undef tB2d2
    #undef tB2d3
}

template<typename gpu, typename DType>
__global__ void matrix_kernel(DType *y, DType *x, DType *k, int offset){ //gridDim B/4, 3, 1,  blockDim 25 * 32
    
    __shared__ DType tileA[1250];    // 50 * 25
    __shared__ DType tileB1[25 * 32];//
    __shared__ DType tileB2[25 * 32];
    __shared__ DType tileB3[25 * 32];
    __shared__ DType tileB4[25 * 32];
    __shared__ DType tileB5[25 * 32];
    __shared__ DType tileB6[25 * 32];
    
    int b = blockIdx.x;  // batch index 
    int bx = blockIdx.y; // 3
    //int by = blockIdx.z; // 1
    int tx = threadIdx.x; // 32
    int ty = threadIdx.y; // 25
    int index = ty * 32 + tx;
    #define tA2d(i1, i0) tileA[i1 * 50 + i0]
    //DType val[6] = {0};
    //int tileTmpIndex = 0; 
    //int local_w, local_h, tileFirstRowIndex;
    // load filters from global to shared memory
   if(ty >= 5){ 
        //if(tx < 25){
        if(index < 785){
            int tileAIndex = (ty - 5) * 32 + tx;
            int newTy = tileAIndex / 25;
            int newTx = tileAIndex % 25;
            //int kIndex = ty * 25 + tx; 
            //tA2d(tx, ty) = k[kIndex];                               //A[Aindex];
            //tA2d(tx, ty + TILE_WIDTH) = k[kIndex + 625];
            tA2d(newTx, newTy) = k[tileAIndex];
            tA2d(newTx, newTy + TILE_WIDTH) = k[tileAIndex + 625];
        }
    }else{
    // load data from global to shared memory
    DType val[6] = {0};
    int tileTmpIndex = 0;
    int local_w, local_h, tileFirstRowIndex;
    /*
    if(bx == 0 && index < 152){
        for(int h = 0; h < 6; ++h){
            val[h] = x[(b + offset) * 784 + 0 * 784 + index + 112 * h]; //x4d(b + offset, 0, local_h, local_w);
        }   
        local_w = index % 28;
        local_h = index / 28;
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0){ //&& local_h + 5 - 1 - i < 28)
                    tileFirstRowIndex = (local_h - i) * 24 + local_w - j;
                    if(0 <= tileFirstRowIndex && tileFirstRowIndex < 32 ){
                        tileTmpIndex = (i * 5 + j) * 32 + tileFirstRowIndex;
                        tileB1[tileTmpIndex] = val[0];
                        tileB2[tileTmpIndex] = val[1];
                        tileB3[tileTmpIndex] = val[2];
                        tileB4[tileTmpIndex] = val[3];
                        tileB5[tileTmpIndex] = val[4];
                        tileB6[tileTmpIndex] = val[5];
                    }
                }
            }
        }
    }

    if(bx == 1 && index < 152){
        for(int h = 0; h < 6; ++h){
            val[h] = x[(b + offset) * 784 + 0 * 784 + index + 36 + 112 * h];
        }
            local_w = (index + 36) % 28;
            local_h = (index + 36) / 28;
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0){// && local_h + 5 - 1 - i<28){
                    tileFirstRowIndex = (local_h - i) * 24 + local_w - j;
                    if(32<= tileFirstRowIndex && tileFirstRowIndex < 64 ){
                        tileTmpIndex = (i * 5 + j) * 32 + tileFirstRowIndex - 32;
                        tileB1[tileTmpIndex] = val[0];
                        tileB2[tileTmpIndex] = val[1];
                        tileB3[tileTmpIndex] = val[2];
                        tileB4[tileTmpIndex] = val[3];
                        tileB5[tileTmpIndex] = val[4]; 
                        tileB6[tileTmpIndex] = val[5]; 
                    }
                }
            }
        }
    }

    if(bx == 2 && index < 152){
        for(int h = 0; h < 6; ++h){
            val[h] = x[(b + offset) * 784 + 0 * 784 + index + 72 + 112 * h];
        }    
        local_w = (index + 72) % 28;
        local_h = (index + 72) / 28;
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0){// && local_h + 5 - 1 - i<28){
                    tileFirstRowIndex = (local_h - i) * 24 + local_w - j;
                    if(64 <= tileFirstRowIndex && tileFirstRowIndex < 96){
                        tileTmpIndex = (i * 5 + j) * 32 + tileFirstRowIndex - 64;
                            tileB1[tileTmpIndex] = val[0];
                            tileB2[tileTmpIndex] = val[1];
                            tileB3[tileTmpIndex] = val[2];
                            tileB4[tileTmpIndex] = val[3];
                            tileB5[tileTmpIndex] = val[4];
                            tileB6[tileTmpIndex] = val[5];
                    }
                }
            }
        }
    }
    */
    int local_index, xIndex, bound;
    if(index < 152){
        xIndex = (b + offset) * 784 + index + 36 * bx;
        for(int h = 0; h < 6; ++h){
            val[h] = x[xIndex + 112 * h];
        }
        local_index = (index + 36 * bx);
        local_w = local_index % 28;
        local_h = local_index / 28;
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j < 5; ++j){
                if(local_w - j >= 0 && local_w+5-1-j < 28 && local_h - i >= 0){
                    tileFirstRowIndex = (local_h - i) * 24 + local_w - j;
                    bound = 32*bx;
                    if(bound <= tileFirstRowIndex && tileFirstRowIndex < bound+32){
                        tileTmpIndex = (i * 5 + j) * 32 + tileFirstRowIndex - bound;
                            tileB1[tileTmpIndex] = val[0];
                            tileB2[tileTmpIndex] = val[1];
                            tileB3[tileTmpIndex] = val[2];
                            tileB4[tileTmpIndex] = val[3];
                            tileB5[tileTmpIndex] = val[4];
                            tileB6[tileTmpIndex] = val[5];
                    }
                }
            }
        }
    }

    }
    __syncthreads();
    // compute
    DType sum[12] = {0};
    DType tmpA1, tmpA2, tmpB;
    int tileBIndex;
    for(int k = 0; k < TILE_WIDTH; ++k){
        tmpA1 = tA2d(k, ty); tmpA2 = tA2d(k, ty + 25);
        tileBIndex = k * 32 + tx;
        tmpB = tileB1[tileBIndex];  
        sum[0] += tmpA1 * tmpB; 
        sum[1] += tmpA2 * tmpB;
        tmpB = tileB2[tileBIndex];
        sum[2] += tmpA1 * tmpB;
        sum[3] += tmpA2 * tmpB;
        tmpB = tileB3[tileBIndex];
        sum[4] += tmpA1 * tmpB;
        sum[5] += tmpA2 * tmpB; 
        tmpB = tileB4[tileBIndex];
        sum[6] += tmpA1 * tmpB;
        sum[7] += tmpA2 * tmpB; 
        tmpB = tileB5[tileBIndex];
        sum[8] += tmpA1 * tmpB;
        sum[9] += tmpA2 * tmpB; 
        tmpB = tileB6[tileBIndex];
        sum[10] += tmpA1 * tmpB;
        sum[11] += tmpA2 * tmpB; 
    }
    //__syncthreads();
    
    //write to global memory
        int row = ty, col = bx * 32 + tx;
        int tmp = (b + offset) * 28800 + row *576 + col;
        for(int i = 0; i < 12; i += 2){
            y[tmp + 96 * (i >> 1)] = sum[i];//y[tmp + 3 * 32 * (i >> 1)] = sum[i];
            y[tmp + 14400 + 96 * (i >> 1)] = sum[i + 1];//y[tmp + 14400 + 3 * 32 * (i >> 1)] = sum[i + 1];
        }
 
    #undef tA2d
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
    if(bid > 0){  // load kernel
        if(b == 0 && local_h < 25 && local_w < 25){
            int m = (bid - 1) * 25 + local_h;
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
    cudaStreamCreate(&s4); //cudaStreamCreate(&s5);
    
    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = w.shape_[0];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[2];
    printf("The filter size is K * K, where K = %d \n", K);
    printf("B = %d, M = %d, C = %d, H = %d, W = %d \n", B, M, C, H, W);
    //int Hout = H - K + 1;
    //int Wout = W - K + 1;     
    /*
    DType *X_unrolled1;
    DType *X_unrolled2;
    DType *X_unrolled3;
    DType *X_unrolled4;
    //DType *X_unrolled5;
    DType *W_unrolled1;
    //DType *W_unrolled2;
    //DType *W_unrolled3;
    //DType *W_unrolled4;
    cudaMalloc(&X_unrolled1, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&X_unrolled2, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&X_unrolled3, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&X_unrolled4, B/4 * K * K * Hout * Wout * sizeof(DType));
    cudaMalloc(&W_unrolled1, M * K * K * sizeof(DType));
    */
    // Set the kernel dimensions,    
    dim3 blockDimU(28, 28, 1); // 
    dim3 gridDimU(B/4, 3, 1);    // 
    dim3 gridDimU2(B/4, 1, 1);
    //cudaDeviceSynchronize();

    dim3 blockDim(32, TILE_WIDTH, 1);  // 25, 25
    //dim3 gridDim(B, (Hout * Wout - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    dim3 gridDim(B/4, 18/6, 1);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s1>>>(y.dptr_, x.dptr_, w.dptr_, 0);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s2>>>(y.dptr_, x.dptr_, w.dptr_, 2500);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s3>>>(y.dptr_, x.dptr_, w.dptr_, 5000);
    matrix_kernel<gpu,DType><<<gridDim, blockDim, 0, s4>>>(y.dptr_, x.dptr_, w.dptr_, 7500);
    // Call the kernel                                0 is sharemem s is stream
    //for(int i = 0; i < 1; ++i){
    /*
    unroll_kernel<gpu,DType><<<gridDimU, blockDimU, 0, s1>>>(X_unrolled1, W_unrolled1, x.dptr_, w.dptr_,0);
    unroll_kernel<gpu,DType><<<gridDimU2, blockDimU, 0, s2>>>(X_unrolled2, W_unrolled1, x.dptr_, w.dptr_,2500);
    unroll_kernel<gpu,DType><<<gridDimU2, blockDimU, 0, s3>>>(X_unrolled3, W_unrolled1, x.dptr_, w.dptr_,5000);
    unroll_kernel<gpu,DType><<<gridDimU2, blockDimU, 0, s4>>>(X_unrolled4, W_unrolled1, x.dptr_, w.dptr_,7500);
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s1>>>(y.dptr_, X_unrolled1, W_unrolled1,H,W,K,0);
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s2>>>(y.dptr_, X_unrolled2, W_unrolled1,H,W,K,2500);
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s3>>>(y.dptr_, X_unrolled3, W_unrolled1,H,W,K,5000);
    forward_mul_kernel<gpu, DType><<<gridDim, blockDim, 0, s4>>>(y.dptr_, X_unrolled4, W_unrolled1,H,W,K,7500);
    */
    //}
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaStreamSynchronize(s3);
    cudaStreamSynchronize(s4);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    /*cudaFree(X_unrolled1);
    cudaFree(X_unrolled2);
    cudaFree(X_unrolled3);
    cudaFree(X_unrolled4);//cudaFree(X_unrolled5);
    cudaFree(W_unrolled1);
    */
    //cudaFree(W_unrolled2);
    //cudaFree(W_unrolled3);
    //cudaFree(W_unrolled4);
}

}
}

#endif
