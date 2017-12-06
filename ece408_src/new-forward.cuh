
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 25
namespace mxnet
{
namespace op
{

__global__ void matrix_kernel(float *y, float *x, float *w){ //gridDim B/4, 3, 1,  blockDim 25 * 32

    __shared__ float tileA[1250];    // 50 * 25
    __shared__ float tileB[7200];    // tile width can be selected from 32, 16, 8, 6, 3, 1
    int b = blockIdx.x << 1;   // batch index
    int tx = threadIdx.x; // 
    int ty = threadIdx.y; //
    float val;
    int index = (ty << 5) + tx, image_index = ty * 576 + tx;
    int i,j,topR, topC, local_w, local_h, xIndex, tileBx, tileAx, tmp; // col;
    float sum[18] = {0};
    float tmpB;
    //------------------------ load filters from global to shared memory -----------------------------        
    if(ty >= 14){
            int kIndex = ((ty - 14) << 5) + tx;
            int tAIndex = kIndex / 25 + kIndex % 25 * 50;
            tileA[tAIndex] = w[kIndex];
            tileA[tAIndex + TILE_WIDTH] = w[kIndex + 625];
            kIndex += 352;
            if(kIndex < 625){
                tAIndex = kIndex / 25 + kIndex % 25 * 50;
                tileA[tAIndex] = w[kIndex];
                tileA[tAIndex + TILE_WIDTH] = w[kIndex + 625];
            }
    }else{
    //------------------------ load data from global to shared memory ----------------------------
            local_w = index % 28;// + blockOffset; //8 * bx;
            local_h = index / 28;// + bx * 12;
            xIndex = b * 784 + local_h * 28 + local_w;
            val = x[xIndex];
        for(i = 0; i < 5; ++i){
            for(j = 0; j < 5; ++j){
                    topC = local_w - j;
                    topR = local_h - i;
                    if(0 <= topR && topR < 12  && 0 <= topC && topC < 24){
                            tileB[(i * 5 + j) * 288 + topR * 24 + topC] = val;
                    }
            }
        }
    }
    __syncthreads();
    
    //----------------------------------- compute ------------------------------------
    if(ty < 25){
    for(i = 0; i < 9; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 9] += tileA[tileAx + 25] * tmpB; //tmpA2 * tmpB;
        }
    }
    }
    //---------------------------- write to global memory ----------------------------
        //col = tx; // 288 = 12 *24 ,   row = ty;
        tmp = b * 28800 + image_index;
        
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i]; // 3 * 8
        }
        tmp += 14400;   // 14400 = 25 * (24 * 24)
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i + 9]; // 3 * 8
        }
   __syncthreads(); 
 //---------------------------------- test below ---------------------------------- 
    if(ty < 14){   
        val = x[xIndex + 12*28];
        for(i = 0; i < 5; ++i){
            for(j = 0; j < 5; ++j){
                    topC = local_w - j;
                    topR = local_h + 12 - i;
                    if(12 <= topR && topR < 24  && 0 <= topC && topC < 24){
                            tileB[(i * 5 + j) * 288 + (topR - 12) * 24 + topC] = val;
                    }
            }
        }
    }
        __syncthreads();
       memset(sum, 0, 72);
       for(i = 0; i < 9; ++i){
            for(j = 0; j < TILE_WIDTH; ++j){
                tileAx = j * 50 + ty;
                tileBx = (i << 5) + tx;
                tmpB = tileB[tileBx + j * 288];
                sum[i] += tileA[tileAx] * tmpB;
                sum[i + 9] += tileA[tileAx + 25] * tmpB; 
            }
        }

    //---------------------------- write to global memory ----------------------------
        //col = 288 + tx; // 288 = 12 *24 ,   row = ty;
        
        tmp = b * 28800 + image_index + 288;
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i];
        }
        tmp += 14400;   // 14400 = 25 * (24 * 24)
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i + 9];
        }
     __syncthreads();
    // --------------------------  another batch ------------------------
      b += 1;
     if(ty < 14){
        val = x[xIndex + 784];
         for(i = 0; i < 5; ++i){
            for(j = 0; j < 5; ++j){
                    topC = local_w - j;
                    topR = local_h - i;
                    if(0 <= topR && topR < 12  && 0 <= topC && topC < 24){
                            tileB[(i * 5 + j) * 288 + (topR) * 24 + topC] = val;
                    }
            }
        }

     }
        __syncthreads();
       memset(sum, 0, 72);
       for(i = 0; i < 9; ++i){
            for(j = 0; j < TILE_WIDTH; ++j){
                tileAx = j * 50 + ty;
                tileBx = (i << 5) + tx;
                tmpB = tileB[tileBx + j * 288]; // 25 * 8
                sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
                sum[i + 9] += tileA[tileAx + 25] * tmpB; //tmpA2 * tmpB;
            }
        }
        
        tmp = b * 28800 + image_index; //col;
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i]; // 3 * 8
        }
        tmp += 14400;   // 14400 = 25 * (24 * 24)
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i + 9]; // 3 * 8
        }
        __syncthreads();

     if(ty < 14){
        val = x[xIndex + 784 + 12*28];
         for(i = 0; i < 5; ++i){
            for(j = 0; j < 5; ++j){
                    topC = local_w - j;
                    topR = local_h + 12 - i;
                    if(12 <= topR && topR < 24  && 0 <= topC && topC < 24){
                            tileB[(i * 5 + j) * 288 + (topR - 12) * 24 + topC] = val;
                    }
            }
        }

     }
     __syncthreads();
       memset(sum, 0, 72);
       for(i = 0; i < 9; ++i){
            for(j = 0; j < TILE_WIDTH; ++j){
                tileAx = j * 50 + ty;
                tileBx = (i << 5) + tx;
                tmpB = tileB[tileBx + j * 288]; // 25 * 8
                sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
                sum[i + 9] += tileA[tileAx + 25] * tmpB; //tmpA2 * tmpB;
            }
        }
        tmp = b * 28800 + image_index + 288; //col;
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i]; // 3 * 8
        }
        tmp += 14400;   // 14400 = 25 * (24 * 24)
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i + 9]; // 3 * 8
        }


}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    const int B = x.shape_[0];
    //printf("size of float = %d, B = %d \n", sizeof(float), B);
    //cudaStream_t s1, s2;// s3, s4; //s5;//, s6, s7, s8, s9, s10; //= y.stream_->stream_;
    // Extract the tensor dimensions into B,M,C,H,W,K
    dim3 blockDim(32, 32, 1);  // 25, 25
    //dim3 gridDim(B, (Hout * Wout - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    //dim3 gridDim(300, 1, 1);
    dim3 gridDim1(B >> 1, 1, 1);
    matrix_kernel<<<gridDim1, blockDim>>>(y.dptr_, x.dptr_, w.dptr_); 
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
