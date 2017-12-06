
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
    int b = blockIdx.x << 2;   // batch index
    int tx = threadIdx.x; // 
    int ty = threadIdx.y; //
    float val;
    int index = (ty << 5)*3 + tx, image_index = ty * 576 + tx;
    int i,j,topR, topC, local_w, local_h, xIndex, tileBx, tileAx, tmp; // col;
    float sum[15] = {0}; // 18
    float tmpB;
    //------------------------ load filters from global to shared memory -----------------------------        
    if(index >= 448){
            int kIndex = index - 448;
            int tAIndex = kIndex / 25 + kIndex % 25 * 50;
            tileA[tAIndex] = w[kIndex];
            tileA[tAIndex + TILE_WIDTH] = w[kIndex + 625];
            kIndex += 512;  // 512 = 960 - 448
            if(kIndex < 625){
                tAIndex = kIndex / 25 + kIndex % 25 * 50;
                tileA[tAIndex] = w[kIndex];
                tileA[tAIndex + TILE_WIDTH] = w[kIndex + 625];
            }
    }else{
    //------------------------ load data from global to shared memory ----------------------------
            local_w = index % 28;// 
            local_h = index / 28;//
            xIndex = b * 784 + index;//local_h * 28 + local_w;
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
    for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
    }
    //---------------------------- write to global memory ----------------------------
        tmp = b * 28800 + image_index;
        for(i = 0; i < 15; i += 3){
        //y[tmp + (i / 3)*5760 + (i%3)*96] = sum[i];
         y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;

        }
        __syncthreads(); 
 //---------------------------------- test below ---------------------------------- 
    if(index < 448){   
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
    memset(sum, 0, 60); // 60 = 4 * 15
    for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
    }
    //---------------------------- write to global memory ----------------------------
        tmp = b * 28800 + image_index + 288;
        for(i = 0; i < 15; i += 3){
            y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;
        }
        __syncthreads();
    // --------------------------  second batch ------------------------
      b += 1;
     if(index < 448){
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
     memset(sum, 0, 60);
     for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
     }
        tmp = b * 28800 + image_index; //col;
        for(i = 0; i < 15; i += 3){
            y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;
        }
     __syncthreads();
     if(index < 448){
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
     memset(sum, 0, 60);
     for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
     }
     tmp = b * 28800 + image_index + 288; //col;
     for(i = 0; i < 15; i += 3){
             y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;
     } 
     __syncthreads();
   // ----------------------------  third batch -------------------------
     b += 1;
     if(index < 448){
        val = x[xIndex + 1568];
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
     memset(sum, 0, 60);
     for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
     }
        tmp = b * 28800 + image_index; //col;
        for(i = 0; i < 15; i += 3){
            y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;
        }
     __syncthreads();
     if(index < 448){
        val = x[xIndex + 784*2 + 12*28];
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
     memset(sum, 0, 60);
     for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
     }
     tmp = b * 28800 + image_index + 288; //col;
     for(i = 0; i < 15; i += 3){
             y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;
     }
     __syncthreads();
// -----------------------------  fourth batch -----------------------
     b += 1;
     if(index < 448){
        val = x[xIndex + 784*3];
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
     memset(sum, 0, 60);
     for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
     }
        tmp = b * 28800 + image_index; //col;
        for(i = 0; i < 15; i += 3){
            y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;
        }
     __syncthreads();
     if(index < 448){
        val = x[xIndex + 784*3 + 12*28];
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
     memset(sum, 0, 60);
     for(i = 0; i < 3; ++i){
        for( j = 0; j < TILE_WIDTH; ++j){
            tileAx = j * 50 + ty;
            tileBx = (i << 5) * 3 + tx;
            tmpB = tileB[tileBx + j * 288]; // 25 * 8
            sum[i] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[i + 3] += tileA[tileAx + 10] * tmpB; //tmpA2 * tmpB;
            sum[i + 6] += tileA[tileAx + 20] * tmpB;
            sum[i + 9] += tileA[tileAx + 30] * tmpB;
            sum[i + 12] += tileA[tileAx + 40] * tmpB;
        }
     }
     tmp = b * 28800 + image_index + 288; //col;
     for(i = 0; i < 15; i += 3){
             y[tmp ] = sum[i];
            y[tmp + 96] = sum[i + 1];
            y[tmp + 192] = sum[i + 2];
            tmp += 5760;
     } 
}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    const int B = x.shape_[0] >> 2;
    dim3 blockDim(96, 10, 1);  // 25, 25
    dim3 gridDim(B, 1, 1);
    matrix_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_); 
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}


}
}

#endif
