
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 25
#define TILE_HEIGHT 25
namespace mxnet
{
namespace op
{

__global__ void matrix_kernel(float *y, float *x, float *w, int offset){ //gridDim B/4, 3, 1,  blockDim 25 * 32

    __shared__ float tileA[1250];    // 50 * 25
    __shared__ float tileB[7200];    // tile width can be selected from 32, 16, 8, 6, 3, 1
    int batchId = blockIdx.x * 2;   // batch index
    int tx = threadIdx.x; // 8 (mutable)
    int ty = threadIdx.y; // 25
    float val; 
    int index = (ty << 5) + tx;
    int i,j,k,b, topR, topC, local_w, local_h, xIndex, tileBx, tileAx, tmp, col;
    float sum[18] = {0};
    float tmpB;
    #define tA2d(i1, i0) tileA[i1 * 50 + i0]
    for(b = batchId; b <= batchId + 1; ++b){
    //------------------------ load filters from global to shared memory -----------------------------        
    if(ty >= 14 && b == batchId * 2){
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
            xIndex = (b + offset) * 784 + local_h * 28 + local_w;
            val = x[xIndex];
        for(i = 0; i < 5; ++i){
            for(j = 0; j < 5; ++j){
                    topC = local_w - j;//(local_h - i) * 24 + local_w - j;
                    topR = local_h - i;
                    if(0 <= topR && topR < 12  && 0 <= topC && topC < 24){
                            tileB[(i * 5 + j) * 288 + topR * 24 + topC] = val;
                    }
            }
        }
    }
    __syncthreads();
    
    //----------------------------------- compute ------------------------------------
    //float sum[18] = {0};
    //float tmpB; 
    //int tileBx, tileAx;
    for(i = 0; i < 9; ++i){
        for( k = 0; k < TILE_WIDTH; ++k){
            tileAx = k * 50 + ty;
            tileBx = (i << 5) + tx;
            tmpB = tileB[tileBx + k * 288]; // 25 * 8
            sum[i << 1] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
            sum[(i << 1) + 1] += tileA[tileAx + 25] * tmpB; //tmpA2 * tmpB;
        }
    }

    //---------------------------- write to global memory ----------------------------
        col = tx; // 288 = 12 *24 ,   row = ty;
        tmp = (b + offset) * 28800 + ty *576 + col;
        
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i << 1]; // 3 * 8
        }
        tmp += 14400;   // 14400 = 25 * (24 * 24)
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[(i << 1) + 1]; // 3 * 8
        }
   __syncthreads(); 
 //---------------------------------- test below ---------------------------------- 
    if(ty < 14){   
        if(index < 112){
            local_w = index % 28;
            local_h = index /28 + 12;
            if(local_w < 24){
                val = tileB[5760 + (local_h - 4) * 24 + local_w];    //5760 = 20 * 288
            }else{
                val = tileB[6912 + (local_h - 4) * 24 + local_w - 4];// 6912 = 24 * 288
            }
        }else{
        local_w = index % 28;
        local_h = index / 28 + 12;
        xIndex = (b + offset) * 784 + local_h * 28 + local_w;
        val = x[xIndex];
        }
        for(i = 0; i < 5; ++i){
            for(j = 0; j < 5; ++j){
                    topC = local_w - j;
                    topR = local_h - i;
                    if(12 <= topR && topR < 24  && 0 <= topC && topC < 24){
                            tileB[(i * 5 + j) * 288 + (topR - 12) * 24 + topC] = val;
                    }
            }
        }
    }
        __syncthreads();
       memset(sum, 0, sizeof(float)*18);
       for(i = 0; i < 9; ++i){
            for(k = 0; k < TILE_WIDTH; ++k){
                tileAx = k * 50 + ty;
                tileBx = (i << 5) + tx;
                tmpB = tileB[tileBx + k * 288]; // 25 * 8
                sum[i << 1] += tileA[tileAx] * tmpB;                   //tmpA1 * tmpB;
                sum[(i << 1) + 1] += tileA[tileAx + 25] * tmpB; //tmpA2 * tmpB;
            }
        }

    //---------------------------- write to global memory ----------------------------
        col = 288 + tx; // 288 = 12 *24 ,   row = ty;
        tmp = (b + offset) * 28800 + ty *576 + col;
        
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[i << 1]; // 3 * 8
        }
        tmp += 14400;   // 14400 = 25 * (24 * 24)
        for(i = 0; i < 9; ++i){
            y[tmp + (i<<5)] = sum[(i << 1) + 1]; // 3 * 8
        }
    
 }

    #undef tA2d
}

// This function is called by new-inl.h
// Any code you write should be executed by this function
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s1, s2, s3, s4, s5;//, s6, s7, s8, s9, s10; //= y.stream_->stream_;
    //cudaStream_t s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,s27,s28,s29,s30,s31,s32; //
    cudaStreamCreate(&s1); //cudaStreamCreate(&s11);cudaStreamCreate(&s21);cudaStreamCreate(&s31);//
    cudaStreamCreate(&s2); //cudaStreamCreate(&s12);cudaStreamCreate(&s22);cudaStreamCreate(&s32);
    cudaStreamCreate(&s3);//cudaStreamCreate(&s13);cudaStreamCreate(&s23);
    cudaStreamCreate(&s4);//cudaStreamCreate(&s14);cudaStreamCreate(&s24);
    cudaStreamCreate(&s5);//cudaStreamCreate(&s15);cudaStreamCreate(&s25);
    /*
    cudaStreamCreate(&s6);cudaStreamCreate(&s16);cudaStreamCreate(&s26);
    cudaStreamCreate(&s7);cudaStreamCreate(&s17);cudaStreamCreate(&s27);
    cudaStreamCreate(&s8);cudaStreamCreate(&s18);cudaStreamCreate(&s28);
    cudaStreamCreate(&s9);cudaStreamCreate(&s19);cudaStreamCreate(&s29);
    cudaStreamCreate(&s10);cudaStreamCreate(&s20);cudaStreamCreate(&s30);
    */
    // Extract the tensor dimensions into B,M,C,H,W,K
    dim3 blockDim(32, TILE_WIDTH, 1);  // 25, 25
    //dim3 gridDim(B, (Hout * Wout - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1);
    //dim3 gridDim(300, 1, 1);
    dim3 gridDim1(1000, 1, 1);
    //for(int i = 0; i < 1; ++i){
    matrix_kernel<<<gridDim1, blockDim, 0, s1>>>(y.dptr_, x.dptr_, w.dptr_, 0); //+ i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s2>>>(y.dptr_, x.dptr_, w.dptr_, 2000); //+ i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s3>>>(y.dptr_, x.dptr_, w.dptr_, 4000); //+ i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s4>>>(y.dptr_, x.dptr_, w.dptr_, 6000); // + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s5>>>(y.dptr_, x.dptr_, w.dptr_, 8000); // + i*BATCH_SIZE);
    /*
    matrix_kernel<<<gridDim1, blockDim, 0, s6>>>(y.dptr_, x.dptr_, w.dptr_, 1600); // + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s7>>>(y.dptr_, x.dptr_, w.dptr_, 1920); // + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s8>>>(y.dptr_, x.dptr_, w.dptr_, 2240); // + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s9>>>(y.dptr_, x.dptr_, w.dptr_, 2560); // + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s10>>>(y.dptr_, x.dptr_, w.dptr_, 2880);// + i*BATCH_SIZE);
    
    matrix_kernel<<<gridDim1, blockDim, 0, s11>>>(y.dptr_, x.dptr_, w.dptr_, 3200);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s12>>>(y.dptr_, x.dptr_, w.dptr_, 3520);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s13>>>(y.dptr_, x.dptr_, w.dptr_, 3840);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s14>>>(y.dptr_, x.dptr_, w.dptr_, 4160);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s15>>>(y.dptr_, x.dptr_, w.dptr_, 4480);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s16>>>(y.dptr_, x.dptr_, w.dptr_, 4800);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s17>>>(y.dptr_, x.dptr_, w.dptr_, 5120);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s18>>>(y.dptr_, x.dptr_, w.dptr_, 5440);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s19>>>(y.dptr_, x.dptr_, w.dptr_, 5760);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim1, blockDim, 0, s20>>>(y.dptr_, x.dptr_, w.dptr_, 6080);// + i*BATCH_SIZE);
      
    matrix_kernel<<<gridDim, blockDim, 0, s21>>>(y.dptr_, x.dptr_, w.dptr_, 6400);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s22>>>(y.dptr_, x.dptr_, w.dptr_, 6700);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s23>>>(y.dptr_, x.dptr_, w.dptr_, 7000);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s24>>>(y.dptr_, x.dptr_, w.dptr_, 7300);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s25>>>(y.dptr_, x.dptr_, w.dptr_, 7600);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s26>>>(y.dptr_, x.dptr_, w.dptr_, 7900);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s27>>>(y.dptr_, x.dptr_, w.dptr_, 8200);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s28>>>(y.dptr_, x.dptr_, w.dptr_, 8500);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s29>>>(y.dptr_, x.dptr_, w.dptr_, 8800);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s30>>>(y.dptr_, x.dptr_, w.dptr_, 9100);// + i*BATCH_SIZE);
    
    matrix_kernel<<<gridDim, blockDim, 0, s31>>>(y.dptr_, x.dptr_, w.dptr_, 9400);// + i*BATCH_SIZE);
    matrix_kernel<<<gridDim, blockDim, 0, s32>>>(y.dptr_, x.dptr_, w.dptr_, 9700);// + i*BATCH_SIZE);
    */
    //}
    cudaStreamSynchronize(s1);//cudaStreamSynchronize(s11);cudaStreamSynchronize(s21);cudaStreamSynchronize(s31);
    cudaStreamSynchronize(s2);//cudaStreamSynchronize(s12);cudaStreamSynchronize(s22);cudaStreamSynchronize(s32);
    cudaStreamSynchronize(s3);//cudaStreamSynchronize(s13);cudaStreamSynchronize(s23);
    cudaStreamSynchronize(s4);//cudaStreamSynchronize(s14);cudaStreamSynchronize(s24);
    cudaStreamSynchronize(s5);//cudaStreamSynchronize(s15);cudaStreamSynchronize(s25);
    /*
    cudaStreamSynchronize(s6);cudaStreamSynchronize(s16);cudaStreamSynchronize(s26);
    cudaStreamSynchronize(s7);cudaStreamSynchronize(s17);cudaStreamSynchronize(s27);
    cudaStreamSynchronize(s8);cudaStreamSynchronize(s18);cudaStreamSynchronize(s28);
    cudaStreamSynchronize(s9);cudaStreamSynchronize(s19);cudaStreamSynchronize(s29);
    cudaStreamSynchronize(s10);cudaStreamSynchronize(s20);cudaStreamSynchronize(s30);
    */
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
