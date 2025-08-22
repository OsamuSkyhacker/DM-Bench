

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

/* New Code Start */
/* 2025-07-14 */

/**** 新 forward kernel：支持任意 hid，可配 TILE_X/TILE_Y ****/
__global__ void
bpnn_layerforward_CUDA(float *input,          /* (in+1)            */
                       float *output_hid,     /* (hid+1) - 这里没用，但接口保留 */
                       float *w_in_hid,       /* (in+1)*(hid+1)    */
                       float *partial,        /* gridDim.x * hid */
                        int   in,
                        int   hid)
{
    __shared__ float in_tile[TILE_Y];
    __shared__ float w_tile[TILE_Y][TILE_X];

    /* ① 计算全局索引 --------------------------------------------------- */
    int i_global = blockIdx.y * TILE_Y + threadIdx.y + 1;  // 输入 idx ∈ [1..in]
    int h_global = blockIdx.x * TILE_X + threadIdx.x + 1;  // 隐层 idx ∈ [1..hid]

    /* ② 把输入搬到 shared ---------------------------------------------- */
    if (threadIdx.x == 0 && i_global <= in)
        in_tile[threadIdx.y] = input[i_global];
    __syncthreads();

    /* ③ 读权重×输入 ---------------------------------------------------- */
    if (i_global <= in && h_global <= hid) {
        long idx = (long)i_global * (long)(hid + 1) + h_global;  // 列主序
        w_tile[threadIdx.y][threadIdx.x] = w_in_hid[idx] * in_tile[threadIdx.y];
    } else {
        w_tile[threadIdx.y][threadIdx.x] = 0.f;
    }
    __syncthreads();

    /* ④ y 方向归约 (树形) ---------------------------------------------- */
    for (int stride = TILE_Y >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride)
            w_tile[threadIdx.y][threadIdx.x] +=
                w_tile[threadIdx.y + stride][threadIdx.x];
        __syncthreads();
    }

    /* ⑤ 写回每个 hidden-unit 在当前 block 的局部和 --------------------- */
    if (threadIdx.y == 0 && h_global <= hid)
        atomicAdd(&partial[blockIdx.x * hid + (h_global - 1)], w_tile[0][threadIdx.x]);
}


/**** 新 adjust-weights kernel：只把 by→bx，并改列主序索引 ****/
__global__ void
bpnn_adjust_weights_cuda(float *delta,        /* (hid+1)         */
                         int   hid,
                         float *ly,           /* (in+1)          */
                         int   in,
                         float *w,            /* (in+1)*(hid+1)  */
                         float *oldw)         /* (in+1)*(hid+1)  */
{
    int bx = blockIdx.x;       /* 隐层 tile (同 forward) */
    int tx = threadIdx.x;      /* 0..TILE_X-1 → hidden   */
    int ty = threadIdx.y;      /* 0..TILE_Y-1 → input    */

    int h_global = bx * TILE_X + tx + 1;      // hidden idx
    int i_global = ty + 1;                    // input idx (不分块)

    if (h_global > hid || i_global > in) return;

    long idx = (long)i_global * (long)(hid + 1) + h_global;  // 列主序

    float dw = ETA * delta[h_global] * ly[i_global]
             + MOMENTUM * oldw[idx];

    w[idx]    += dw;
    oldw[idx]  = dw;

    /* 处理偏置项（输入 0）—— 只在一个线程里做 */
    if (i_global == 1 && ty == 0 && tx == 0 && bx == 0) {
        long b_idx = h_global;                // 行 0 的位置
        float dwb  = ETA * delta[h_global] + MOMENTUM * oldw[b_idx];
        w[b_idx]   += dwb;
        oldw[b_idx] = dwb;
    }
}

/* New Code End */

#endif