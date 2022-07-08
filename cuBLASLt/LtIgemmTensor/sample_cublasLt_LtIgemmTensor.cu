/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "sample_cublasLt_LtIgemmTensor.h"
#include "helpers.h"


// this transpose kernel is supposed to work on the row-major layout
template <typename T>
__global__ void transpose_kernel(const T* src,
                                 T* dst,
                                 int row,
                                 int col) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col
  if (i<row && j<col)  dst[j * row + i] = src[i * col + j];
}

template <typename T>
void transpose_kernelLauncher(const T* input,
                              T* output,
                              int row,
                              int col,
                              cudaStream_t stream) {
  dim3 grid((row + 31) / 32, (col + 31) / 32);
  dim3 block(32, 32);
  transpose_kernel<<<grid, block, 0, stream>>>(input, output, row, col);
}

// transpose matrix & transfrom row-major to COL32 & quantize
// input matrix is (m, n) row-major
// output matrix is (n, m) COL32, using char4 to write out
// m should be a mutiple of 32
// grid((m+31)/32, (n+31)/32)
// block(8, 32)
// But have confusion about how the scale is used.
template <typename T>
__global__ void row_major_to_col32_quantize_kernel(const T* input,
                                                 char4* output,
                                                 int m,
                                                 int n)
{
    // const float scale = __ldg(scale_ptr);

    int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    int m_id = blockIdx.y * blockDim.y + threadIdx.y;

    bool check = ((m_id < m) && (n_id < n));
    if (check) {
        char4 tmp;
        tmp.x = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id) * 1.0));
        tmp.y = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id+1) * 1.0));
        tmp.z = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id+2) * 1.0));
        tmp.w = __float2int_rn(static_cast<float>(__ldg(input + m_id * n + n_id+3) * 1.0));
        // COL32_col = n_id >> 5 ; COL32_row = (m_id << 5) + (n_id & 31)
        // COL32_idx = (COL32_col << 5) * m + COL32_row = (n_id & 0xffffffe0)*m + (m_id << 5) + (n_id & 31)
        output[((n_id & 0xffffffe0) * m + (m_id << 5) + (n_id & 31)) >> 2] = tmp;
    }
}

template <typename T>
void row_major_to_col32_quantize_kernelLauncher(const T* input,
                                                int8_t* output,
                                                // T* scale,
                                                const int m,
                                                const int n,
                                                cudaStream_t stream) {
  dim3 grid((m + 31) / 32, (n + 31) / 32);
  dim3 block(32, 32);

  row_major_to_col32_quantize_kernel<<<grid, block, 0, stream>>>(
      input,
      (char4*)output,
      m,
      n);
}

// convert COL32 to row-major 
// and dequantize using weight scales and input scales
template <typename T>
__global__ void col32_to_row_major_dequantize_kernel(T* output,
                                                  const int32_t* input,
                                                  const int m,  // hidden
                                                  const int n,  // batch size
                                                  const float max_range) 
{
  int m_id = blockIdx.x * blockDim.x + threadIdx.x;  // hidden
  int n_id = blockIdx.y * blockDim.y + threadIdx.y;  // batch size

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    // int tmp = m_id * n + n_id;
    // printf("%d, %d, %d, %d\n", m_id, m, n_id, n);
    output[n_id * m + m_id] =
        ((T)(input[(m_id & 0xffffffe0) * n + (n_id << 5) + (m_id & 31)]) *
            1.0 / max_range * 1.0 / max_range);
  }
}

template <typename T>
void col32_to_row_major_dequantize_kernelLauncher(const int32_t* input,
                                                  T* output,
                                                  const int batch_size, // m
                                                  const int hidden_units,  // n
                                                  cudaStream_t stream) {
  dim3 grid((hidden_units + 31) / 32, (batch_size + 31) / 32);
  dim3 block(32, 32);

  col32_to_row_major_dequantize_kernel<<<grid, block, 0, stream>>>(
      output, input, hidden_units, batch_size, 127.0f);
}

int roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

/// Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers
///
/// For better performance data order transforms should be offline as much as possible.
///
/// transa, transb assumed N; alpha, beta are host pointers, tensor ops allowed, alpha assumed 1, beta assumed 0,
/// stream assumed 0
void LtIgemmTensor(cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const int8_t *A,
                   int lda,
                   const int8_t *B,
                   int ldb,
                   float *C,
                   int ldc,
                   cudaStream_t stream) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    int32_t alpha = 1, beta = 0;
    cublasOperation_t opTranspose = CUBLAS_OP_T;

    // tensor op igemm kernels require specialized memory order of data
    cublasLtMatrixTransformDesc_t transformDesc = NULL;
    int8_t *Atransform = NULL, *Btransform = NULL;
    int32_t *Ctransform                   = NULL;
    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
    float transformAlpha = 1.0f, transformBeta = 0.0f;
    cublasLtOrder_t order_COL32       = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

    int ldatransform = 32 * m;
    int ldbtransform = 32 * roundoff(n, 8);
    int ldctransform = 32 * m;

    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldatransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(int8_t) * roundoff(k, 32) / 32 * ldbtransform));
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(int32_t) * roundoff(n, 32) / 32 * ldctransform));

    checkCublasStatus(cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));

    checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    // tensor op igemm kernels only support NT gemm
    checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose)));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for original matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));

    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    checkCublasStatus(cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldatransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // data memory order is set to CUBLASLT_ORDER_COL4_4R2_8C in order to achieve best performance on Turing devices.
    // for best performance on Ampere, consider setting the memory order to CUBLASLT_ORDER_COL32_2R_4R4.
    checkCublasStatus(cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbtransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C, sizeof(order_COL4_4R2_8C)));

    checkCublasStatus(cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldctransform));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32)));

    // ---------------------------------------------------------------------------------------------
    // transforms and computation
    
    row_major_to_col32_quantize_kernelLauncher(A,
                                              Atransform,
                                              m,
                                              k,
                                              stream);
    
    /*
    int8_t *ATranspose;
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&ATranspose), sizeof(int8_t) * m * k));
    transpose_kernelLauncher<int8_t>(A, ATranspose, m, k, stream);
    
    std::vector<int8_t> AhostTranspose(m * k);
    checkCudaStatus(cudaMemcpyAsync(AhostTranspose.data(), ATranspose, AhostTranspose.size() * sizeof(AhostTranspose[0]), cudaMemcpyDeviceToHost, stream));
    std::cout << "transpose A: " << AhostTranspose.size();
    for (int i=0; i<AhostTranspose.size(); i++) {
        std::cout << " " << static_cast<int>(AhostTranspose.at(i));
    }
    std::cout << std::endl;
    */
    // checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0));
    
    std::vector<int8_t> AhostT(m * n);
    checkCudaStatus(cudaMemcpyAsync(AhostT.data(), Atransform, AhostT.size() * sizeof(AhostT[0]), cudaMemcpyDeviceToHost, stream));
    
    std::cout << "transform A: " << AhostT.size();
    for (int i=0; i<AhostT.size(); i++) {
        std::cout << " " << static_cast<int>(AhostT.at(i));
    }
    std::cout << std::endl;
    

    int8_t *BTranspose;
    checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BTranspose), sizeof(int8_t) * n * k));
    transpose_kernelLauncher<int8_t>(B, BTranspose, k, n, stream);
    
    // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, BTranspose, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0));

    // no need to transform C matrix as beta is assumed to be 0
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     matmulDesc,
                                     &alpha,
                                     Atransform,
                                     AtransformDesc,
                                     Btransform,
                                     BtransformDesc,
                                     &beta,
                                     Ctransform,
                                     CtransformDesc,
                                     Ctransform,
                                     CtransformDesc,
                                     NULL,
                                     NULL,
                                     0,
                                     0));

    opTranspose = CUBLAS_OP_N;
    checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));

    // transform outputs to COL order
    // checkCublasStatus(cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, Ctransform, CtransformDesc, &transformBeta, NULL, NULL, C, Cdesc, 0));


    std::cout << "transpose output";
    
    col32_to_row_major_dequantize_kernelLauncher(Ctransform,
                                                 C,
                                                 m,
                                                 n,
                                                 stream);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (CtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(CtransformDesc));
    if (BtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(BtransformDesc));
    if (AtransformDesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(AtransformDesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if (transformDesc) checkCublasStatus(cublasLtMatrixTransformDescDestroy(transformDesc));

    // wait until device is done before freeing transformed buffers
    checkCudaStatus(cudaDeviceSynchronize());
    if (Ctransform) checkCudaStatus(cudaFree(Ctransform));
    if (Btransform) checkCudaStatus(cudaFree(Btransform));
    if (Atransform) checkCudaStatus(cudaFree(Atransform));
    // if (ATranspose) checkCudaStatus(cudaFree(ATranspose));
    if (BTranspose) checkCudaStatus(cudaFree(BTranspose));
}
