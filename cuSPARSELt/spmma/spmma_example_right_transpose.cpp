/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
//#include "stdafx.h"
#include <bitset>
#include <iostream>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

constexpr int EXIT_UNSUPPORTED = 2;

int main(void) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    // Host problem definition, row-major order
    constexpr int m     = 16; // b * s
    constexpr int n     = 16; // output dim
    constexpr int k     = 32; // input dim
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_TRANSPOSE;
    auto          type  = CUDA_R_32F;
    auto          compute_type = CUSPARSE_COMPUTE_TF32;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW); // true
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE); // false
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE); // true
    auto     num_A_rows     = (isA_transposed) ? k : m;                  // m
    auto     num_A_cols     = (isA_transposed) ? m : k;                  // k
    auto     num_B_rows     = (isB_transposed) ? n : k;                  // n
    auto     num_B_cols     = (isB_transposed) ? k : n;                  // k
    auto     num_C_rows     = m;                                         // m
    auto     num_C_cols     = n;                                         // n
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;   // Acol
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;   // Bcol
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;   // Ccol
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;   // Arow
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;   // Brow
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;   // Crow
    auto     A_size         = A_height * lda * sizeof(float);           // Arow * Acol * half
    auto     B_size         = B_height * ldb * sizeof(float);           // Brow * Bcol * half
    auto     C_size         = C_height * ldc * sizeof(float);           // Crow * Ccol * half
    float hA[m * k];
    float hB[k * n];
    float hC[m * n] = {};
    float hBias[n] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    for (int i = 0; i < 128; i=i+1) {
        hA[4*i] = static_cast<float>(static_cast<float>(1.0));
        hA[4*i+1] = static_cast<float>(static_cast<float>(2.0));
        hA[4*i+2] = static_cast<float>(static_cast<float>(3.0));
        hA[4*i+3] = static_cast<float>(static_cast<float>(4.0));
    }
    std::cout << "original weights: ";
    for (int i = 0; i < 128; i=i+1) {
        if (i % 16 <= 7) {
        hB[4*i] = static_cast<float>(static_cast<float>(1.0));
        hB[4*i+1] = static_cast<float>(static_cast<float>(0.0));
        hB[4*i+2] = static_cast<float>(static_cast<float>(0.0));
        hB[4*i+3] = static_cast<float>(static_cast<float>(4.0));
        } else {
        hB[4*i] = static_cast<float>(static_cast<float>(0.0));
        hB[4*i+1] = static_cast<float>(static_cast<float>(2.0));
        hB[4*i+2] = static_cast<float>(static_cast<float>(3.0));
        hB[4*i+3] = static_cast<float>(static_cast<float>(0.0));
        } 
        
        std::cout << hB[4*i] << " " << hB[4*i+1] << " " << hB[4*i+2] << " " << hB[4*i+3] << " ";
    }
    std::cout << std::endl;
    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    float *dA, *dB, *dC, *dD, *dB_compressed, *dBias;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &dBias, sizeof(float)*n) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dBias, hBias, sizeof(float)*n, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    size_t compressed_size;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matB, n,
                                            k, k, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )


    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize2(&handle, &matB,
                                                            &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB_compressed, compressed_size) )
    CHECK_CUSPARSE( cusparseLtSpMMACompress2(
      &handle, &matB, 0, CUSPARSE_OPERATION_TRANSPOSE, dB, dB_compressed, stream))
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matB, n,
                                            k, k, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )

    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matA, m,
                                            k, k, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, m,
                                            n, n, alignment,
                                            type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )

    CHECK_CUSPARSE( cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &dBias, sizeof(dBias)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel,
                                                 &workspace_size))

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )
    //--------------------------------------------------------------------------
    // Compress the A matrix
    // CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
    //                                               &compressed_size) )
    // CHECK_CUDA( cudaMalloc((void**) &dB_compressed, compressed_size) )

    // CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dB,
    //                                       dB_compressed, stream) )

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // check d_compressed
    void* B_compressed = new char[compressed_size];
    std::cout << compressed_size;
    CHECK_CUDA( cudaMemcpy(B_compressed, dB_compressed, compressed_size, cudaMemcpyDeviceToHost) )
    std::cout << "compressed weights: ";
    // B_compressed: 64 half values + 64 2bit values
    for (int i=0; i<64; i=i+4) {
      // tmp is the current 8-bit char (correspoding to 4 non-zero values)
      unsigned char tmp = reinterpret_cast<unsigned char*>(reinterpret_cast<float*>(B_compressed)+64)[i/4];
      std::bitset<8> x(tmp);
      // std::cout << x;
      std::cout << " " << static_cast<float>(reinterpret_cast<float*>(B_compressed)[i]);
      // std::cout << "(" << (x >> 6) << ")";
      std::cout << " " << static_cast<float>(reinterpret_cast<float*>(B_compressed)[i+1]);
      // std::cout << "(" << (x << 2 >> 6) << ")";
      std::cout << " " << static_cast<float>(reinterpret_cast<float*>(B_compressed)[i+2]);
      // std::cout << "(" << (x << 4 >> 6) << ")";
      std::cout << " " << static_cast<float>(reinterpret_cast<float*>(B_compressed)[i+3]);
      // std::cout << "(" << (x << 6 >> 6) << ")";
    }
    std::cout << std::endl;
    

    // Perform the matrix multiplication
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA, dB_compressed,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);
    // host computation
    float hC_result[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int k1 = 0; k1 < k; k1++) {
                auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                // std::cout << "m: " << i << " n: " << j << " k: " << k1 << " value: " << hB[posB] << std::endl;
                sum      += static_cast<float>(hA[posA]) *  // [i][k]
                            static_cast<float>(hB[posB]);   // [k][j]
            }
            auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            hC_result[posC] = sum + hBias[j];  // [i][j]
        }
    }
    // host-device comparison
    std::cout << "outputs:";
    int correct = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            auto device_value = static_cast<float>(hC[pos]);
            std::cout << " " << device_value; 
            auto host_value   = hC_result[pos];
            if (device_value != host_value) {
                // direct floating point comparison is not reliable
                // std::printf("(%d, %d):\t%f vs. %f\n",
                //             i, j, host_value, device_value);
                correct = 0;
                // break;
            }
        }
    }
    if (correct)
        std::printf("spmma_example test PASSED\n");
    else
        std::printf("spmma_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dB_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(dBias) )
    CHECK_CUDA( cudaFree(d_valid) )
    return EXIT_SUCCESS;
}
