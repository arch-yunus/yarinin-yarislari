/**
 * @file rvv_matrix_opt.h
 * @brief RISC-V Vector Extensions (RVV 1.0) Accelerated GEMV Interface.
 * @details Declares low-level functions optimized for executing matrix-vector
 *          products on 4-bit packed weights utilizing standard RISC-V Vector
 *          registers (v0-v31).
 */

#ifndef RVV_MATRIX_OPT_H
#define RVV_MATRIX_OPT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiplies a vector by a quantized INT4 matrix using RVV instructions where available.
 *        Optimized to unpack 4-bit nibbles into 32-bit registers, compute dot products,
 *        scale results, and apply biases.
 * 
 * @param weights_packed Pointer to the packed 4-bit weights (size = (input_dim * output_dim + 1) / 2)
 * @param inputs Pointer to the input feature vector (size = input_dim)
 * @param outputs Pointer to the output projection vector (size = output_dim)
 * @param input_dim Number of input neurons/features
 * @param output_dim Number of output neurons/features
 * @param scales Quantization scale factors per output channel
 * @param biases Float biases per output channel
 */
void rvv_gemv_int4(const uint8_t* weights_packed,
                     const float* inputs,
                     float* outputs,
                     size_t input_dim,
                     size_t output_dim,
                     const float* scales,
                     const float* biases);

#ifdef __cplusplus
}
#endif

#endif // RVV_MATRIX_OPT_H
