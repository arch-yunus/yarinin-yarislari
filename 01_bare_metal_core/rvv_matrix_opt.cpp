/**
 * @file rvv_matrix_opt.cpp
 * @brief RISC-V Vector Extensions (RVV 1.0) Accelerated GEMV Implementation.
 * @details Implements 4-bit quantized weight unpacking and dot-product calculations
 *          optimized with hardware vector instructions (or software fallback simulation
 *          resembling XuanTie C910 RVV register-level behavior).
 */

#include "rvv_matrix_opt.h"
#include <algorithm>
#include <cmath>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

/**
 * @brief Software simulation of RISC-V Vector (RVV 1.0) registers for non-RISCV targets.
 *        Simulates vsetvli, vector load (vle32.v), vector widening arithmetic, and reduction.
 */
static void rvv_gemv_int4_fallback(const uint8_t* weights_packed,
                                   const float* inputs,
                                   float* outputs,
                                   size_t input_dim,
                                   size_t output_dim,
                                   const float* scales,
                                   const float* biases) {
    for (size_t j = 0; j < output_dim; ++j) {
        float sum = 0.0f;
        
        // Simulating RVV register chunks: vector length (vl) based on VLEN=128 bits
        // For float32, VLEN/32 = 4 elements per vector register.
        size_t i = 0;
        while (i < input_dim) {
            // vsetvli vl, avl, e32, m1
            size_t vl = std::min(static_cast<size_t>(4), input_dim - i);
            
            // Simulating v0-v31 vector registers
            float v_input[4] = {0.0f};
            float v_weight[4] = {0.0f};
            
            // vle32.v v1, (inputs + i)
            for (size_t k = 0; k < vl; ++k) {
                v_input[k] = inputs[i + k];
            }
            
            // Unpack 4-bit weights into v2
            for (size_t k = 0; k < vl; ++k) {
                size_t linear_idx = (i + k) * output_dim + j;
                size_t byte_idx = linear_idx / 2;
                bool is_high_nibble = (linear_idx % 2) == 0;
                
                uint8_t raw_byte = weights_packed[byte_idx];
                int8_t w_val = is_high_nibble ? ((raw_byte >> 4) & 0x0F) : (raw_byte & 0x0F);
                
                // Sign extend 4-bit to signed 8-bit
                if (w_val & 0x08) {
                    w_val -= 16;
                }
                v_weight[k] = static_cast<float>(w_val);
            }
            
            // vfmul.vv v3, v1, v2 (Vector Multiply)
            // vfredusum.vs v4, v3, v0 (Vector Reduction Sum)
            for (size_t k = 0; k < vl; ++k) {
                sum += v_input[k] * v_weight[k];
            }
            
            i += vl;
        }
        
        // Apply scales and biases: output = sum * scale + bias
        outputs[j] = sum * scales[j] + biases[j];
    }
}

extern "C" void rvv_gemv_int4(const uint8_t* weights_packed,
                               const float* inputs,
                               float* outputs,
                               size_t input_dim,
                               size_t output_dim,
                               const float* scales,
                               const float* biases) {
#ifdef __riscv_vector
    // If compiling for native RV64GC-NPU with vector support
    // We execute vectorized load, unpack, and multiply-accumulate
    for (size_t j = 0; j < output_dim; ++j) {
        float sum = 0.0f;
        size_t avl = input_dim;
        size_t i = 0;
        
        // Vector accumulator initialized to 0
        vfloat32m1_t v_acc = vfmv_v_f_f32m1(0.0f, vsetvl_e32m1(1));
        
        while (avl > 0) {
            size_t vl = vsetvl_e32m1(avl);
            
            // Load inputs: vle32.v
            vfloat32m1_t v_in = vle32_v_f32m1(&inputs[i], vl);
            
            // Unpack 4-bit weights dynamically and load
            float temp_weights[64]; // buffer for vector chunk
            for (size_t k = 0; k < vl; ++k) {
                size_t linear_idx = (i + k) * output_dim + j;
                size_t byte_idx = linear_idx / 2;
                bool is_high_nibble = (linear_idx % 2) == 0;
                uint8_t raw_byte = weights_packed[byte_idx];
                int8_t w_val = is_high_nibble ? ((raw_byte >> 4) & 0x0F) : (raw_byte & 0x0F);
                if (w_val & 0x08) w_val -= 16;
                temp_weights[k] = static_cast<float>(w_val);
            }
            vfloat32m1_t v_w = vle32_v_f32m1(temp_weights, vl);
            
            // Vector multiply-accumulate: vfmacc.vv
            v_acc = vfmacc_vv_f32m1(v_acc, v_in, v_w, vl);
            
            i += vl;
            avl -= vl;
        }
        
        // Reduce sum: vfredusum.vs
        float red_sum[1] = {0.0f};
        vfloat32m1_t v_red = vfmv_v_f_f32m1(0.0f, 1);
        v_red = vfredusum_vs_f32m1_f32m1(v_red, v_acc, v_red, input_dim);
        vse32_v_f32m1(red_sum, v_red, 1);
        
        outputs[j] = red_sum[0] * scales[j] + biases[j];
    }
#else
    // Hardware Vector Extension Fallback
    rvv_gemv_int4_fallback(weights_packed, inputs, outputs, input_dim, output_dim, scales, biases);
#endif
}
