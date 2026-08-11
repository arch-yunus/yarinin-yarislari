#include "bsp_register_map.h"
#include "rvv_matrix_opt.h"
#include <array>
#include <concepts>
#include <algorithm>
#include <numbers>
#include <cmath>

#ifdef HOST_SIMULATION
extern "C" {
uint8_t simulated_npu_mem[sizeof(NPU_TypeDef)] = {0};
uint8_t simulated_actuator_mem[sizeof(Actuator_TypeDef) * ACTUATOR_COUNT] = {0};
uint8_t simulated_power_sys_mem[sizeof(PowerSys_TypeDef)] = {0};
uint8_t simulated_imu_mem[sizeof(IMU_TypeDef)] = {0};
}
#endif

// Control Loop Constants
constexpr float DT = 0.002f;              // 500 Hz control loop frequency (2 ms)
constexpr float IMU_LPF_ALPHA = 0.25f;    // Low-Pass Filter smoothing factor
constexpr float MAX_TORQUE_NM = 60.0f;    // Maximum motor torque limit (N-m)

// Template concepts for safety and bounds checks
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

/**
 * @brief First-order Low-Pass Filter for sensor noise reduction
 */
class LowPassFilter {
private:
    float prev_val{0.0f};
    float alpha{0.1f};

public:
    constexpr LowPassFilter() = default;
    constexpr LowPassFilter(float cut_off_alpha) : alpha(cut_off_alpha) {}

    float apply(float input) {
        float output = (alpha * input) + ((1.0f - alpha) * prev_val);
        prev_val = output;
        return output;
    }

    void reset() {
        prev_val = 0.0f;
    }
};

/**
 * @brief PID Controller with Integrator Anti-Windup
 */
class PidController {
private:
    float kp{1.0f};
    float ki{0.0f};
    float kd{0.0f};
    float integral{0.0f};
    float prev_error{0.0f};
    float limit{50.0f};

public:
    constexpr PidController() = default;
    constexpr PidController(float p, float i, float d, float lim) 
        : kp(p), ki(i), kd(d), limit(lim) {}

    float calculate(float setpoint, float measurement) {
        float error = setpoint - measurement;
        
        // Proportional term
        float p_out = kp * error;
        
        // Integral term with windup limit clamp
        integral += error * DT;
        float i_out = ki * integral;
        if (i_out > limit) {
            i_out = limit;
            integral = limit / ki;
        } else if (i_out < -limit) {
            i_out = -limit;
            integral = -limit / ki;
        }
        
        // Derivative term
        float derivative = (error - prev_error) / DT;
        float d_out = kd * derivative;
        
        prev_error = error;
        
        // Combine and clamp output
        float total_out = p_out + i_out + d_out;
        return std::clamp(total_out, -limit, limit);
    }

    void reset() {
        integral = 0.0f;
        prev_error = 0.0f;
    }
};

// Global filter and controller instances for the balancing axis
static LowPassFilter pitch_gyro_filter(IMU_LPF_ALPHA);
static LowPassFilter roll_gyro_filter(IMU_LPF_ALPHA);
static PidController balance_pid(1.5f, 0.2f, 0.18f, MAX_TORQUE_NM);

// Raw IMU conversion helpers (based on simulated sensor sensitivities)
constexpr float RAW_GYRO_TO_RADS = 0.0001f;
constexpr float RAW_ACCEL_TO_MPS2 = 0.000598f; // Converts 16-bit raw back to Gs/meters-per-sec2

inline int16_t convert_torque_to_mNm(float torque_nm) {
    float clamped = std::clamp(torque_nm, -MAX_TORQUE_NM, MAX_TORQUE_NM);
    return static_cast<int16_t>(clamped * 1000.0f);
}

/**
 * @brief Calibrates joint limits and zeroes out actuator encoders on startup
 */
extern "C" bool run_joint_calibration() {
    // 1. Enter Low-Current Calibration mode
    for (int i = 0; i < ACTUATOR_COUNT; ++i) {
        ACTUATOR[i].MODE = 0; // Idle
    }

    // 2. Slow sweep to locate mechanical hardstops
    bool calibration_success = true;
    for (int i = 0; i < ACTUATOR_COUNT; ++i) {
        volatile Actuator_TypeDef* joint = &ACTUATOR[i];
        
        // Check for thermal faults before homing
        if (joint->FAULT_STATUS & 0x01) {
            calibration_success = false;
            break;
        }
        
        // Simulating register command to trigger internal actuator homing sequence
        joint->MODE = 2; // Position mode
        joint->TARGET_POS = 0; // Return to zero index offset
    }

    // Reset filters
    pitch_gyro_filter.reset();
    roll_gyro_filter.reset();
    balance_pid.reset();

    return calibration_success;
}

/**
 * @brief System initialization routine called by bootloader
 */
extern "C" void init_system() {
    // Hardware Reset
    NPU->CTRL |= NPU_CTRL_RESET;
    for (volatile int i = 0; i < 200; ++i); // Delay loop
    NPU->CTRL &= ~NPU_CTRL_RESET;

    // Run calibration
    if (run_joint_calibration()) {
        // Enable balancing mode by switching actuators to torque control (Mode 1)
        for (int i = 0; i < ACTUATOR_COUNT; ++i) {
            ACTUATOR[i].MODE = 1;
            ACTUATOR[i].TARGET_TORQUE = 0;
        }
        NPU->IRQ_EN |= 1; // Enable NPU inference completion interrupt
    }
}

/**
 * @brief Runs NPU hardware inference using RVV matrix acceleration kernels.
 *        This models the actual coprocessor execution flow.
 */
extern "C" void trigger_npu_hardware_inference() {
    if (!(NPU->CTRL & NPU_CTRL_START)) {
        return;
    }
    
    NPU->STATUS &= ~NPU_STATUS_DONE;
    NPU->STATUS |= NPU_STATUS_BUSY;
    
    const auto* npu_inputs = reinterpret_cast<const float*>(NPU->INPUT_ADDR);
    auto* npu_outputs = reinterpret_cast<float*>(NPU->OUTPUT_ADDR);
    const auto* packed_weights = reinterpret_cast<const uint8_t*>(NPU->WEIGHT_ADDR);
    
    if (npu_inputs == nullptr || npu_outputs == nullptr || packed_weights == nullptr) {
        NPU->STATUS &= ~NPU_STATUS_BUSY;
        NPU->STATUS |= NPU_STATUS_ERR;
        return;
    }
    
    struct NpuWeightsLayout {
        uint8_t layer1_weights[16];      // 4x8 packed = 16 bytes
        float layer1_scales[8];          // 8 floats
        float layer1_biases[8];          // 8 floats
        uint8_t layer2_weights[16];      // 8x4 packed = 16 bytes
        float layer2_scales[4];          // 4 floats
        float layer2_biases[4];          // 4 floats
    };
    
    const auto* w_layout = reinterpret_cast<const NpuWeightsLayout*>(packed_weights);
    
    // Layer 1 forward: [4 inputs -> 8 hidden neurons]
    float h1[8] = {0.0f};
    rvv_gemv_int4(
        w_layout->layer1_weights,
        npu_inputs,
        h1,
        4,
        8,
        w_layout->layer1_scales,
        w_layout->layer1_biases
    );
    
    // Activation: Tanh
    float h1_activated[8] = {0.0f};
    for (int k = 0; k < 8; ++k) {
        h1_activated[k] = std::tanh(h1[k]);
    }
    
    // Layer 2 forward: [8 hidden neurons -> 4 outputs]
    float out_val[4] = {0.0f};
    rvv_gemv_int4(
        w_layout->layer2_weights,
        h1_activated,
        out_val,
        8,
        4,
        w_layout->layer2_scales,
        w_layout->layer2_biases
    );
    
    // Write predictions to outputs
    npu_outputs[0] = out_val[0];
    npu_outputs[1] = out_val[1];
    npu_outputs[2] = out_val[2]; // Recurrent state 0
    npu_outputs[3] = out_val[3]; // Recurrent state 1
    
    NPU->CTRL &= ~NPU_CTRL_START;
    NPU->STATUS &= ~NPU_STATUS_BUSY;
    NPU->STATUS |= NPU_STATUS_DONE;
}

/**
 * @brief Main 500 Hz balancing control step.
 * @details Reads raw sensor registers, filters pitch rates, updates PID,
 *          applies dynamic joint torque, and streams outputs to custom NPU layers.
 */
extern "C" void run_balancing_step() {
    // 1. Read Raw registers
    int16_t raw_gyro_y = IMU->GYRO_Y;
    int16_t raw_gyro_x = IMU->GYRO_X;
    int16_t raw_accel_z = IMU->ACCEL_Z;
    int16_t raw_accel_x = IMU->ACCEL_X;

    // 2. Apply filtering and conversions
    float pitch_rate = pitch_gyro_filter.apply(static_cast<float>(raw_gyro_y) * RAW_GYRO_TO_RADS);
    float roll_rate = roll_gyro_filter.apply(static_cast<float>(raw_gyro_x) * RAW_GYRO_TO_RADS);

    // Compute estimate of tilt from accelerometers
    float ax = static_cast<float>(raw_accel_x) * RAW_ACCEL_TO_MPS2;
    float az = static_cast<float>(raw_accel_z) * RAW_ACCEL_TO_MPS2;
    float estimated_tilt = std::atan2(ax, std::max(0.01f, az));

    // 3. Compute balancing command torque via PID
    float target_tilt = 0.0f; // Perfect alignment
    float feedback_torque = balance_pid.calculate(target_tilt, estimated_tilt);

    // 4. Distribute torque to physical joint registers
    int16_t ankle_torque = convert_torque_to_mNm(feedback_torque);
    int16_t knee_torque = convert_torque_to_mNm(feedback_torque * 0.65f); // Scale knee force

    ACTUATOR[0].TARGET_TORQUE = ankle_torque;       // Right Ankle
    ACTUATOR[1].TARGET_TORQUE = -ankle_torque;      // Left Ankle (Inverted orientation)
    ACTUATOR[2].TARGET_TORQUE = knee_torque;        // Right Knee
    ACTUATOR[3].TARGET_TORQUE = -knee_torque;       // Left Knee (Inverted orientation)

    // 5. Transfer states to Cognitive NPU accelerator registers for trajectory forecasting
    auto* npu_inputs = reinterpret_cast<float*>(NPU->INPUT_ADDR);
    if (npu_inputs != nullptr) {
        npu_inputs[0] = estimated_tilt;
        npu_inputs[1] = pitch_rate;
        npu_inputs[2] = roll_rate;
        npu_inputs[3] = feedback_torque;

        NPU->CTRL |= NPU_CTRL_START; // Trigger async inference
        trigger_npu_hardware_inference(); // Run synchronous step simulation
    }
}
