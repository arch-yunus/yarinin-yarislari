/**
 * @file neuromorphic_control.cpp
 * @brief Bare-metal Neuromorphic Feedback Control Loop for Humanoid balancing.
 * @details Implements a real-time control loop mapping IMU tilt rates to joint torques
 *          by directly reading and writing to simulated hardware registers defined in
 *          bsp_register_map.h. Designed to compile in C++23.
 */

#include "bsp_register_map.h"
#include <array>
#include <concepts>
#include <algorithm>

// Define constants for stability
constexpr float GYRO_SCALE = 0.0001f;     // Scale raw sensor value to rad/s
constexpr float DT = 0.002f;              // 500 Hz control loop
constexpr float KP = 1.25f;               // Proportional gain for torque balancing
constexpr float KD = 0.15f;               // Derivative gain for torque balancing

// Emulate hardware register storage in local memory since we are in a simulation environment.
// On physical hardware, these variables are linked directly to base peripheral addresses.
#ifdef SIMULATION_ENVIRONMENT
static uint8_t mock_peripheral_space[0x100000];
#endif

// Utility to convert physical float torque back to integer register scale (millinewton-meters)
inline int16_t float_to_mNm(float torque) {
    // Clamping to avoid motor driver overload
    float clamped = std::clamp(torque, -50.0f, 50.0f);
    return static_cast<int16_t>(clamped * 1000.0f);
}

/**
 * @brief Initialize all actuators to torque control mode
 */
extern "C" void init_system() {
    // Reset NPU
    NPU->CTRL |= NPU_CTRL_RESET;
    for (volatile int i = 0; i < 100; ++i); // Small hardware spin delay
    NPU->CTRL &= ~NPU_CTRL_RESET;

    // Enable NPU Done interrupt
    NPU->IRQ_EN |= 1;

    // Put all 12 Joint Actuators in Torque Mode (Mode 1)
    for (int i = 0; i < ACTUATOR_COUNT; ++i) {
        // Pointer arithmetic on volatile structure array base
        volatile Actuator_TypeDef* joint = &ACTUATOR[i];
        joint->MODE = 1;
        joint->TARGET_TORQUE = 0;
    }
}

/**
 * @brief Run a single iteration of the stabilization loop (called at 500 Hz via timer IRQ)
 */
extern "C" void run_balancing_step() {
    // 1. Read IMU Pitch Angular Velocity (Gyro Y)
    int16_t raw_gyro_y = IMU->GYRO_Y;
    int16_t raw_accel_z = IMU->ACCEL_Z;
    int16_t raw_accel_x = IMU->ACCEL_X;

    // Parse sensor inputs to physical float equivalents
    float pitch_velocity = static_cast<float>(raw_gyro_y) * GYRO_SCALE;
    
    // Simple estimation of tilt angle from accel
    float tilt_angle = static_cast<float>(raw_accel_x) / (static_cast<float>(raw_accel_z) + 0.001f);

    // 2. PD Controller mapping tilt state to balancing torque
    // Target state: tilt = 0, pitch velocity = 0
    float error = 0.0f - tilt_angle;
    float error_dot = 0.0f - pitch_velocity;
    float control_torque = (KP * error) + (KD * error_dot);

    // 3. Command knee and ankle actuators to counteract tilt
    // Ankle joints: Actuators 0 and 1
    // Knee joints: Actuators 2 and 3
    int16_t torque_command = float_to_mNm(control_torque);
    
    ACTUATOR[0].TARGET_TORQUE = torque_command;       // Right Ankle Pitch
    ACTUATOR[1].TARGET_TORQUE = -torque_command;      // Left Ankle Pitch (inverted orientation)
    ACTUATOR[2].TARGET_TORQUE = torque_command * 0.7f; // Right Knee Pitch
    ACTUATOR[3].TARGET_TORQUE = -torque_command * 0.7f;// Left Knee Pitch (inverted orientation)

    // 4. Start NPU Inference for next-step trajectory prediction
    // Load input vectors (tilt, velocities) directly into NPU input memory area
    auto* input_buffer = reinterpret_cast<float*>(NPU->INPUT_ADDR);
    if (input_buffer != nullptr) {
        input_buffer[0] = tilt_angle;
        input_buffer[1] = pitch_velocity;
        
        // Signal NPU to begin hardware inference
        NPU->CTRL |= NPU_CTRL_START;
    }
}
