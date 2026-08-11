/**
 * @file bsp_register_map.h
 * @brief Board Support Package (BSP) Register Definitions for Neuromorphic Edge Core.
 * @details This file maps the hardware register space for the simulated neuromorphic
 *          microprocessor (NPU-MCU hybrid). It handles register-level interfaces for
 *          limb torque actuators, raw IMU, solid-state battery sensor interfaces,
 *          and hardware-accelerated INT4 neural network weights.
 * 
 * Target Architecture: Custom RISC-V 64-bit Core with Neuromorphic Extension (RV64-NPU)
 */

#ifndef BSP_REGISTER_MAP_H
#define BSP_REGISTER_MAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HOST_SIMULATION
extern uint8_t simulated_npu_mem[];
extern uint8_t simulated_actuator_mem[];
extern uint8_t simulated_power_sys_mem[];
extern uint8_t simulated_imu_mem[];
#endif

// Base addresses for peripherals
#define NPU_BASE_ADDR            0x40080000U // Neural Processing Unit Control
#define ACTUATOR_BASE_ADDR       0x40090000U // Joint Actuator Controller (PWM/FOC)
#define POWER_SYS_BASE_ADDR      0x400A0000U // Solid-State Battery & Power Management
#define IMU_BASE_ADDR            0x400B0000U // 6-Axis Inertial Measurement Unit

/**
 * @brief NPU Control & Inference Registers
 */
typedef struct {
    volatile uint32_t CTRL;         /**< 0x00: NPU Control Register (Run, Stop, Reset) */
    volatile uint32_t STATUS;       /**< 0x04: Inference Status (Busy, Ready, Error) */
    volatile uint32_t WEIGHT_ADDR;   /**< 0x08: Start pointer of quantized weights (INT4/INT2) */
    volatile uint32_t INPUT_ADDR;    /**< 0x0C: Start pointer of sensory input vectors */
    volatile uint32_t OUTPUT_ADDR;   /**< 0x10: Start pointer of activation output vectors */
    volatile uint32_t LAYER_COUNT;   /**< 0x14: Number of active network layers */
    volatile uint32_t NEURON_SCALE;  /**< 0x18: Dynamic scaling factor for low-bit quantization */
    volatile uint32_t IRQ_EN;        /**< 0x1C: Interrupt Enable Register */
} NPU_TypeDef;

#ifdef HOST_SIMULATION
#define NPU ((NPU_TypeDef *) simulated_npu_mem)
#else
#define NPU ((NPU_TypeDef *) NPU_BASE_ADDR)
#endif

// NPU CTRL register masks
#define NPU_CTRL_START          (1U << 0)
#define NPU_CTRL_RESET          (1U << 1)
#define NPU_CTRL_INT_EN         (1U << 2)

// NPU STATUS register masks
#define NPU_STATUS_BUSY         (1U << 0)
#define NPU_STATUS_DONE         (1U << 1)
#define NPU_STATUS_ERR          (1U << 2)

/**
 * @brief Actuator Controller Registers (Field Oriented Control - FOC interface)
 */
typedef struct {
    volatile uint32_t MODE;         /**< 0x00: Control Mode (0=Idle, 1=Torque, 2=Position) */
    volatile int32_t  TARGET_POS;   /**< 0x04: Target angle in microradians */
    volatile int32_t  ACTUAL_POS;   /**< 0x08: Actual angle in microradians */
    volatile int16_t  TARGET_TORQUE;/**< 0x0C: Target torque in millinewton-meters */
    volatile int16_t  ACTUAL_TORQUE;/**< 0x0E: Actual measured torque */
    volatile uint32_t TEMPERATURE;  /**< 0x10: Actuator winding temperature in mC */
    volatile uint32_t FAULT_STATUS; /**< 0x14: Actuator fault flags (Over-current, thermal, etc.) */
} Actuator_TypeDef;

// 12-DOF System support (e.g. 12 Joint Actuators for basic humanoid limbs)
#define ACTUATOR_COUNT 12

#ifdef HOST_SIMULATION
#define ACTUATOR ((volatile Actuator_TypeDef *) simulated_actuator_mem)
#else
#define ACTUATOR ((volatile Actuator_TypeDef *) ACTUATOR_BASE_ADDR)
#endif

/**
 * @brief Power & Energy Dynamics Interface (optimized for Solid-State Battery management)
 */
typedef struct {
    volatile uint32_t BAT_SOC;      /**< 0x00: State of Charge (0 - 10000 represent 0.00% to 100.00%) */
    volatile uint32_t BAT_VOLTAGE;  /**< 0x04: Cell package voltage in millivolts */
    volatile int32_t  BAT_CURRENT;  /**< 0x08: Package current in microamps (positive=charging) */
    volatile uint32_t BAT_TEMP;     /**< 0x0C: Solid-state electrolyte temperature in millikelvin */
    volatile uint32_t SYS_PWR_STATE;/**< 0x10: 0=Active, 1=Light Sleep, 2=Deep Sleep (uW), 3=Hibernation */
    volatile uint32_t SLEEP_TIMER;  /**< 0x14: Deep sleep wake up timer ticks (microseconds) */
} PowerSys_TypeDef;

#ifdef HOST_SIMULATION
#define POWER_SYS ((PowerSys_TypeDef *) simulated_power_sys_mem)
#else
#define POWER_SYS ((PowerSys_TypeDef *) POWER_SYS_BASE_ADDR)
#endif

/**
 * @brief IMU Register Interface for balancing loops
 */
typedef struct {
    volatile int16_t ACCEL_X;       /**< 0x00: Accel X Raw */
    volatile int16_t ACCEL_Y;       /**< 0x02: Accel Y Raw */
    volatile int16_t ACCEL_Z;       /**< 0x04: Accel Z Raw */
    volatile int16_t GYRO_X;        /**< 0x06: Gyro X Raw */
    volatile int16_t GYRO_Y;        /**< 0x08: Gyro Y Raw */
    volatile int16_t GYRO_Z;        /**< 0x0A: Gyro Z Raw */
    volatile uint32_t STATUS;       /**< 0x0C: Sensor data ready status */
} IMU_TypeDef;

#ifdef HOST_SIMULATION
#define IMU ((IMU_TypeDef *) simulated_imu_mem)
#else
#define IMU ((IMU_TypeDef *) IMU_BASE_ADDR)
#endif

#ifdef __cplusplus
}
#endif

#endif // BSP_REGISTER_MAP_H
