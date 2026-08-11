"""
unified_system_orchestrator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A master real-time simulation orchestrator demonstrating the symbiotic relationship
between the bare-metal control systems, edge AI recurrent model, solid-state battery
power dynamics, and decentralized 3D eVTOL swarm coordination.
"""

import sys
import os
import math
import time
import random

# Ensure modules in child folders can be imported directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "02_cognitive_edge")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "03_swarm_protocols")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "04_energy_dynamics")))

try:
    from inference_engine import RecurrentCognitiveEdgeAI
    from evtol_traffic_ctrl import eVTOLNode, OBSTACLES_3D
    from battery_optimizer import SolidStateBatteryPack, PowerStateController
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ANSI terminal colors for premium CLI diagnostics
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
MAGENTA = "\033[1;35m"
RESET = "\033[0m"
BOLD = "\033[1m"

class SimulatedLowPassFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.prev_val = 0.0

    def apply(self, val: float) -> float:
        output = (self.alpha * val) + ((1.0 - self.alpha) * self.prev_val)
        self.prev_val = output
        return output

class SimulatedPidController:
    def __init__(self, kp: float, ki: float, kd: float, limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.02 # 50 Hz macro step for integration

    def calculate(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement
        p_out = self.kp * error
        self.integral += error * self.dt
        
        # Anti-windup
        i_out = self.ki * self.integral
        if abs(i_out) > self.limit:
            i_out = math.copysign(self.limit, i_out)
            self.integral = i_out / self.ki if self.ki != 0 else 0
            
        d_out = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        total = p_out + i_out + d_out
        return max(-self.limit, min(self.limit, total))

def draw_progress_bar(val: float, width: int = 15) -> str:
    """Returns a visual progress bar string for battery level/SOC."""
    filled = int(round(val * width))
    bar = "#" * filled + "-" * (width - filled)
    color = GREEN if val > 0.5 else (YELLOW if val > 0.2 else RED)
    return f"{color}[{bar}] {val*100.0:.1f}%{RESET}"

def run_symbiotic_orchestrator():
    print(f"\n{BOLD}{CYAN}==========================================================================")
    print("      YARININ YARISLARI - UNIFIED DEEP-TECH SYMBIOSIS SIMULATOR")
    print(f"=========================================================================={RESET}")
    
    # 1. Initialize Solid-State Battery & Power controller
    # Models WeLion 150 kWh pack (360 Wh/kg) discharging under active stabilization loads
    battery = SolidStateBatteryPack(capacity_wh=150000.0, nominal_voltage=380.0, energy_density_whkg=360.0)
    power_ctrl = PowerStateController(battery)
    
    # Let's seed battery warm and with high ambient temp to trigger throttling quickly
    battery.temperature_k = 273.15 + 41.5 # 41.5 C internal temperature
    ambient_temp = 273.15 + 38.0 # 38.0 C hot desert drone deck ambient
    
    # 2. Initialize Cognitive Recurrent Edge AI
    weights_path = os.path.join(os.path.dirname(__file__), "02_cognitive_edge", "weights_config.json")
    try:
        npu_engine = RecurrentCognitiveEdgeAI(weights_path)
        print(f"  +- {GREEN}SUCCESS{RESET}: Cognitive recurrent edge AI engine online (INT4 packed).")
    except Exception as e:
        print(f"  +- {RED}ERROR{RESET}: Failed to load cognitive weights: {e}")
        sys.exit(1)
        
    # 3. Initialize 3D Airspace eVTOL Nodes
    fleet = [eVTOLNode(node_id=301), eVTOLNode(node_id=302), eVTOLNode(node_id=303)]
    print(f"  +- {GREEN}SUCCESS{RESET}: 3D eVTOL flight control swarm online ({len(fleet)} agents).")
    
    # 4. Initialize Bare-metal feedback filters and controllers
    pitch_filter = SimulatedLowPassFilter(0.25)
    roll_filter = SimulatedLowPassFilter(0.25)
    stabilizer_pid = SimulatedPidController(1.5, 0.2, 0.18, 60.0) # 60 N-m torque limit
    
    time.sleep(1.0)
    
    print(f"\n{BOLD}Starting real-time simulation tracking...{RESET}\n")
    
    # Run simulation for 20 epochs
    for epoch in range(1, 21):
        # A. Power Dynamics & Throttling
        system_current = power_ctrl.adjust_system_throttling()
        # Heavy stabilization and motor control cycles drain battery
        # We simulate a current load drawn from the battery pack
        battery.update(-system_current, 1.0, ambient_temp_k=ambient_temp)
        
        # Limit control torque limits based on power throttling
        dynamic_torque_limit = stabilizer_pid.limit
        if power_ctrl.max_motor_power_w < 400.0:
            # Power throttled: limit humanoid stabilizer peak forces
            stabilizer_pid.limit = 60.0 * (power_ctrl.max_motor_power_w / 400.0)
        else:
            stabilizer_pid.limit = 60.0
            
        # B. Humanoid Balance Loop (simulated C++ registers logic)
        # Simulate platform tilt oscillation (pitch) with random wind gust perturbations
        t = epoch * 0.2
        raw_pitch_rate = 0.5 * math.sin(t) + random.uniform(-0.15, 0.15)
        raw_roll_rate = 0.3 * math.cos(t) + random.uniform(-0.1, 0.1)
        
        filtered_pitch_rate = pitch_filter.apply(raw_pitch_rate)
        filtered_roll_rate = roll_filter.apply(raw_roll_rate)
        
        # Calculate estimated tilt based on raw sensor fusion
        estimated_tilt = raw_pitch_rate * 0.85 + random.uniform(-0.02, 0.02)
        
        # PID torque output calculation
        target_tilt = 0.0
        stabilization_torque = stabilizer_pid.calculate(target_tilt, estimated_tilt)
        
        # C. Cognitive AI recurrent forecasting step
        # Neural network concatenated inputs: [tilt, pitch_rate, roll_rate, torque] + past hidden state
        npu_inputs = [estimated_tilt, filtered_pitch_rate, filtered_roll_rate, stabilization_torque]
        
        start_time = time.perf_counter()
        npu_offsets = npu_engine.run_inference(npu_inputs)
        end_time = time.perf_counter()
        npu_latency_us = (end_time - start_time) * 1e6
        
        # D. eVTOL Fleet Trajectory calculations
        for node in fleet:
            node.update_physics()
            
        # Simulated multi-path communications exchange
        for sender in fleet:
            state = sender.get_state()
            for receiver in fleet:
                if receiver.node_id != sender.node_id:
                    receiver.peers[sender.node_id] = state
                    
        # E. Print Console Dashboard
        print(f"{BOLD}{MAGENTA}--- TIME STEP TICK: {epoch:02d} ---{RESET}")
        
        # 1. Battery & Power state
        temp_c = battery.temperature_k - 273.15
        soc_bar = draw_progress_bar(battery.soc)
        status_flag = f"{RED}THROTTLED{RESET}" if power_ctrl.cpu_clock_mhz < 240.0 else f"{GREEN}OPTIMAL{RESET}"
        print(f"{BOLD}[BATTERY & THERMALS]{RESET}")
        print(f"  SOC: {soc_bar} | SOH: {battery.soh*100.0:8.5f}% | Temp: {temp_c:.1f} C ({status_flag})")
        print(f"  Voltage: {battery.voltage:.1f} V | CPU Clock: {power_ctrl.cpu_clock_mhz:.0f} MHz | Max Motor Pwr: {power_ctrl.max_motor_power_w:.0f} W")
        
        # 2. Humanoid control state (C++ & NPU coprocessor output)
        print(f"{BOLD}[HUMANOID STABILIZATION]{RESET}")
        print(f"  Est. Tilt: {estimated_tilt:+.4f} rad | Pitch Rate: {filtered_pitch_rate:+.4f} rad/s | Roll Rate: {filtered_roll_rate:+.4f} rad/s")
        print(f"  Stabilization PID Torque: {stabilization_torque:+.2f} N-m (Limit: {stabilizer_pid.limit:.1f} N-m)")
        print(f"  NPU Joint Predictions: Ankle={npu_offsets[0]:+.4f} N-m, Knee={npu_offsets[1]:+.4f} N-m | Latency: {npu_latency_us:.1f} us")
        
        # 3. eVTOL Swarm Status
        print(f"{BOLD}[3D eVTOL SWARM TELEMETRY]{RESET}")
        for node in fleet:
            state = node.get_state()
            # Check warning
            min_clearance = float('inf')
            closest_obs = ""
            for obs in OBSTACLES_3D:
                if state.z >= obs["z_min"] and state.z <= obs["z_max"]:
                    dist = math.hypot(state.x - obs["x"], state.y - obs["y"])
                    clear = dist - obs["radius"]
                    if clear < min_clearance:
                        min_clearance = clear
                        closest_obs = obs["name"]
            alert_str = f"{RED}[ALERT] NEIGHBOR/OBSTACLE CLEARANCE: {min_clearance:.1f}m{RESET}" if min_clearance < 10.0 else f"{GREEN}AIRSPACE CLEAR{RESET}"
            print(f"  Node {state.id} [{state.role:<8}] Pos: ({state.x:+.1f}, {state.y:+.1f}, {state.z:+.1f}) | Bat: {state.battery:.1f}% | Tracking: {state.tracking_source:<22} | {alert_str}")
            
        print("-" * 74)
        time.sleep(0.3) # Fast-forward simulation speed
        
    print(f"\n{BOLD}{GREEN}==========================================================================")
    print("          SYMBIOTIC REAL-TIME SIMULATION COMPLETED SUCCESSFULLY")
    print(f"=========================================================================={RESET}\n")

if __name__ == "__main__":
    run_symbiotic_orchestrator()
