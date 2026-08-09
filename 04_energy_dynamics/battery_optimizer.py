"""
battery_optimizer.py
~~~~~~~~~~~~~~~~~~~
A state estimation and power management optimizer for solid-state batteries,
specifically modeled after the WeLion 150 kWh pack (360 Wh/kg) and CATL's 500 Wh/kg cells.
Implements State of Charge (SOC), State of Health (SOH) degradation modeling, 
temperature-based current limiting, and dynamic CPU/motor scaling.
"""

import time
import math

class SolidStateBatteryPack:
    """
    Models the physical behavior of a Solid-State Battery pack.
    Tracks state of charge, temperature, internal resistance, and capacity fade (SOH).
    """
    def __init__(self, capacity_wh: float = 150000.0, nominal_voltage: float = 400.0, energy_density_whkg: float = 360.0):
        self.capacity_wh = capacity_wh
        self.nominal_capacity = capacity_wh # Initial factory capacity
        self.nominal_voltage = nominal_voltage
        self.energy_density = energy_density_whkg
        self.pack_weight = capacity_wh / energy_density_whkg
        
        # State variables
        self.soc = 1.0  # State of Charge (0.0 to 1.0)
        self.soh = 1.0  # State of Health (1.0 = 100% capacity retention)
        self.temperature_k = 298.15  # Pack temperature in Kelvin (25 C)
        self.voltage = nominal_voltage
        
        # Degradation tracking
        self.total_cycle_count = 0.0
        self.base_resistance_ohms = 0.045

    def get_internal_resistance(self) -> float:
        """Models electrolyte thermal activation and degradation increase: R = R0 * (1/SOH) * thermal_coeff"""
        temp_c = self.temperature_k - 273.15
        thermal_coefficient = math.exp(200.0 / (temp_c + 273.15) - 200.0 / 298.15)
        # Resistance increases as battery health (SOH) degrades
        degradation_factor = 1.0 / max(0.5, self.soh)
        return self.base_resistance_ohms * thermal_coefficient * degradation_factor

    def update(self, current_amps: float, dt_seconds: float, ambient_temp_k: float = 298.15):
        """Updates battery state variables and applies electrochemical degradation rules."""
        r_int = self.get_internal_resistance()
        
        # Terminal Voltage = OCV - I * R_int
        terminal_voltage = self.voltage - (current_amps * r_int)
        power_w = terminal_voltage * current_amps
        
        # Compute energy delta (negative for discharge, positive for charge)
        energy_delta_wh = (power_w * (dt_seconds / 3600.0))
        
        # Track cycle equivalent (one full discharge/charge equal to nominal capacity)
        self.total_cycle_count += abs(energy_delta_wh) / (2.0 * self.nominal_capacity)
        
        # Update SOH based on cycle fatigue and temperature exposure
        # Solid-state cells degrade faster if run hot (e.g. > 50C)
        temp_c = self.temperature_k - 273.15
        temp_degrade_factor = 1.0 + (max(0.0, temp_c - 40.0) * 0.05)
        cycle_degradation = 0.00005 * temp_degrade_factor * (abs(energy_delta_wh) / self.nominal_capacity)
        self.soh = max(0.0, self.soh - cycle_degradation)
        
        # Update SOC based on current actual degraded capacity
        actual_capacity = self.nominal_capacity * self.soh
        self.soc += energy_delta_wh / actual_capacity
        self.soc = max(0.0, min(1.0, self.soc))
        
        # OCV Curve modeling
        self.voltage = self.nominal_voltage * (0.85 + 0.15 * math.log(9.0 * self.soc + 1.0))
        
        # Heat generation (Joule heating + entropy reaction heat) vs cooling
        heat_gen_w = (current_amps ** 2) * r_int
        thermal_mass_j_k = self.pack_weight * 800.0
        convection_coefficient = 5.0
        
        heat_loss_w = convection_coefficient * (self.temperature_k - ambient_temp_k)
        self.temperature_k += ((heat_gen_w - heat_loss_w) * dt_seconds) / thermal_mass_j_k
        self.temperature_k = max(233.15, self.temperature_k)

class PowerStateController:
    """
    Manages processor clock frequencies, motor speed bounds, and sensor limits
    based on battery safety zones and state-of-health diagnostics.
    """
    def __init__(self, battery: SolidStateBatteryPack):
        self.battery = battery
        self.system_state = "ACTIVE" # ACTIVE, LIGHT_SLEEP, DEEP_SLEEP
        
        # Power profiles (Watts)
        self.cpu_clock_mhz = 240.0 # Standard clock
        self.max_motor_power_w = 400.0
        
    def adjust_system_throttling(self) -> float:
        """
        Dynamically adjusts CPU frequency and motor current caps.
        Returns safety-limited system current draw (Amps).
        """
        temp_c = self.battery.temperature_k - 273.15
        soc = self.battery.soc
        
        # 1. Thermal protection throttling: Limit motor load if battery heats up
        if temp_c > 50.0:
            # Overheating warning: throttle motors to 40% power
            self.max_motor_power_w = 150.0
            self.cpu_clock_mhz = 120.0
        elif temp_c > 42.0:
            # Warm state: throttle to 75% power
            self.max_motor_power_w = 300.0
            self.cpu_clock_mhz = 180.0
        else:
            self.max_motor_power_w = 400.0
            self.cpu_clock_mhz = 240.0
            
        # 2. Voltage collapse protection: Throttle when low SOC is detected
        if soc < 0.15:
            # Low voltage emergency mode
            self.max_motor_power_w = 80.0
            self.cpu_clock_mhz = 48.0 # Power saving state
            self.system_state = "LIGHT_SLEEP"
            
        # Compute combined load
        if self.system_state == "ACTIVE":
            total_load_w = self.max_motor_power_w + (self.cpu_clock_mhz * 0.2) # ~48W CPU at max clock
        elif self.system_state == "LIGHT_SLEEP":
            total_load_w = 12.0 # Actuators off
        else: # DEEP_SLEEP
            total_load_w = 0.005 # Gated logic
            
        # Current = P / V
        return total_load_w / self.battery.voltage

def main():
    print("==================================================")
    print("Energy Dynamics - Advanced Battery Safety Simulator")
    print("==================================================")
    
    battery = SolidStateBatteryPack(capacity_wh=150000.0, nominal_voltage=380.0, energy_density_whkg=360.0)
    controller = PowerStateController(battery)
    
    print(f"Initial Health: {battery.soh * 100.0:.2f}% SOH | Capacity: {battery.capacity_wh/1000.0:.1f} kWh")
    
    # Simulate an extreme discharge load (e.g. running motors at full power for 2 hours in warm ambient)
    print("\nSimulating 2-hour heavy duty run under 40 C ambient:")
    ambient_temp = 273.15 + 40.0 # 40C ambient
    
    # 2 hours split into 10-minute intervals
    for interval in range(1, 13):
        system_current = controller.adjust_system_throttling()
        # Discharge battery
        battery.update(-system_current, 600.0, ambient_temp_k=ambient_temp)
        
        temp_c = battery.temperature_k - 273.15
        print(f"Min {interval*10:03d}: SOC: {battery.soc*100.0:5.2f}% | SOH: {battery.soh*100.0:8.5f}% | Temp: {temp_c:.1f}C | Clock: {controller.cpu_clock_mhz:.0f}MHz | Motor Limit: {controller.max_motor_power_w:.0f}W")
        
    print("\nSimulation Completed.")
    print(f"Final State of Charge: {battery.soc * 100.0:.2f}%")
    print(f"Final State of Health: {battery.soh * 100.0:.5f}% (Fatigue degradation applied)")
    print(f"Total Equivalent Cycles: {battery.total_cycle_count:.6f}")
    print("==================================================")

if __name__ == "__main__":
    main()
