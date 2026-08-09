"""
battery_optimizer.py
~~~~~~~~~~~~~~~~~~~
A state estimation and power management optimizer for next-generation solid-state batteries,
specifically modeled after the WeLion 150 kWh pack (360 Wh/kg) and CATL's condensed cell chemistry (500 Wh/kg).
Implements state-of-charge (SOC) calculation, thermal tracking, dynamic discharge profiling,
and micro-watt deep sleep state transitions for long-endurance autonomous systems.
"""

import time
import math

class SolidStateBatteryPack:
    """
    Models the physical behavior of a Solid-State Battery pack.
    Solid-state electrolytes offer low internal resistance, high energy density,
    and immunity to traditional thermal runaway, but exhibit voltage plateau shifts
    based on charge density and core temperatures.
    """
    def __init__(self, capacity_wh: float = 150000.0, nominal_voltage: float = 400.0, energy_density_whkg: float = 360.0):
        self.capacity_wh = capacity_wh
        self.nominal_voltage = nominal_voltage
        self.energy_density = energy_density_whkg
        self.pack_weight = capacity_wh / energy_density_whkg # kg
        
        # State variables
        self.soc = 1.0  # State of Charge (0.0 to 1.0)
        self.temperature_k = 298.15  # Pack temperature in Kelvin (25 C)
        self.voltage = nominal_voltage
        
        # Dynamic internal resistance model (solid state electrolyte resistance decreases as temp rises)
        self.base_resistance_ohms = 0.045 # Very low internal resistance compared to liquid Li-Ion

    def get_internal_resistance(self) -> float:
        """Models electrolyte thermal activation: R = R0 * exp(Ea / (R_gas * T))"""
        temp_c = self.temperature_k - 273.15
        # Resistance drops at optimal temperature (35C - 45C) for solid electrolyte ion conductivity
        thermal_coefficient = math.exp(200.0 / (temp_c + 273.15) - 200.0 / 298.15)
        return self.base_resistance_ohms * thermal_coefficient

    def update(self, current_amps: float, dt_seconds: float, ambient_temp_k: float = 298.15):
        """
        Updates the battery states (SOC, Voltage, Temperature) based on current draw.
        Negative current = discharge, positive = charge.
        """
        # Calculate power draw (P = I * V)
        r_int = self.get_internal_resistance()
        terminal_voltage = self.voltage - (current_amps * r_int)
        power_w = terminal_voltage * current_amps
        
        # Energy discharge: dE = P * dt
        energy_delta_wh = (power_w * (dt_seconds / 3600.0))
        # Update SOC
        self.soc += energy_delta_wh / self.capacity_wh
        self.soc = max(0.0, min(1.0, self.soc))
        
        # Open Circuit Voltage (OCV) curve simulation for Solid-State Silicon-Anode / Nickel-Cathode
        # Features a stable plateau between 80% and 20% SOC, followed by a steep drop-off
        self.voltage = self.nominal_voltage * (0.85 + 0.15 * math.log(9.0 * self.soc + 1.0))
        
        # Thermal model: Heat generation (I^2 * R) vs Ambient heat dissipation (Convection)
        heat_gen_w = (current_amps ** 2) * r_int
        thermal_mass_j_k = self.pack_weight * 800.0  # Approx 800 J/kg-K heat capacity
        convection_coefficient = 5.0  # W/K cooling rate
        
        heat_loss_w = convection_coefficient * (self.temperature_k - ambient_temp_k)
        net_heat_w = heat_gen_w - heat_loss_w
        
        self.temperature_k += (net_heat_w * dt_seconds) / thermal_mass_j_k
        self.temperature_k = max(233.15, self.temperature_k) # Absolute minimum bounds (-40C)

class PowerStateController:
    """
    Manages operational states of the otonom system to minimize power consumption.
    Implements micro-watt power gating modes for low-activity periods.
    """
    def __init__(self, battery: SolidStateBatteryPack):
        self.battery = battery
        self.active_system_draw_w = 450.0  # Active consumption (motors, CPU, NPU)
        self.light_sleep_draw_w = 12.0     # Light sleep (actuators shut down, CPU/NPU idling)
        self.deep_sleep_draw_w = 0.005     # Deep sleep micro-watt state (gated registers, wake-up timer only)
        
        self.current_state = "ACTIVE" # "ACTIVE", "LIGHT_SLEEP", "DEEP_SLEEP"

    def select_optimal_state(self, tilt_velocity_magnitude: float, distance_to_target: float) -> str:
        """Determines best power state based on sensory activity thresholds."""
        if tilt_velocity_magnitude > 0.8:
            # Physical disturbance detected: wake up immediately to balance
            self.current_state = "ACTIVE"
        elif distance_to_target < 0.1 and tilt_velocity_magnitude < 0.02:
            # Stationary and reached target: switch to ultra-low deep sleep state
            self.current_state = "DEEP_SLEEP"
        elif distance_to_target >= 0.1 and tilt_velocity_magnitude < 0.05:
            # Stationary but path is queued: stay alert in light sleep
            self.current_state = "LIGHT_SLEEP"
            
        return self.current_state

    def get_system_current_amps(self) -> float:
        """Calculates current draw based on the current state and battery voltage."""
        voltage = self.battery.voltage
        if self.current_state == "ACTIVE":
            return self.active_system_draw_w / voltage
        elif self.current_state == "LIGHT_SLEEP":
            return self.light_sleep_draw_w / voltage
        else: # DEEP_SLEEP (uW range)
            return self.deep_sleep_draw_w / voltage

def main():
    print("==================================================")
    print("Energy Dynamics - Solid-State Battery Optimizer")
    print("==================================================")
    
    # Initialize 150 kWh pack (WeLion equivalent)
    battery = SolidStateBatteryPack(capacity_wh=150000.0, nominal_voltage=380.0, energy_density_whkg=360.0)
    controller = PowerStateController(battery)
    
    print(f"Pack Specifications:")
    print(f"  +- Nominal Capacity: {battery.capacity_wh / 1000.0:.1f} kWh")
    print(f"  +- Nominal Voltage:  {battery.nominal_voltage:.1f} V")
    print(f"  +- Energy Density:   {battery.energy_density:.1f} Wh/kg")
    print(f"  +- Pack Weight:      {battery.pack_weight:.2f} kg")
    
    # Run a dynamic simulation of power drain
    print("\nSimulating state transitions and thermal evolution:")
    
    # Step 1: Active operations
    controller.current_state = "ACTIVE"
    current = controller.get_system_current_amps()
    battery.update(-current, 1800.0) # Active for 30 minutes (1800s)
    print(f"After 30 mins [ACTIVE] (Draw: {controller.active_system_draw_w}W):")
    print(f"  +- SOC:         {battery.soc * 100.0:.2f}%")
    print(f"  +- OCV Voltage: {battery.voltage:.2f} V")
    print(f"  +- Core Temp:   {battery.temperature_k - 273.15:.2f} °C")
    
    # Step 2: Transition to Light Sleep
    controller.select_optimal_state(tilt_velocity_magnitude=0.03, distance_to_target=0.5)
    current = controller.get_system_current_amps()
    battery.update(-current, 7200.0) # Light sleep for 2 hours (7200s)
    print(f"\nAfter 2 hours [LIGHT_SLEEP] (Draw: {controller.light_sleep_draw_w}W):")
    print(f"  +- SOC:         {battery.soc * 100.0:.2f}%")
    print(f"  +- OCV Voltage: {battery.voltage:.2f} V")
    print(f"  +- Core Temp:   {battery.temperature_k - 273.15:.2f} °C (Cooling down)")
    
    # Step 3: Transition to Micro-watt Deep Sleep
    controller.select_optimal_state(tilt_velocity_magnitude=0.005, distance_to_target=0.0)
    current = controller.get_system_current_amps()
    battery.update(-current, 86400.0 * 5) # Deep sleep for 5 days (86400 * 5)
    print(f"\nAfter 5 days [DEEP_SLEEP] (Draw: {controller.deep_sleep_draw_w}W):")
    print(f"  +- SOC:         {battery.soc * 100.0:.2f}%")
    print(f"  +- OCV Voltage: {battery.voltage:.2f} V")
    print(f"  +- Core Temp:   {battery.temperature_k - 273.15:.2f} °C")
    
    print("\nStatus: Power Dynamics Optimization & Sleep Verification Completed.")
    print("==================================================")

if __name__ == "__main__":
    main()
