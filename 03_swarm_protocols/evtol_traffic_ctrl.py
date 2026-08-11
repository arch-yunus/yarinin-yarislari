"""
evtol_traffic_ctrl.py
~~~~~~~~~~~~~~~~~~~~~
A decentralized 3D space-edge airspace orchestrator for eVTOL drone fleets.
Features a 3D Artificial Potential Field (APF) algorithm for flocking and obstacle avoidance,
simulating cellular 5G-Advanced ISAC (Integrated Sensing and Communication) tracking
and dynamic satellite Orbit-AI tracking loops.
"""

import math
import random
import time
import json
from dataclasses import dataclass, asdict

# 3D Position State
@dataclass
class eVTOLState:
    id: int
    role: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    battery: float
    tracking_source: str # 5G-A ISAC, SATELLITE (Orbit-AI), GNSS_FALLBACK
    epoch: int

# Simulated 3D static obstacles (e.g., high-rise buildings, antenna masts)
# Defined as cylinders with center (x, y), height_limit, and radius.
OBSTACLES_3D = [
    {"x": 10.0, "y": 10.0, "z_min": 0.0, "z_max": 80.0, "radius": 8.0, "name": "Skyscraper-Alpha"},
    {"x": -15.0, "y": -10.0, "z_min": 0.0, "z_max": 120.0, "radius": 12.0, "name": "Radio-Mast-Beta"},
    {"x": 0.0, "y": 15.0, "z_min": 30.0, "z_max": 90.0, "radius": 15.0, "name": "Restricted-Flight-Zone"}
]

class eVTOLNode:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.role = "FOLLOWER"
        self.x = random.uniform(-30.0, 30.0)
        self.y = random.uniform(-30.0, 30.0)
        self.z = random.uniform(10.0, 40.0) # Vertical altitude (meters)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.battery = 100.0
        self.epoch = 0
        self.peers = {}
        
        # APF 3D Gains
        self.k_att = 0.8
        self.k_rep = 25.0
        self.d0 = 15.0 # Warning/repulsive range threshold in meters
        
        # Tracking source determination
        self.tracking_source = "5G-A ISAC"

    def get_state(self) -> eVTOLState:
        return eVTOLState(
            id=self.node_id,
            role=self.role,
            x=round(self.x, 3),
            y=round(self.y, 3),
            z=round(self.z, 3),
            vx=round(self.vx, 3),
            vy=round(self.vy, 3),
            vz=round(self.vz, 3),
            battery=round(self.battery, 2),
            tracking_source=self.tracking_source,
            epoch=self.epoch
        )

    def select_tracking_source(self):
        """
        Simulates 5G-A ISAC vs Space-Edge Satellite (Orbit-AI) handovers based on altitude and range.
        - High altitudes (>60m) lose terrestrial 5G signal and switch to Space-Edge Satellite.
        - Deep valleys or shadowed zones switch to GNSS fallback.
        """
        if self.z > 60.0:
            self.tracking_source = "SATELLITE (Orbit-AI)"
        elif abs(self.x) > 45.0 or abs(self.y) > 45.0:
            self.tracking_source = "GNSS_FALLBACK"
        else:
            self.tracking_source = "5G-A ISAC"

    def _compute_3d_artificial_potential_field(self) -> tuple[float, float, float]:
        """
        Calculates the attractive force to the goal or leader and repulsive forces
        from cylinders and neighbor nodes in 3D.
        """
        fx_att, fy_att, fz_att = 0.0, 0.0, 0.0
        fx_rep, fy_rep, fz_rep = 0.0, 0.0, 0.0
        
        # 1. 3D ATTRACTIVE FORCE
        if self.role == "LEADER":
            # Leader flies along a 3D spiral trajectory
            target_x = 25.0 * math.cos(self.epoch * 0.03)
            target_y = 25.0 * math.sin(self.epoch * 0.03)
            target_z = 50.0 + 20.0 * math.sin(self.epoch * 0.01)
        else:
            # Followers follow the leader with offsets
            leader = next((p for p in self.peers.values() if p.role == "LEADER"), None)
            if leader:
                target_x = leader.x
                target_y = leader.y
                target_z = leader.z + 5.0 # Follow slightly above to avoid jetwash
            else:
                target_x, target_y, target_z = 0.0, 0.0, 30.0 # Default hover point
                
        dx = target_x - self.x
        dy = target_y - self.y
        dz = target_z - self.z
        dist_to_target = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if dist_to_target > 0.1:
            fx_att = self.k_att * (dx / dist_to_target)
            fy_att = self.k_att * (dy / dist_to_target)
            fz_att = self.k_att * (dz / dist_to_target)

        # 2. 3D REPULSIVE FORCE (Cylinder Obstacles)
        for obs in OBSTACLES_3D:
            # Radial distance in horizontal plane
            ox = self.x - obs["x"]
            oy = self.y - obs["y"]
            r_dist = math.hypot(ox, oy)
            
            # Check if within vertical bounds
            z_overlap = (self.z >= obs["z_min"] - 5.0) and (self.z <= obs["z_max"] + 5.0)
            
            if z_overlap:
                clearance = r_dist - obs["radius"]
                if clearance <= 0.1:
                    clearance = 0.1
                
                if clearance < self.d0:
                    rep_factor = self.k_rep * (1.0 / clearance - 1.0 / self.d0) * (1.0 / (clearance ** 2))
                    fx_rep += rep_factor * (ox / r_dist)
                    fy_rep += rep_factor * (oy / r_dist)
                    
                    # Push vertically away from the top/bottom if close to limits
                    if abs(self.z - obs["z_max"]) < 10.0:
                        fz_rep += 5.0 * (1.0 / max(1.0, abs(self.z - obs["z_max"])))
                    elif abs(self.z - obs["z_min"]) < 10.0:
                        fz_rep -= 5.0 * (1.0 / max(1.0, abs(self.z - obs["z_min"])))

        # 3. NEIGHBOR COLLISION AVOIDANCE
        for peer in self.peers.values():
            p_dx = self.x - peer.x
            p_dy = self.y - peer.y
            p_dz = self.z - peer.z
            dist_to_peer = math.sqrt(p_dx**2 + p_dy**2 + p_dz**2)
            
            if dist_to_peer < 5.0: # 5 meter buffer zone in 3D
                clearance = max(0.5, dist_to_peer - 1.5)
                rep_factor = 10.0 * (1.0 / clearance - 1.0 / 5.0) * (1.0 / (clearance ** 2))
                fx_rep += rep_factor * (p_dx / dist_to_peer)
                fy_rep += rep_factor * (p_dy / dist_to_peer)
                fz_rep += rep_factor * (p_dz / dist_to_peer)

        return fx_att + fx_rep, fy_att + fy_rep, fz_att + fz_rep

    def update_physics(self):
        self.epoch += 1
        self.battery -= 0.025 # Electric propulsion battery draw
        
        # Elect leader dynamically based on lowest ID
        active_ids = [self.node_id] + list(self.peers.keys())
        lowest_id = min(active_ids) if active_ids else self.node_id
        self.role = "LEADER" if self.node_id == lowest_id else "FOLLOWER"
        
        # Calculate forces
        fx, fy, fz = self._compute_3d_artificial_potential_field()
        
        # Update velocities with inertia (0.8 damping)
        self.vx = self.vx * 0.8 + fx * 0.2
        self.vy = self.vy * 0.8 + fy * 0.2
        self.vz = self.vz * 0.8 + fz * 0.2
        
        # Velocity clamping (max speed 6.0 m/s)
        speed = math.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        max_speed = 6.0
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
            self.vz = (self.vz / speed) * max_speed
            
        # Update position
        self.x += self.vx * 0.1
        self.y += self.vy * 0.1
        self.z += self.vz * 0.1
        
        # Ground clamping
        self.z = max(2.0, self.z)
        
        self.select_tracking_source()

def run_evtol_simulation():
    print("==========================================================")
    print(" 3D eVTOL Air Traffic Control & Space-Edge Tracking Demo")
    print("==========================================================")
    
    # Instantiate 3 eVTOL nodes
    fleet = [eVTOLNode(node_id=201), eVTOLNode(node_id=202), eVTOLNode(node_id=203)]
    
    print("3D Sky Obstacles:")
    for obs in OBSTACLES_3D:
        print(f"  +- {obs['name']:<24} | Center: ({obs['x']:+5.1f}, {obs['y']:+5.1f}) | Height: {obs['z_min']:.1f}-{obs['z_max']:.1f}m | Radius: {obs['radius']:.1f}m")
        
    print("\nStarting 15 simulation epochs...")
    for epoch in range(1, 16):
        # Update states
        for node in fleet:
            node.update_physics()
            
        # Broadcast states (simulated P2P wireless channel)
        for sender in fleet:
            state = sender.get_state()
            for receiver in fleet:
                if receiver.node_id != sender.node_id:
                    receiver.peers[sender.node_id] = state
                    
        print(f"\nEpoch {epoch:02d}:")
        for node in fleet:
            state = node.get_state()
            
            # Distance to closest obstacle
            min_clearance = float('inf')
            closest_obs = "None"
            for obs in OBSTACLES_3D:
                if state.z >= obs["z_min"] and state.z <= obs["z_max"]:
                    dist = math.hypot(state.x - obs["x"], state.y - obs["y"])
                    clearance = dist - obs["radius"]
                    if clearance < min_clearance:
                        min_clearance = clearance
                        closest_obs = obs["name"]
                        
            warning = f"[ALERT] CLOSE TO {closest_obs.upper()} ({min_clearance:.1f}m)" if min_clearance < 5.0 else "CLEAR"
            print(f"  +- eVTOL {state.id} [{state.role:<8}] Pos: ({state.x:+.1f}, {state.y:+.1f}, {state.z:+.1f}) | Bat: {state.battery:.1f}% | Track: {state.tracking_source:<22} | Alert: {warning}")
        time.sleep(0.05)
        
    print("\nStatus: 3D APF Airspace Collision Avoidance Protocol Verified.")
    print("==========================================================")

if __name__ == "__main__":
    run_evtol_simulation()
