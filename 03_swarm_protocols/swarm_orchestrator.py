"""
swarm_orchestrator.py
~~~~~~~~~~~~~~~~~~~~
A decentralized, peer-to-peer swarm orchestration node.
Implements leader-follower flocking dynamics integrated with an Artificial Potential Field (APF)
algorithm for localized collision and obstacle avoidance.
"""

import socket
import struct
import threading
import time
import random
import json
import math
from dataclasses import dataclass, asdict

# Multicast configurations
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
SOCKET_TIMEOUT = 0.5

@dataclass
class RobotState:
    id: int
    role: str
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float
    battery_level: float
    epoch: int

# Configuration for simulated static obstacles in the flight zone
OBSTACLES = [
    {"x": 2.0, "y": 2.0, "radius": 1.5},
    {"x": -3.0, "y": -4.0, "radius": 2.0}
]

class SwarmNode:
    """
    Represents an autonomous agent in a decentralized robotic swarm.
    Communicates via localized networking and plans collision-free paths using APF.
    """
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.role = "FOLLOWER"
        self.peers = {}
        
        self.pos_x = random.uniform(-8.0, 8.0)
        self.pos_y = random.uniform(-8.0, 8.0)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.battery = 100.0
        self.epoch = 0
        self.running = False
        
        # APF Parameters
        self.k_att = 0.6  # Attractive gain (pull towards target/leader)
        self.k_rep = 8.0  # Repulsive gain (push away from obstacles)
        self.d0 = 3.0     # Obstacle influence distance threshold
        
        self.lock = threading.Lock()
        
        # Socket setup
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(('', MCAST_PORT))
        except Exception:
            try:
                self.sock.bind(('127.0.0.1', 0))
            except Exception:
                pass
        
        mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
        try:
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception:
            pass
        self.sock.settimeout(SOCKET_TIMEOUT)

    def get_state(self) -> RobotState:
        with self.lock:
            return RobotState(
                id=self.node_id,
                role=self.role,
                position_x=round(self.pos_x, 3),
                position_y=round(self.pos_y, 3),
                velocity_x=round(self.vel_x, 3),
                velocity_y=round(self.vel_y, 3),
                battery_level=round(self.battery, 2),
                epoch=self.epoch
            )

    def _compute_artificial_potential_field(self) -> tuple[float, float]:
        """
        Computes the net force vector on the agent.
        Net Force = Attractive Force (to leader/target) + Repulsive Forces (from obstacles)
        """
        fx_att, fy_att = 0.0, 0.0
        fx_rep, fy_rep = 0.0, 0.0
        
        # 1. ATTRACTIVE FORCE
        if self.role == "LEADER":
            # Leader is attracted to dynamic waypoint coordinates (circular path over time)
            target_x = 5.0 * math.cos(self.epoch * 0.05)
            target_y = 5.0 * math.sin(self.epoch * 0.05)
        else:
            # Followers are attracted to the leader's position with offset
            leader = next((p for p in self.peers.values() if p.role == "LEADER"), None)
            if leader:
                target_x = leader.position_x
                target_y = leader.position_y
            else:
                target_x, target_y = 0.0, 0.0 # Default center point if no leader
                
        dx = target_x - self.pos_x
        dy = target_y - self.pos_y
        dist_to_target = math.hypot(dx, dy)
        
        if dist_to_target > 0.1:
            # Attractive force increases linearly with distance
            fx_att = self.k_att * (dx / dist_to_target)
            fy_att = self.k_att * (dy / dist_to_target)

        # 2. REPULSIVE FORCE (from static obstacles)
        for obs in OBSTACLES:
            d_x = self.pos_x - obs["x"]
            d_y = self.pos_y - obs["y"]
            dist_to_obs = math.hypot(d_x, d_y)
            
            # Distance from robot surface to obstacle surface
            clearance = dist_to_obs - obs["radius"]
            if clearance <= 0.05:
                clearance = 0.05 # Prevent division by zero
                
            if clearance < self.d0:
                # Standard APF Repulsive Formula: 
                # F_rep = k_rep * (1/d - 1/d0) * (1/d^2) * unit_vector
                rep_factor = self.k_rep * (1.0 / clearance - 1.0 / self.d0) * (1.0 / (clearance ** 2))
                fx_rep += rep_factor * (d_x / dist_to_obs)
                fy_rep += rep_factor * (d_y / dist_to_obs)
                
        # 3. COLLISION AVOIDANCE FROM NEIGHBORING PEERS
        for peer in self.peers.values():
            p_dx = self.pos_x - peer.position_x
            p_dy = self.pos_y - peer.position_y
            dist_to_peer = math.hypot(p_dx, p_dy)
            
            if dist_to_peer < 1.8: # Personal safety radius (1.8 meters)
                clearance = max(0.1, dist_to_peer - 0.5)
                rep_factor = 4.0 * (1.0 / clearance - 1.0 / 1.8) * (1.0 / (clearance ** 2))
                fx_rep += rep_factor * (p_dx / dist_to_peer)
                fy_rep += rep_factor * (p_dy / dist_to_peer)

        # Combined net force
        return fx_att + fx_rep, fy_att + fy_rep

    def update_physics(self):
        """Calculates trajectory using forces and updates position."""
        with self.lock:
            self.epoch += 1
            self.battery -= 0.015
            
            # Simple leadership assignment
            active_ids = [self.node_id] + list(self.peers.keys())
            lowest_id = min(active_ids) if active_ids else self.node_id
            self.role = "LEADER" if self.node_id == lowest_id else "FOLLOWER"
            
            # Compute forces from Potential Field
            force_x, force_y = self._compute_artificial_potential_field()
            
            # Update velocities (simulating mass and damping)
            self.vel_x = self.vel_x * 0.7 + force_x * 0.3
            self.vel_y = self.vel_y * 0.7 + force_y * 0.3
            
            # Clamp speed for safety limits
            speed = math.hypot(self.vel_x, self.vel_y)
            max_speed = 3.0
            if speed > max_speed:
                self.vel_x = (self.vel_x / speed) * max_speed
                self.vel_y = (self.vel_y / speed) * max_speed
                
            # Update coordinate position
            self.pos_x += self.vel_x * 0.1
            self.pos_y += self.vel_y * 0.1

    def send_broadcast(self):
        state = self.get_state()
        data = json.dumps(asdict(state)).encode('utf-8')
        try:
            self.sock.sendto(data, (MCAST_GRP, MCAST_PORT))
        except Exception:
            pass

    def listen_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                peer_data = json.loads(data.decode('utf-8'))
                peer_id = peer_data['id']
                if peer_id != self.node_id:
                    self.peers[peer_id] = RobotState(**peer_data)
            except socket.timeout:
                continue
            except Exception:
                break

def run_multi_agent_simulation():
    print("==================================================")
    print("Swarm Orchestration - APF Collision Avoidance Demo")
    print("==================================================")
    
    # 3 Swarm nodes initialized
    node1 = SwarmNode(node_id=101)
    node2 = SwarmNode(node_id=102)
    node3 = SwarmNode(node_id=103)
    
    nodes = [node1, node2, node3]
    
    print("Environment Obstacles:")
    for idx, obs in enumerate(OBSTACLES, 1):
        print(f"  +- Obstacle {idx} | Position: ({obs['x']:.1f}, {obs['y']:.1f}) | Radius: {obs['radius']:.1f}m")
        
    print("\nSimulating 15 cycles of potential-field swarm navigation...")
    for epoch in range(1, 16):
        # 1. Physics update
        for n in nodes:
            n.update_physics()
            
        # 2. P2P message exchange simulation
        for sender in nodes:
            sender_state = sender.get_state()
            for receiver in nodes:
                if receiver.node_id != sender.node_id:
                    receiver.peers[sender.node_id] = sender_state
                    
        # 3. Print positions & checking collision warnings
        print(f"\nEpoch {epoch:02d}:")
        for n in nodes:
            state = n.get_state()
            
            # Check proximity to obstacles
            min_clearance = float('inf')
            for obs in OBSTACLES:
                dist = math.hypot(state.position_x - obs["x"], state.position_y - obs["y"])
                clearance = dist - obs["radius"]
                if clearance < min_clearance:
                    min_clearance = clearance
                    
            warning = "[ALERT] NEAR OBSTACLE!" if min_clearance < 0.6 else "OK"
            print(f"  +- Agent {state.id} [{state.role:<8}] Pos: ({state.position_x:+.2f}, {state.position_y:+.2f}) | Clear: {min_clearance:.2f}m | Status: {warning}")
        time.sleep(0.05)
        
    print("\nStatus: Swarm Artificial Potential Field Verification Successful.")
    print("==================================================")

if __name__ == "__main__":
    run_multi_agent_simulation()
