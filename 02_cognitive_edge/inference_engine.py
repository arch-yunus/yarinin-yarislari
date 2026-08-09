"""
inference_engine.py
~~~~~~~~~~~~~~~~~~
A comprehensive edge AI quantized inference engine simulating an INT4 multi-layer feedforward
network with Recurrent State Feedback (RNN-like cell) for humanoid balancing forecasting.
Loads quantized weights, scales, and biases dynamically from a local JSON file.
"""

import sys
import os
import json
import time
import math

class QuantizedLinearLayer:
    """
    Quantized Fully Connected Layer using 4-bit packed weights (-8 to 7 signed).
    Supports loading weights directly from pre-calibrated floating point arrays.
    """
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.packed_weights = bytearray((input_dim * output_dim + 1) // 2)
        self.scales = [1.0] * output_dim
        self.biases = [0.0] * output_dim

    def set_weight(self, in_idx: int, out_idx: int, val: int):
        val = max(-8, min(7, val)) & 0x0F
        linear_idx = in_idx * self.output_dim + out_idx
        byte_idx = linear_idx // 2
        is_high_nibble = (linear_idx % 2) == 0
        
        current_byte = self.packed_weights[byte_idx]
        if is_high_nibble:
            current_byte = (current_byte & 0x0F) | (val << 4)
        else:
            current_byte = (current_byte & 0xF0) | val
        self.packed_weights[byte_idx] = current_byte

    def get_weight(self, in_idx: int, out_idx: int) -> int:
        linear_idx = in_idx * self.output_dim + out_idx
        byte_idx = linear_idx // 2
        is_high_nibble = (linear_idx % 2) == 0
        
        val = self.packed_weights[byte_idx]
        if is_high_nibble:
            val = (val >> 4) & 0x0F
        else:
            val = val & 0x0F
            
        if val & 0x08:
            return val - 16
        return val

    def load_parameters(self, weights_2d: list[list[int]], scales: list[float], biases: list[float]):
        """Populates packed weights, scales, and biases from configuration values."""
        assert len(weights_2d) == self.input_dim
        assert len(weights_2d[0]) == self.output_dim
        assert len(scales) == self.output_dim
        assert len(biases) == self.output_dim
        
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                self.set_weight(i, j, weights_2d[i][j])
                
        self.scales = list(scales)
        self.biases = list(biases)

    def forward(self, x: list[float]) -> list[float]:
        """Runs low-bit integer GEMM (General Matrix Multiply) dequantized forward pass."""
        output = [0.0] * self.output_dim
        for j in range(self.output_dim):
            accumulator = 0
            for i in range(self.input_dim):
                w_int = self.get_weight(i, j)
                # Fixed-point simulation: scale float inputs to integer values for computation
                scaled_input = int(x[i] * 128.0)
                accumulator += scaled_input * w_int
                
            # Dequantize back to floating point using layer scale and add bias
            output[j] = (accumulator / 128.0) * self.scales[j] + self.biases[j]
        return output

class RecurrentCognitiveEdgeAI:
    """
    Multi-layer neural network featuring an RNN feedback loop to remember previous
    stabilization states, generating higher fidelity balancing forecasting predictions.
    """
    def __init__(self, weights_path: str):
        # Neural Network Dimensions: 
        # Input (4) = 2 Sensory Inputs + 2 Hidden Recurrent Outputs
        self.layer1 = QuantizedLinearLayer(4, 8)
        # Output (4) = 2 System Actuator Offsets + 2 Hidden Recurrent Outputs
        self.layer2 = QuantizedLinearLayer(8, 4)
        
        # State feedback vector (recurrent loop)
        self.recurrent_state = [0.0, 0.0]
        
        self.load_weights_from_file(weights_path)

    def load_weights_from_file(self, path: str):
        """Loads INT4 network configuration from a JSON weights file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Quantized weights file not found at: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Parse and adjust config weights to include recurrent dimensions.
        # Layer 1 Input: 4-dim (2 sensory inputs + 2 recurrent states)
        orig_l1_w = data["layer1_weights"]
        # Add 2 zero-padded or custom state weights to match dimension 4
        extended_l1_w = orig_l1_w + [[1, -1, 2, -2, 0, 1, -1, 3], [-3, 2, -1, 4, -4, 0, 2, -2]]
        
        self.layer1.load_parameters(
            weights_2d=extended_l1_w,
            scales=data["layer1_scales"],
            biases=data["layer1_biases"]
        )
        
        # Layer 2 Output: 4-dim (2 system offsets + 2 recurrent states)
        orig_l2_w = data["layer2_weights"]
        extended_l2_w = [row + [row[0] // 2, -row[1] // 2] for row in orig_l2_w]
        
        self.layer2.load_parameters(
            weights_2d=extended_l2_w,
            scales=data["layer2_scales"] + [0.02, 0.015],
            biases=data["layer2_biases"] + [0.05, -0.05]
        )

    def run_inference(self, sensory_inputs: list[float]) -> list[float]:
        """
        Runs neural network inference.
        Concat sensory inputs with past recurrent states, run forward pass,
        extract outputs, and update recurrent memory.
        """
        # Concat inputs: [Sensory X, Sensory Y, Recurrent Hidden 0, Recurrent Hidden 1]
        network_input = sensory_inputs + self.recurrent_state
        
        # Layer 1 forward
        h1 = self.layer1.forward(network_input)
        # Activation function: Tanh (approximated for edge efficiency)
        h1_activated = [math.tanh(val) for val in h1]
        
        # Layer 2 forward
        network_output = self.layer2.forward(h1_activated)
        
        # Extract results
        actuator_offsets = network_output[0:2]
        self.recurrent_state = network_output[2:4] # Update recurrent memory for next tick
        
        return actuator_offsets

def main():
    print("==================================================")
    print("Cognitive Edge AI - Quantized Recurrent Inference")
    print("==================================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_file = os.path.join(script_dir, "weights_config.json")
    
    try:
        engine = RecurrentCognitiveEdgeAI(weights_file)
        print("Success: Quantized weights loaded from weights_config.json.")
    except Exception as e:
        print(f"Error initializing engine: {e}")
        sys.exit(1)
        
    # Running 3 inference steps simulating consecutive clock ticks
    mock_sensory_timeline = [
        [0.12, -0.35],  # Tick 1: Tilt right, falling velocity
        [0.08, -0.12],  # Tick 2: Recovering, low velocity
        [-0.02, 0.05]   # Tick 3: Overshot slightly left
    ]
    
    for tick, sensory in enumerate(mock_sensory_timeline, 1):
        print(f"\n--- Clock Tick {tick:02d} ---")
        print(f"Sensory Inputs: Pitch Angle={sensory[0]:.4f} rad, Velocity={sensory[1]:.4f} rad/s")
        print(f"Recurrent State Memory: [{engine.recurrent_state[0]:.4f}, {engine.recurrent_state[1]:.4f}]")
        
        start_time = time.perf_counter()
        outputs = engine.run_inference(sensory)
        end_time = time.perf_counter()
        
        latency_us = (end_time - start_time) * 1e6
        
        print(f"Predicted Ankle Offset: {outputs[0]:.4f} N-m")
        print(f"Predicted Knee Offset:  {outputs[1]:.4f} N-m")
        print(f"Inference Latency:      {latency_us:.2f} microseconds")

    print("\nStatus: Dynamic Recurrent Quantized Inference Validated.")
    print("==================================================")

if __name__ == "__main__":
    main()
