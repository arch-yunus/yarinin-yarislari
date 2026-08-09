"""
inference_engine.py
~~~~~~~~~~~~~~~~~~
A lightweight, low-bit quantized neural network inference engine designed to execute
on resource-constrained Edge AI nodes (e.g., custom MCUs or low-power NPUs).
Implements simulated 4-bit weight integer quantization (INT4) and dynamic scale factor calibration.
"""

import sys
import time

class QuantizedLinearLayer:
    """
    Simulates a fully connected linear layer with 4-bit integer quantized weights.
    Weights are packed as 4-bit values (signed, range -8 to 7) to reduce memory footprint.
    """
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 4-bit weights packed into 8-bit integers (two weights per byte)
        # Sign-extended values: -8 to 7
        self.packed_weights = bytearray((input_dim * output_dim + 1) // 2)
        
        # Floating point scale factor for each output neuron to dequantize values during GEMM
        self.scales = [1.0] * output_dim
        self.biases = [0.0] * output_dim
        
        self._initialize_mock_weights()

    def _initialize_mock_weights(self):
        """Populates weights with a wave pattern for test determinism."""
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                weight_val = (i + j) % 16 - 8  # Range: -8 to 7
                self.set_weight(i, j, weight_val)
                
        # Small scales and biases
        self.scales = [0.0125 * (k + 1) for k in range(self.output_dim)]
        self.biases = [-0.1 * k for k in range(self.output_dim)]

    def set_weight(self, in_idx: int, out_idx: int, val: int):
        """Sets the 4-bit weight value at (in_idx, out_idx). Val must be [-8, 7]."""
        val = max(-8, min(7, val)) & 0x0F  # Clamp and convert to unsigned 4-bit representation
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
        """Retrieves the de-quantized weight as a signed integer."""
        linear_idx = in_idx * self.output_dim + out_idx
        byte_idx = linear_idx // 2
        is_high_nibble = (linear_idx % 2) == 0
        
        val = self.packed_weights[byte_idx]
        if is_high_nibble:
            val = (val >> 4) & 0x0F
        else:
            val = val & 0x0F
            
        # Convert unsigned 4-bit back to signed integer (twos complement / sign extension)
        if val & 0x08:
            return val - 16
        return val

    def forward(self, x: list[float]) -> list[float]:
        """Runs INT4 GEMM (General Matrix Multiply) dequantized forward pass."""
        assert len(x) == self.input_dim, f"Dimension mismatch: expected {self.input_dim}, got {len(x)}"
        
        output = [0.0] * self.output_dim
        for j in range(self.output_dim):
            accumulator = 0
            for i in range(self.input_dim):
                # Retrieve integer weight
                w_int = self.get_weight(i, j)
                # Accumulate integer product
                # In real hardware, this loop runs in an integer MAC unit
                accumulator += int(x[i] * 100.0) * w_int  # Multiply scaled input with integer weight
                
            # Dequantize back to float, apply scale and bias
            output[j] = (accumulator / 100.0) * self.scales[j] + self.biases[j]
            
        return output

class CognitiveEdgeAI:
    """
    A multi-layer quantized neural network running local inference
    for humanoid locomotion trajectory optimization.
    """
    def __init__(self):
        # 2-Layer MLP: Input (2) -> Hidden (8) -> Output (2)
        # Input features: [Pitch Angle, Pitch Velocity]
        # Output features: [Ankle Torque Offset, Knee Torque Offset]
        self.layer1 = QuantizedLinearLayer(2, 8)
        self.layer2 = QuantizedLinearLayer(8, 2)

    def run_inference(self, sensory_inputs: list[float]) -> list[float]:
        # Layer 1
        h1 = self.layer1.forward(sensory_inputs)
        # ReLU Activation
        h1_activated = [max(0.0, val) for val in h1]
        # Layer 2
        outputs = self.layer2.forward(h1_activated)
        return outputs

def main():
    print("==================================================")
    print("Cognitive Edge AI - Quantized Inference Engine v1.0")
    print("==================================================")
    
    # Initialize Engine
    engine = CognitiveEdgeAI()
    
    # Simulated sensory inputs [Tilt Angle (rad), Angular Velocity (rad/s)]
    mock_sensor_inputs = [0.15, -0.45]
    
    print(f"Sensory Inputs: Pitch Angle={mock_sensor_inputs[0]:.4f} rad, Velocity={mock_sensor_inputs[1]:.4f} rad/s")
    
    # Benchmark Inference Time
    start_time = time.perf_counter()
    outputs = engine.run_inference(mock_sensor_inputs)
    end_time = time.perf_counter()
    
    duration_us = (end_time - start_time) * 1e6
    
    print("\n--- Inference Results ---")
    print(f"Target Ankle Torque Offset: {outputs[0]:.4f} N-m")
    print(f"Target Knee Torque Offset:  {outputs[1]:.4f} N-m")
    print(f"Inference Latency:          {duration_us:.2f} microseconds")
    print("Status: Edge Quantized Inference Successful (FP32 Emulation)")
    print("==================================================")

if __name__ == "__main__":
    main()
