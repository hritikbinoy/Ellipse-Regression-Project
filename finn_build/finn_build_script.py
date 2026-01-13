
# ===================================================================
# FINN FPGA Build Script for Ellipse Regression QONNX Model
# ===================================================================

import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveUniqueNodeNames, GiveReadableTensorNames

# Import FINN transformations
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.absorb import AbsorbSignBiasIntoMultiThreshold
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferBinaryStreamingFCLayer
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

# ===================================================================
# STEP 1: Load and Prepare QONNX Model
# ===================================================================

model_path = "finn_build/ellipse_regression_qonnx.onnx"
model = ModelWrapper(model_path)

print("Step 1: Loading QONNX model...")
print(f"  Input: {model.graph.input[0].name}")
print(f"  Output: {model.graph.output[0].name}")

# ===================================================================
# STEP 2: Apply Pre-processing Transformations
# ===================================================================

print("\nStep 2: Pre-processing transformations...")

# Infer shapes
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())

# Convert QONNX to FINN
model = model.transform(ConvertQONNXtoFINN())

print("  ✓ QONNX converted to FINN")

# ===================================================================
# STEP 3: Streamline and Optimize
# ===================================================================

print("\nStep 3: Streamlining and optimization...")

model = model.transform(Streamline())
model = model.transform(LowerConvsToMatMul())
model = model.transform(MakeMaxPoolNHWC())
model = model.transform(AbsorbSignBiasIntoMultiThreshold())
model = model.transform(MoveScalarLinearPastInvariants())
model = model.transform(InferDataLayouts())

print("  ✓ Model streamlined")

# ===================================================================
# STEP 4: Convert to Hardware Dataflow Layers
# ===================================================================

print("\nStep 4: Converting to hardware layers...")

# Create dataflow partition
model = model.transform(CreateDataflowPartition())

# Convert to streaming FC layers (for fully-connected parts)
model = model.transform(InferBinaryStreamingFCLayer())

print("  ✓ Hardware layers created")

# ===================================================================
# STEP 5: Set FPGA Configuration
# ===================================================================

print("\nStep 5: Configuring for FPGA target...")

# Target FPGA (choose one):
# - "Pynq-Z1" or "Pynq-Z2": Zynq-7000 (small, cheap)
# - "ZCU104": Zynq UltraScale+ (medium performance)
# - "U250": Alveo data center card (high performance)
# - "KV260": Kria SOM (edge AI)

target_fpga = "Pynq-Z1"  # Change this based on your hardware
target_clk_ns = 10  # 100 MHz clock (10ns period)

print(f"  Target: {target_fpga}")
print(f"  Clock: {1000/target_clk_ns:.0f} MHz ({target_clk_ns}ns period)")

# ===================================================================
# STEP 6: Synthesize to Bitstream (Hardware Compilation)
# ===================================================================

print("\nStep 6: Hardware synthesis (this takes 1-4 hours)...")
print("  [SIMULATION MODE - SKIPPED]")
print("  For actual synthesis, use FINN build_dataflow() function")

# In real deployment:
# from finn.builder.build_dataflow import build_dataflow
# build_dataflow(model, target_fpga, target_clk_ns, output_dir="finn_build/output")

# ===================================================================
# STEP 7: Expected Performance
# ===================================================================

print("\n" + "="*70)
print("EXPECTED FPGA PERFORMANCE")
print("="*70)

# Estimate cycles per inference
num_layers = 7  # 4 conv + 3 fc layers
avg_ops_per_layer = 1000  # simplified estimate

cycles_per_inference = num_layers * avg_ops_per_layer
latency_us = (cycles_per_inference * target_clk_ns) / 1000
throughput = 1_000_000 / latency_us

print(f"Estimated cycles per inference: ~{cycles_per_inference:,}")
print(f"Estimated latency: ~{latency_us:.1f} μs")
print(f"Estimated throughput: ~{throughput:,.0f} samples/sec")
print(f"")
print(f"Speedup vs CPU (3,300 samples/sec): ~{throughput/3300:.1f}x")

print("="*70)
