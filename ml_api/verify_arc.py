import os
import sys

import openvino as ov
import onnxruntime as ort

MODEL = "/model_cache/ml_api/onnx/model-weights.onnx"
DEVICE = os.environ.get("OPENVINO_DEVICE", "GPU")

print("=" * 70)
print("OpenVINO hardware")
print("=" * 70)

core = ov.Core()

for dev in core.available_devices:
    try:
        name = core.get_property(dev, "FULL_DEVICE_NAME")
    except Exception as exc:
        name = str(exc)
    print(f"{dev} = {name}")

print()
print("=" * 70)
print("ONNX Runtime")
print("=" * 70)

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())
print("Requested OpenVINO device:", DEVICE)
print("Model:", MODEL)

if "OpenVINOExecutionProvider" not in ort.get_available_providers():
    print("ERROR: OpenVINOExecutionProvider is unavailable")
    sys.exit(1)

try:
    session = ort.InferenceSession(
        MODEL,
        providers=[
            (
                "OpenVINOExecutionProvider",
                {"device_type": DEVICE},
            ),
            "CPUExecutionProvider",
        ],
    )
except Exception as exc:
    print()
    print("ERROR compiling Obico model:")
    print(exc)
    sys.exit(1)

print()
print("Active providers:", session.get_providers())

print()
print("Model inputs:")
for inp in session.get_inputs():
    print(" ", inp.name, inp.shape, inp.type)

print()
print("Model outputs:")
for out in session.get_outputs():
    print(" ", out.name, out.shape, out.type)

print()
print("Arc/OpenVINO Obico model compilation: OK")
