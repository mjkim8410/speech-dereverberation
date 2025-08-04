import onnxruntime as ort
print("Loaded from:", ort.__file__)
print("Version:", ort.__version__)

print("Available providers:", ort.get_available_providers())
# Expect something like:
# ['CUDAExecutionProvider', 'CPUExecutionProvider']
