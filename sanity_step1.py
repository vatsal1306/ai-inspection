import platform

import torch

print("Platform:", platform.platform())
print("Python:", platform.python_version())
print("Torch:", torch.__version__)

mps_built = torch.backends.mps.is_built()
mps_available = torch.backends.mps.is_available()

print("MPS built:", mps_built)
print("MPS available:", mps_available)

device = torch.device("mps") if mps_available else torch.device("cpu")
print("Selected device:", device)

# quick tensor op
x = torch.randn(1024, 1024, device=device)
y = x @ x.T
print("Tensor op OK. y.shape =", tuple(y.shape))

# library import checks (we'll use these in Step 2)
import transformers
import accelerate
from PIL import Image
import cv2
import numpy as np

print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("Pillow:", Image.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
