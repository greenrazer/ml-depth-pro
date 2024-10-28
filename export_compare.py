import time

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor


class ResizeCustom:
    def __call__(self, image):
        return torch.nn.functional.interpolate(
            image,
            size=(1536, 1536),
            mode="bilinear",
            align_corners=False,
        )


class BatchSingle:
    def __call__(self, image):
        return image.unsqueeze(0)


transform = Compose(
    [
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        BatchSingle(),
        ResizeCustom(),
    ]
)
# Load and preprocess an image.
img_pil = Image.open("data/example.jpg")
image_orig = np.array(img_pil)
_, H, W = image_orig.shape
image = transform(image_orig)

pt_model = torch.jit.load("exports/traced_model.pt").eval()
ct_model = ct.models.MLModel("exports/depthpro.mlpackage")

pt_start_time = time.perf_counter()
pt_out = pt_model(image)
pt_end_time = time.perf_counter()
pt_elapsed_time = pt_end_time - pt_start_time

ct_start_time = time.perf_counter()
ct_out = ct_out = ct_model.predict({"pixel_values": image})
ct_end_time = time.perf_counter()
ct_elapsed_time = ct_end_time - ct_start_time

pt_canonical_inverse_depth, pt_fov_deg = pt_out[0].detach().numpy(), pt_out[1].detach().numpy()
ct_canonical_inverse_depth, ct_fov_deg = ct_out["depth_meters"], ct_out["view_angle"]

diff = np.abs(pt_canonical_inverse_depth - ct_canonical_inverse_depth)
print("min", np.min(diff))
print("median", np.median(diff))
print("mean", np.mean(diff))
print("max", np.max(diff))

percent_time_diff = 100 * abs(ct_elapsed_time - pt_elapsed_time) / pt_elapsed_time
print("pytorch time", pt_elapsed_time, "s")
print("coreml time", ct_elapsed_time, "s")
print(
    "coreml was",
    percent_time_diff,
    "%",
    "faster" if pt_elapsed_time > ct_elapsed_time else "slower",
    "than pytorch",
)
