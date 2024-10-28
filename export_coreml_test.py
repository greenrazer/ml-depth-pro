import torch
from torchvision.transforms import Compose, Normalize, ToTensor

import coremltools as ct
import numpy as np
from PIL import Image


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

ct_model = ct.models.MLModel("exports/depthpro.mlpackage")

out = ct_model.predict({
    "pixel_values": image
})

canonical_inverse_depth, fov_deg = out["depth_meters"], out["view_angle"]

# fov in degrees to focal length in pixels
f_px = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg.squeeze().astype(float)))

width_focal_length_ratio = (W / f_px)

# turns depth in range 0-1 into inverse meters from camera
inverse_depth = canonical_inverse_depth * width_focal_length_ratio

# turns inverse depth back into range 0-1
max_invdepth_vizu = min(inverse_depth.max(), 1e1)
min_invdepth_vizu = max(1e-3, inverse_depth.min())
inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
    max_invdepth_vizu - min_invdepth_vizu
)

color_depth = (inverse_depth_normalized.squeeze() * 255).astype(np.uint8)
Image.fromarray(color_depth).show()