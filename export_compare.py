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

pt_model = torch.jit.load("exports/traced_model.pt").eval()
ct_model = ct.models.MLModel("exports/depthpro.mlpackage")

pt_out = pt_model(image)
ct_out = ct_model.predict({
    "pixel_values": image
})

pt_canonical_inverse_depth, pt_fov_deg = pt_out[0].detach().numpy(), pt_out[1].detach().numpy()
ct_canonical_inverse_depth, ct_fov_deg = ct_out["depth_meters"], ct_out["view_angle"]

diff = np.abs(pt_canonical_inverse_depth-ct_canonical_inverse_depth)
print("min", np.min(diff))
print("median", np.median(diff))
print("mean", np.mean(diff))
print("max", np.max(diff))