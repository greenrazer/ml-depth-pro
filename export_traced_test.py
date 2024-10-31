import numpy as np
import torch
from PIL import Image
from torch.autograd import profiler
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

traced_model = torch.jit.load("exports/traced_model.pt")
traced_model.eval()

with profiler.profile(profile_memory=True, record_shapes=True, with_stack=True) as prof:
    with torch.no_grad():
        a, b = traced_model(image)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage"))
