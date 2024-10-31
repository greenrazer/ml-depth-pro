import numpy as np
import torch
from depth_pro.depth_pro import (
    DepthPro,
    DepthProEncoder,
    MultiresConvDecoder,
    create_backbone_model,
)
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image
from torch.autograd import profiler
from torchvision.transforms import Compose, Normalize, ToTensor


class DepthProWrapper(DepthPro, PyTorchModelHubMixin):
    """Depth Pro network."""

    def __init__(
        self,
        patch_encoder_preset: str,
        image_encoder_preset: str,
        decoder_features: str,
        fov_encoder_preset: str,
        use_fov_head: bool = True,
        **kwargs,
    ):
        """Initialize Depth Pro."""

        patch_encoder, patch_encoder_config = create_backbone_model(
            preset=patch_encoder_preset
        )
        image_encoder, _ = create_backbone_model(
            preset=image_encoder_preset
        )

        fov_encoder = None
        if use_fov_head and fov_encoder_preset is not None:
            fov_encoder, _ = create_backbone_model(preset=fov_encoder_preset)

        dims_encoder = patch_encoder_config.encoder_feature_dims
        hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
        encoder = DepthProEncoder(
            dims_encoder=dims_encoder,
            patch_encoder=patch_encoder,
            image_encoder=image_encoder,
            hook_block_ids=hook_block_ids,
            decoder_features=decoder_features,
        )
        decoder = MultiresConvDecoder(
            dims_encoder=[encoder.dims_encoder[0]] + list(encoder.dims_encoder),
            dim_decoder=decoder_features,
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            last_dims=(32, 1),
            use_fov_head=use_fov_head,
            fov_encoder=fov_encoder,
        )

model = DepthProWrapper.from_pretrained("apple/DepthPro-mixin")
model.eval()

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

torch.set_num_threads(1)

with profiler.profile(profile_memory=True, record_shapes=True, with_stack=True, with_modules=True) as prof:
    with torch.no_grad():
        a, b = model(image)
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage"))
# prof.export_chrome_trace("trace.json")

total_mem = 0
mem_trace_id = []
for i, f in enumerate(prof.function_events):
    if f.name == "aten::empty":
        # print(f.cpu_memory_usage)
        total_mem += f.cpu_memory_usage
        print(i, f.name, f.cpu_memory_usage*1e-9, f.input_shapes)
        mem_trace_id.append([i, f.name, f.cpu_memory_usage, f.input_shapes])

# print(total_mem)
# prof.export_stacks("whatever.json")
