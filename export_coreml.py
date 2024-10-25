from huggingface_hub import PyTorchModelHubMixin
from depth_pro import create_model_and_transforms, load_rgb
from depth_pro.depth_pro import (create_backbone_model,
                                 DepthPro, DepthProEncoder, MultiresConvDecoder)
import depth_pro
from torchvision.transforms import Compose, Normalize, ToTensor

import torch
import coremltools as ct
import numpy as np

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


# Load model and preprocessing transform
model = DepthProWrapper.from_pretrained("apple/DepthPro-mixin")
transform = Compose(
        [
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

model.eval()

# Load and preprocess an image.
image, _, _ = depth_pro.load_rgb("data/example.jpg")
image = transform(image)
image = image.unsqueeze(0)
image = torch.nn.functional.interpolate(
    image,
    size=(1536, 1536),
    mode="bilinear",
    align_corners=False,
)

traced_model = torch.jit.trace(model, (image,))

print(traced_model)

# Run inference.
# prediction = traced_model(image)

ct_model = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
			name='pixel_values', 
			shape=(1, 3, 1536, 1536),
			dtype=np.float32
		),
    ],
	outputs=[
		ct.TensorType(
			name="depth_meters",
			dtype=np.float32
		),
        ct.TensorType(
			name="view_angle",
			dtype=np.float32
		)
	],
	convert_to="mlprogram",
)

ct_model.save("depthpro.mlpackage")