import torch
import coremltools as ct
import numpy as np

traced_model = torch.jit.load("exports/traced_model.pt")
traced_model.eval()

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
    minimum_deployment_target=ct.target.iOS18,
)

ct_model.save("exports/depthpro_iOS18.mlpackage")