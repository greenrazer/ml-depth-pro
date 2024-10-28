import coremltools as ct
import coremltools.optimize as cto

ct_model = ct.models.MLModel("exports/depthpro.mlpackage")

config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=4)
)
compressed_model = cto.coreml.palettize_weights(ct_model, config)

compressed_model.save("exports/depthpro_palettized_wholetensor_4bit.mlpackage")

# ct_model = ct.models.MLModel("exports/depthpro_iOS18.mlpackage")

# config = cto.coreml.OptimizationConfig(
#     global_config=cto.coreml.OpPalettizerConfig(
#         mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=32
#     )
# )
# compressed_model = cto.coreml.palettize_weights(ct_model, config)

# compressed_model.save("exports/depthpro_iOS18_palettized_grouped32_4bit.mlpackage")
