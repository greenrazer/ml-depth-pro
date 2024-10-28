import coremltools as ct
import coremltools.optimize as cto

ct_model = ct.models.MLModel("exports/depthpro.mlpackage")

config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=4)
)
compressed_model = cto.coreml.palettize_weights(ct_model, config)

compressed_model.save("exports/depthpro_palettized_wholetensor_4bit.mlpackage")
