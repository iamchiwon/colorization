from colorizers import *
import coremltools as ct
import torch

#torch_model = eccv16(pretrained=True).eval()
torch_model = siggraph17(pretrained=True).eval()

example_input = torch.rand(1, 1, 256, 256)
traced_model = torch.jit.trace(torch_model, example_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input1", shape=(1, 1, 256, 256))]
)
coreml_model.save("Colorizer.mlpackage")