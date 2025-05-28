import torch
from pytorch_model import Classifier,BasicBlock

# Load the trained PyTorch model
mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
mtailor.load_state_dict(torch.load('pytorch_model_weights.pth'))
mtailor.eval()

# Define dummy input matching the model's input shape
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model to ONNX format
torch.onnx.export(mtailor, dummy_input, "mtailor_model.onnx", 
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("Model successfully converted to ONNX format.")