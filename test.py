import numpy as np
from model import OnnxModelHandler, ImagePreprocessor

# Paths to your ONNX model and test image
model_path = 'mtailor_model.onnx'
image_path = 'n01440764_tench.jpeg'

# Initialize the classes
model_handler = OnnxModelHandler(model_path)
preprocessor = ImagePreprocessor()

# Preprocess the image
input_tensor = preprocessor.preprocess(image_path)

# Make a prediction
output = model_handler.predict(input_tensor)

# Print the output
print("Prediction Class:", output)


# Simple assertion to check output shape (modify according to your model's expected output)
assert isinstance(output, np.ndarray), "Output is not a numpy array"
print("Test passed successfully!")
