import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

class OnnxModelHandler:
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return np.argmax(outputs[0], axis=1)


class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be either a file path string or a PIL Image object")
        
        image_tensor = self.transform(image).unsqueeze(0).numpy()
        return image_tensor

