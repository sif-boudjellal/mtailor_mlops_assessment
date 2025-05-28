from fastapi import FastAPI, File, UploadFile
from model import OnnxModelHandler, ImagePreprocessor
from PIL import Image
from io import BytesIO

app = FastAPI()

model_handler = OnnxModelHandler('mtailor_model.onnx')
preprocessor = ImagePreprocessor()

@app.post("/hello")
def hello():
    return {"message": "Hello Cerebrium!"}

@app.get("/health")
def health():
    return "OK"

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    input_tensor = preprocessor.preprocess(image)
    output = model_handler.predict(input_tensor)
    return {"prediction": output.tolist()}

