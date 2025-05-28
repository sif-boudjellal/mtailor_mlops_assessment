# MTailor MLOps Assessment

## Author
**Sif eddine Boudjellal**  
LinkedIn: [Sif eddine Boudjellal](https://www.linkedin.com/in/sif-eddine-boudjellal/)

## Project Overview
This guide walks through the steps to convert a PyTorch model to ONNX, create a FastAPI server, containerize it, and deploy it to Cerebrium.

## Prerequisites

- Python 3.12
- PyTorch
- Docker
- Cerebrium account and CLI

## Step 1: Convert PyTorch to ONNX Model

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the conversion script:
```bash
python convert_to_onnx.py
```

This will convert the PyTorch model (`pytorch_model_weights.pth`) to ONNX format (`mtailor_model.onnx`).

## Step 2: Test ONNX Model Locally

1. Run the test script to verify the ONNX model:
```bash
python test.py
```

This script will load the ONNX model and run inference on test images to ensure the conversion was successful.

## Step 3: FastAPI Endpoint Implementation

The FastAPI server is implemented in `main.py` and includes:
- Health check endpoint (`/health`)
- Test endpoint (`/hello`)
- Prediction endpoint (`/predict`)

The server uses:
- `model.py` - Contains model handling and image preprocessing logic
- `main.py` - FastAPI application and endpoint definitions

## Step 4: Test Docker Locally

1. Build the Docker image:
```bash
docker build -t mtailor-model .
```

2. Run the container locally:
```bash
docker run -p 8000:8000 mtailor-model
```

3. Test the local deployment:
```bash
python test_server.py
```

## Step 5: Deploy to Cerebrium

1. Make sure you have Cerebrium CLI installed and configured:
```bash
pip install cerebrium
```

2. Login to Cerebrium:
```bash
cerebrium login
```

3. Deploy the model:
```bash
cerebrium deploy
```

The deployment configuration is defined in `cerebrium.toml`.

## Step 6: Test the Deployed Server

1. After deployment, test the server using the provided endpoint URL:
in `test_server.py` file. Past the URL in the CEREBRIUM_ENDPOINT = "<your_cerebrium_endpoint_here>"
```bash
python test_server.py --image <Path to the image>
```

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /hello`: Test endpoint
- `POST /predict`: Image prediction endpoint
  - Accepts: Image file upload
  - Returns: Model predictions

## Project Structure

```
├── cerebrium.toml         # Cerebrium deployment configuration
├── convert_to_onnx.py     # Script to convert PyTorch model to ONNX
├── Dockerfile             # Docker configuration
├── main.py               # FastAPI server implementation
├── model.py              # Model handling and preprocessing
├── requirements.txt      # Python dependencies
├── test_server.py        # Server testing script
└── test.py              # Local model testing script
```