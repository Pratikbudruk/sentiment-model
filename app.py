from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
import os

# Create FastAPI app
app = FastAPI()


# Define a model class for the input data
class TextInput(BaseModel):
    sentence: str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the sentiment analysis model (assuming it's stored in a directory named 'model')
model_directory = './model'  # Set your model directory path here

if not os.path.exists(model_directory):
    raise FileNotFoundError(f"Model directory {model_directory} does not exist.")

# Load pre-trained sentiment analysis pipeline (this can be a locally trained model too)
sentiment_analyzer = pipeline('sentiment-analysis', model=model_directory, device=device.index if device.type == 'cuda' else -1)


@app.get("/status")
async def status():
    """Endpoint to check if the model and server are working."""
    try:
        # Test the model by performing a dummy prediction
        result = sentiment_analyzer("This is a test sentence.")
        return {"status": "success", "model_loaded": True, "test_result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/predict")
async def predict(input_data: TextInput):
    """Endpoint to predict sentiment of the input sentence."""
    sentence = input_data.sentence
    try:
        # Predict sentiment
        result = sentiment_analyzer(sentence)
        torch.cuda.empty_cache()
        return {"sentence": sentence, "prediction": result[0]['label'], "confidence": result[0]['score']}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
