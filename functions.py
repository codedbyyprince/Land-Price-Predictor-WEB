import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# ✅ Pre-load model and pipeline once at startup (not every function call)
MODEL_REPO = "mlwithprince/landpricepredictor"

# Download from Hugging Face only once — then cached locally
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl")
PIPELINE_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="pipline.pkl")

# Load the model and pipeline
model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

def predict_price(lat, lon, pop_range, ocean_proxi):
    # Prepare input data
    data = pd.DataFrame({
        'latitude': [lat],
        'longitude': [lon],
        'population_range': [pop_range],
        'ocean_proximity': [ocean_proxi]
    })

    # Transform input using the loaded pipeline
    transformed = pipeline.transform(data)

    # Ensure it's a proper numpy array
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.array(transformed, dtype=float).reshape(1, -1)

    # Predict using the loaded model
    prediction = model.predict(transformed)

    # Return as a simple float
    return float(np.ravel(prediction)[0])
