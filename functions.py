import os
import pandas as pd
import numpy as np
import joblib

def predict_price(lat, lon, pop_range, ocean_proxi):
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "model.pkl")
    pipeline_path = os.path.join(base, "pipline.pkl")

    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    data = pd.DataFrame({
        'latitude': [lat],
        'longitude': [lon],
        'population_range': [pop_range],
        'ocean_proximity': [ocean_proxi]
    })

    transformed = pipeline.transform(data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.array(transformed, dtype=float).reshape(1, -1)

    prediction = model.predict(transformed)

    # flatten in case model returns [[value]] or [value]
    return float(np.ravel(prediction)[0])
