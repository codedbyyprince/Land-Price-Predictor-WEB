import pandas as pd
import numpy as np 
import joblib

def predict_price(lat,lon,po_range,ocean_proxi):
    model = joblib.load('/media/prince/5A4E832F4E83034D/testing /traning the model/model.pkl')
    pipline = joblib.load('/media/prince/5A4E832F4E83034D/testing /traning the model/pipeline.pkl')
    data = pd.DataFrame({
    'latitude': [lat],
    'longitude': [lon],
    'population': [po_range],
    'ocean_proximity': [ocean_proxi]
    })
    transformed_data = pipline.transfomr(data)
    prediction = model.predict(transformed_data)
    data['land_value'] = prediction
    return data['land_value']