import numpy as np
from flask import Flask, render_template, redirect, url_for, request
from functions import predict_price
app = Flask(__name__)

population_ranges = np.array([
    '0-2000', '2000-4000', '4000-6000', '6000-8000', '8000-10000',
    '10000-12000', '12000-14000', '14000-16000', '16000-18000', '34000-36000'
])

@app.route('/')
def home():
    return render_template('home.html', pop_ranges = population_ranges)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    lat = request.form.get('lat')
    lon = request.form.get('lon')
    pop_range = request.form.get('pop')
    ocean_proxi = request.form.get('ocean_proximity')
    result = predict_price(lat, lon, pop_range, ocean_proxi)
    return render_template('predict.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)