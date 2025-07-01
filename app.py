from flask import Flask, render_template, request
from markupsafe import Markup
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pickle
from werkzeug.utils import secure_filename

# Load trained models
classifier = load_model('Trained_model.h5')
crop_recommendation_model_path = 'Crop_Recommendation.pkl'

# Load crop recommendation model safely
with open(crop_recommendation_model_path, 'rb') as f:
    crop_recommendation_model = pickle.load(f)

# Load fertilizer dictionary
from fertilizer import fertilizer_dict

app = Flask(__name__)

# Ensure the upload folder exists
upload_folder = 'static/user_uploaded'
os.makedirs(upload_folder, exist_ok=True)

# Route for Fertilizer Prediction
@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():
    crop_name = request.form['cropname']
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    # Read crop NPK data
    df = pd.read_csv('Data/Crop_NPK.csv')
    crop_data = df[df['Crop'] == crop_name]

    if crop_data.empty:
        return "Invalid crop name", 400

    N_desired, P_desired, K_desired = crop_data.iloc[0][['N', 'P', 'K']]
    n, p, k = N_desired - N_filled, P_desired - P_filled, K_desired - K_filled

    key1 = "NHigh" if n < 0 else "Nlow" if n > 0 else "NNo"
    key2 = "PHigh" if p < 0 else "Plow" if p > 0 else "PNo"
    key3 = "KHigh" if k < 0 else "Klow" if k > 0 else "KNo"

    return render_template('Fertilizer-Result.html',
                           recommendation1=Markup(fertilizer_dict[key1]),
                           recommendation2=Markup(fertilizer_dict[key2]),
                           recommendation3=Markup(fertilizer_dict[key3]),
                           diff_n=abs(n), diff_p=abs(p), diff_k=abs(k))

# Function to predict pests using trained classifier
def pred_pest(pest):
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)
        return np.argmax(result, axis=1)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 'x'

# Home route
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

# Routes for different pages
@app.route('/CropRecommendation.html')
def crop():
    return render_template('CropRecommendation.html')

@app.route('/FertilizerRecommendation.html')
def fertilizer():
    return render_template('FertilizerRecommendation.html')

@app.route('/PesticideRecommendation.html')
def pesticide():
    return render_template('PesticideRecommendation.html')

# Route for pest image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return "No file selected", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    pred = pred_pest(pest=file_path)
    if pred == 'x':
        return render_template('unaptfile.html')

    pest_identified = {
        0: 'aphids', 1: 'armyworm', 2: 'beetle', 3: 'bollworm',
        4: 'earthworm', 5: 'grasshopper', 6: 'mites', 7: 'mosquito',
        8: 'sawfly', 9: 'stem borer'
    }.get(pred[0], 'unknown')

    return render_template(f'{pest_identified}.html', pred=pest_identified)

# Crop Prediction route
@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    try:
        features = [
            int(request.form['nitrogen']), int(request.form['phosphorous']),
            int(request.form['potassium']), float(request.form['temperature']),
            float(request.form['humidity']), float(request.form['ph']),
            float(request.form['rainfall'])
        ]
        
        data = np.array([features])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        
        return render_template('crop-result.html',
                               prediction=final_prediction,
                               pred=f'img/crop/{final_prediction}.jpg')
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
