# from flask import Flask, request, render_template
# import numpy as np
# import pickle
#
# # load model and scaler
# model = pickle.load(open('model.pkl', 'rb'))
# ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
#
# # create flask app
# app = Flask(__name__)
#
#
# @app.route('/')
# def index():
#     return render_template("index.html")
#
#
# @app.route("/predict", methods=['POST'])
# def predict():
#
#     # get input values
#     N = float(request.form['Nitrogen'])
#     P = float(request.form['Phosporus'])
#     K = float(request.form['Potassium'])
#     temp = float(request.form['Temperature'])
#     humidity = float(request.form['Humidity'])
#     ph = float(request.form['Ph'])
#     rainfall = float(request.form['Rainfall'])
#
#     # convert to numpy array
#     features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
#
#     # scale features
#     scaled_features = ms.transform(features)
#
#     # probability prediction
#     probs = model.predict_proba(scaled_features)
#
#     # get top 3 predictions
#     top3 = np.argsort(probs[0])[-3:][::-1]
#
#     # crop dictionary
#     crop_dict = {
#         1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
#         6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
#         10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#         14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
#         17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
#         20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
#     }
#
#     # convert predicted numbers to crop names
#     crops = [crop_dict[int(i)] for i in model.classes_[top3]]
#
#     # crop descriptions
#     crop_description = {
#         "Rice": "Rice grows well in high rainfall and humid climates.",
#         "Maize": "Maize grows best in warm climates with moderate rainfall.",
#         "Jute": "Jute requires warm temperature and high humidity.",
#         "Cotton": "Cotton grows well in warm climate with moderate rainfall.",
#         "Coconut": "Coconut thrives in tropical climates with high humidity.",
#         "Papaya": "Papaya grows best in warm climates with well drained soil.",
#         "Orange": "Orange grows well in warm climates with moderate rainfall.",
#         "Apple": "Apple requires cool temperature with moderate rainfall.",
#         "Grapes": "Grapes grow well in warm and dry climates.",
#         "Mango": "Mango thrives in tropical climates with warm temperature.",
#         "Banana": "Banana requires high humidity and nutrient rich soil.",
#         "Pomegranate": "Pomegranate grows well in dry climates.",
#         "Muskmelon": "Muskmelon grows well in warm climate with moderate rainfall.",
#         "Watermelon": "Watermelon grows best in warm climate with well drained soil.",
#         "Lentil": "Lentil grows best in cool climate with low rainfall.",
#         "Blackgram": "Blackgram requires warm climate and moderate rainfall.",
#         "Mungbean": "Mungbean grows well in warm climate with moderate rainfall.",
#         "Mothbeans": "Mothbeans tolerate dry climate and low rainfall.",
#         "Pigeonpeas": "Pigeonpeas grow well in tropical and semi arid climates.",
#         "Kidneybeans": "Kidneybeans require moderate rainfall and fertile soil.",
#         "Chickpea": "Chickpea grows best in cool climate with low rainfall.",
#         "Coffee": "Coffee grows well in humid climates with moderate temperature."
#     }
#
#     # descriptions for each crop
#     descriptions = [
#         crop_description.get(crop, "Suitable crop for given conditions.")
#         for crop in crops
#     ]
#
#     return render_template(
#         "index.html",
#         crops=crops,
#         descriptions=descriptions
#     )
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# -------------------------
# Load model & scaler
# -------------------------
model = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# -------------------------
# Crop mapping
# -------------------------
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# -------------------------
# Crop descriptions
# -------------------------
crop_description = {
    "Rice": "Rice grows best in high rainfall areas with warm and humid climate and fertile soil.",
    "Maize": "Maize thrives in warm climate with moderate rainfall and nutrient-rich soil.",
    "Jute": "Jute requires warm temperature, high humidity, and heavy rainfall.",
    "Cotton": "Cotton grows well in warm climate with moderate rainfall and well-drained soil.",
    "Coconut": "Coconut thrives in tropical regions with high humidity and abundant rainfall.",
    "Papaya": "Papaya grows best in warm climates with well-drained soil and moderate rainfall.",
    "Orange": "Orange prefers warm climate with moderate rainfall and slightly acidic soil.",
    "Apple": "Apple requires cool temperature with moderate rainfall and well-drained soil.",
    "Grapes": "Grapes grow well in warm and dry climates with good drainage soil.",
    "Mango": "Mango thrives in tropical climates with warm temperature and moderate rainfall.",
    "Banana": "Banana requires high humidity, warm temperature, and nutrient-rich soil.",
    "Pomegranate": "Pomegranate grows well in dry climate with low to moderate rainfall.",
    "Chickpea": "Chickpea grows best in cool climate with low rainfall conditions.",
    "Coffee": "Coffee grows well in humid climate with moderate temperature and rich soil."
}

# -------------------------
# Home Route
# -------------------------
@app.route('/')
def index():
    return render_template("index.html")


# -------------------------
# Prediction Route
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Input collection
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Validation
        if not (0 <= ph <= 14):
            return render_template("index.html", error="pH must be between 0 and 14")

        if not (0 <= humidity <= 100):
            return render_template("index.html", error="Humidity must be between 0 and 100")

        if N < 0 or P < 0 or K < 0:
            return render_template("index.html", error="N, P, K values cannot be negative")

        # Prepare input
        features = pd.DataFrame(
            [[N, P, K, temp, humidity, ph, rainfall]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )

        # Scale input
        scaled = ms.transform(features)

        # Predict probabilities
        probs = model.predict_proba(scaled)[0]

        # Top 3 crops
        top3_idx = np.argsort(probs)[-3:][::-1]
        classes = model.classes_

        crops = [crop_dict[int(classes[i])] for i in top3_idx]
        confidences = [round(probs[i] * 100, 2) for i in top3_idx]

        # Description for primary crop
        desc = crop_description.get(
            crops[0],
            "Suitable crop based on soil and climate conditions."
        )

        return render_template(
            "index.html",
            crops=crops,
            confidences=confidences,
            description=desc
        )

    except Exception:
        return render_template(
            "index.html",
            error="Invalid input. Please check all values."
        )


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)