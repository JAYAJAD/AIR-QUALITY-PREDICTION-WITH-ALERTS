"""
Flask Backend API — BreatheSafe AQI Predictor
================================================
Season is now AUTO-DETECTED from today's date. No manual season input needed.

SETUP:
  pip install flask flask-cors catboost joblib pandas numpy

RUN:
  python app.py  →  http://localhost:5000
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from catboost import CatBoostRegressor
import joblib, pandas as pd, numpy as np, json, os
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
model = CatBoostRegressor()
model.load_model(os.path.join(BASE, "aqi_model.cbm"))
le_city   = joblib.load(os.path.join(BASE, "label_encoder_city.pkl"))
le_season = joblib.load(os.path.join(BASE, "label_encoder_season.pkl"))
with open(os.path.join(BASE, "cities.json")) as f:
    CITIES = json.load(f)
print("✅ Model loaded")

def auto_season():
    """Detect season from today's month automatically."""
    m = datetime.now().month
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5):  return "Summer"
    if m in (6, 7, 8, 9): return "Monsoon"
    return "Autumn"

def classify_aqi(aqi):
    diseases = []
    if aqi >= 80:
        diseases.append({"name":"Asthma","risk":"HIGH" if aqi>=150 else "MODERATE","warning":"DO NOT go outdoors" if aqi>=150 else "Limit outdoor time, carry inhaler"})
    if aqi >= 80:
        diseases.append({"name":"COPD / Emphysema","risk":"HIGH" if aqi>=150 else "MODERATE","warning":"Stay indoors, keep windows closed" if aqi>=150 else "Avoid outdoor exertion"})
    if aqi >= 100:
        diseases.append({"name":"Heart Disease","risk":"HIGH" if aqi>=150 else "MODERATE","warning":"Risk of heart attack elevated — avoid outdoors" if aqi>=150 else "Reduce outdoor exertion"})
    if aqi >= 80:
        diseases.append({"name":"Chronic Bronchitis","risk":"HIGH" if aqi>=150 else "MODERATE","warning":"Stay indoors — risk of acute exacerbation" if aqi>=150 else "Wear N95 mask outdoors"})
    if aqi >= 50:
        diseases.append({"name":"Lung Cancer Patients","risk":"HIGH" if aqi>=100 else "CAUTION","warning":"HEPA purifier required indoors" if aqi>=100 else "Limit outdoor exposure"})

    if aqi <= 50:   label, color = "Good", "#4ade80"
    elif aqi <= 100: label, color = "Moderate", "#fde047"
    elif aqi <= 150: label, color = "Unhealthy for Sensitive Groups", "#fb923c"
    elif aqi <= 200: label, color = "Unhealthy", "#f87171"
    elif aqi <= 300: label, color = "Very Unhealthy", "#c084fc"
    else:            label, color = "Hazardous", "#fca5a5"

    alert = aqi > 100
    return {"label": label, "color": color, "alert": alert, "diseases": diseases, "season_detected": auto_season()}

@app.route("/health")
def health():
    return jsonify({"status":"ok","today":datetime.now().isoformat(),"season":auto_season()})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()
        city   = d.get("city","").strip()
        season = auto_season()   # ← auto from date, no user input needed
        pm25   = float(d.get("pm25",0))
        pm10   = float(d.get("pm10",0))
        no2    = float(d.get("no2",0))
        so2    = float(d.get("so2",0))
        co     = float(d.get("co",0))
        o3     = float(d.get("o3",0))

        city_enc   = le_city.transform([city])[0] if city in le_city.classes_ else len(le_city.classes_)//2
        season_enc = le_season.transform([season])[0]

        feat = pd.DataFrame([{"city_enc":city_enc,"season_enc":season_enc,
                               "pm25":pm25,"pm10":pm10,"no2":no2,"so2":so2,"co":co,"o3":o3}])
        aqi = round(float(np.clip(model.predict(feat)[0], 0, 500)), 1)
        return jsonify({"success":True,"city":city,"aqi":aqi,"season":season,
                        "date":datetime.now().strftime("%d %b %Y"),
                        "classification":classify_aqi(aqi)})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 400

if __name__ == "__main__":
    print(f"🌍 BreatheSafe API running at http://localhost:5000")
    print(f"📅 Today's season auto-detected: {auto_season()}")
    app.run(debug=True, host="0.0.0.0", port=5000)
