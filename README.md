<<<<<<< HEAD
# AQI Predictor — Complete Setup Guide

## Project Structure
```
aqi-predictor/
├── train_aqi_model.py   ← Step 1: Train the ML model
├── app.py               ← Step 2: Run the Flask API
├── index.html           ← Step 3: Open in browser
├── aqi_model.cbm        ← (generated after training)
├── label_encoder_city.pkl
├── label_encoder_season.pkl
└── cities.json
```

---

## STEP 1 — Install Python Dependencies

```bash
pip install catboost pandas numpy scikit-learn joblib flask flask-cors
```

---

## STEP 2 — Get Real Training Data (Recommended)

### Option A: Kaggle AQI India Dataset (Best for Indian cities)
1. Go to: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Download `city_day.csv`
3. In `train_aqi_model.py`, replace the synthetic data section with:

```python
df = pd.read_csv("city_day.csv")
df = df.dropna(subset=["AQI"])
df.rename(columns={"City":"city","PM2.5":"pm25","PM10":"pm10",
                   "NO2":"no2","SO2":"so2","CO":"co","O3":"o3","AQI":"aqi"}, inplace=True)
df["season"] = pd.to_datetime(df["Date"]).dt.month.map({
    12:"Winter",1:"Winter",2:"Winter",
    3:"Summer",4:"Summer",5:"Summer",
    6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
    10:"Autumn",11:"Autumn"
})
df = df[["city","season","pm25","pm10","no2","so2","co","o3","aqi"]].dropna()
```

### Option B: OpenAQ API (Global cities)
- https://openaq.org/developers/api/

### Option C: Use Synthetic Data (Quick Start)
The script already includes 5000 synthetic samples — just run it directly!

---

## STEP 3 — Train the Model

```bash
python train_aqi_model.py
```

This generates:
- `aqi_model.cbm` — trained CatBoost model
- `label_encoder_city.pkl` — city name encoder
- `label_encoder_season.pkl` — season encoder
- `cities.json` — list of known cities

---

## STEP 4 — Start the Flask API

```bash
python app.py
```

Server starts at: http://localhost:5000

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"Delhi","season":"Winter","pm25":120,"pm10":200,"no2":60,"so2":25,"co":2.5,"o3":40}'
```

---

## STEP 5 — Open the Website

Simply open `index.html` in your browser. That's it!

> **Note**: The website works in "Demo Mode" even without the Flask backend running.
> Connect the backend for real ML predictions.

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | /health | Server status |
| GET | /cities | List of known cities |
| POST | /predict | Predict AQI |

### POST /predict — Request Body
```json
{
  "city": "Delhi",
  "season": "Winter",
  "pm25": 120.0,
  "pm10": 200.0,
  "no2": 60.0,
  "so2": 25.0,
  "co": 2.5,
  "o3": 40.0
}
```

### POST /predict — Response
```json
{
  "success": true,
  "city": "Delhi",
  "aqi": 183.4,
  "classification": {
    "label": "Unhealthy",
    "color": "#ef4444",
    "emoji": "🚫",
    "description": "Everyone may begin to experience health effects.",
    "advice": "Everyone should reduce outdoor exertion...",
    "health_risks": ["Severe asthma attacks", "..."],
    "caution_groups": ["🚫 Asthma patients — DO NOT GO OUTSIDE", "..."]
  }
}
```

---

## AQI Classification & Health Diseases

| AQI Range | Category | At-Risk Diseases |
|-----------|----------|-----------------|
| 0–50 | Good | None |
| 51–100 | Moderate | Asthma (mild), Allergies |
| 101–150 | Unhealthy for Sensitive Groups | Asthma, COPD, Heart Disease, Childhood Asthma |
| 151–200 | Unhealthy | Asthma, COPD, Bronchitis, Cardiovascular, Lung Cancer |
| 201–300 | Very Unhealthy | All of the above + Neurological effects |
| 301–500 | Hazardous | Entire population at immediate risk |

---

## Deploying to the Web (Optional)

### Backend: Deploy on Render / Railway / Heroku
1. Create `requirements.txt`:
   ```
   flask
   flask-cors
   catboost
   joblib
   pandas
   numpy
   scikit-learn
   ```
2. Deploy to Render.com (free tier available)
3. Update `API_BASE` in `index.html` to your deployed URL

### Frontend: Deploy on GitHub Pages / Netlify
- Just upload `index.html` — no build step needed!
=======
# AIR-QUALITY-PREDICTION-WITH-ALERTS
Air pollution has become a major environmental and health concern in many cities. This project focuses on predicting the Air Quality Index (AQI) of different locations and providing health alerts and safety recommendations based on the predicted AQI levels.
>>>>>>> f93087b80e30a4bdb0fba19ead6edc636effe3e0
