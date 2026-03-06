"""
AQI Model Training — CatBoost (Season auto-detected from date)
==============================================================
Run: python train_aqi_model.py
Generates: aqi_model.cbm, label_encoder_city.pkl, label_encoder_season.pkl, cities.json
"""
import pandas as pd, numpy as np, joblib, json
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

np.random.seed(42)

cities = [
    "Delhi","Mumbai","Kolkata","Chennai","Bangalore","Hyderabad","Pune",
    "Ahmedabad","Jaipur","Lucknow","Kanpur","Patna","Agra","Varanasi","Nagpur",
    "New York","Los Angeles","Chicago","Houston","London","Paris","Berlin",
    "Beijing","Shanghai","Tokyo","Seoul","Singapore","Sydney","Toronto","Dubai"
]
base_aqi = {
    "Delhi":195,"Kanpur":185,"Patna":180,"Lucknow":172,"Agra":165,"Varanasi":160,
    "Kolkata":140,"Mumbai":125,"Ahmedabad":118,"Jaipur":112,"Hyderabad":98,
    "Pune":92,"Chennai":85,"Nagpur":98,"Bangalore":82,
    "Beijing":155,"Shanghai":130,"Tokyo":60,"Seoul":78,"Singapore":48,
    "New York":52,"Los Angeles":63,"Chicago":52,"Houston":58,
    "London":40,"Paris":46,"Berlin":38,"Toronto":38,
    "Dubai":88,"Sydney":27
}
# Season auto from month (same logic as backend)
def month_to_season(m):
    if m in (12,1,2): return "Winter"
    if m in (3,4,5):  return "Summer"
    if m in (6,7,8,9): return "Monsoon"
    return "Autumn"

season_mult = {"Winter":1.4,"Summer":1.1,"Monsoon":0.65,"Autumn":1.0}

rows = []
for _ in range(6000):
    city   = np.random.choice(cities)
    month  = np.random.randint(1,13)
    season = month_to_season(month)
    sm     = season_mult[season]
    base   = base_aqi[city] * sm
    pm25 = max(5,  np.random.normal(base*.55, base*.2))
    pm10 = max(10, np.random.normal(base*.9,  base*.3))
    no2  = max(5,  np.random.normal(base*.28, base*.12))
    so2  = max(2,  np.random.normal(base*.1,  base*.06))
    co   = max(0.2,np.random.normal(base*.008,0.003))
    o3   = max(5,  np.random.normal(38, 14))
    # EPA sub-index for PM2.5
    def sub(c,bp,ar):
        for i,(lo,hi) in enumerate(bp):
            if lo<=c<=hi:
                al,ah=ar[i]; return ((ah-al)/(hi-lo))*(c-lo)+al
        return min(500,c*2)
    bp=[(0,12),(12.1,35.4),(35.5,55.4),(55.5,150.4),(150.5,250.4),(250.5,350.4),(350.5,500)]
    ar=[(0,50),(51,100),(101,150),(151,200),(201,300),(301,400),(401,500)]
    aqi = max(sub(pm25,bp,ar), sub(pm10,[(0,54),(55,154),(155,254),(255,354),(355,424),(425,504),(505,604)],ar))
    aqi = round(min(500, max(0, aqi + np.random.normal(0,12))), 1)
    rows.append({"city":city,"season":season,"month":month,"pm25":round(pm25,2),"pm10":round(pm10,2),"no2":round(no2,2),"so2":round(so2,2),"co":round(co,3),"o3":round(o3,2),"aqi":aqi})

df = pd.DataFrame(rows)
print(f"Dataset: {df.shape}\nAQI stats:\n{df.aqi.describe()}")

le_city   = LabelEncoder().fit(df.city)
le_season = LabelEncoder().fit(df.season)
df["city_enc"]   = le_city.transform(df.city)
df["season_enc"] = le_season.transform(df.season)
joblib.dump(le_city,   "label_encoder_city.pkl")
joblib.dump(le_season, "label_encoder_season.pkl")
with open("cities.json","w") as f: json.dump(sorted(cities), f)

X = df[["city_enc","season_enc","pm25","pm10","no2","so2","co","o3"]]
y = df["aqi"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

model = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=8,
                          loss_function="RMSE", random_seed=42, verbose=100,
                          early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=(X_test,y_test), use_best_model=True)
model.save_model("aqi_model.cbm")

y_pred = model.predict(X_test)
print(f"\nMAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R² : {r2_score(y_test,y_pred):.4f}")
print("✅ Done! Run: python app.py")
