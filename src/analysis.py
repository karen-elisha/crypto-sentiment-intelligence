import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# STEP 1: LOAD RAW FILES
# =========================
trades = pd.read_csv("data/historical_data.csv")
sentiment = pd.read_csv("data/fear_greed_index.csv")

# =========================
# STEP 2: CLEAN DATA
# =========================
trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
trades['date'] = trades['Timestamp IST'].dt.date

sentiment['timestamp'] = pd.to_datetime(sentiment['timestamp'], unit='s')
sentiment['date'] = sentiment['timestamp'].dt.date

# =========================
# STEP 3: MERGE DATA
# =========================
merged = pd.merge(trades, sentiment[['date','value','classification']], on='date', how='left')

# Basic features
merged['is_profitable'] = merged['Closed PnL'] > 0
merged['is_long'] = merged['Direction'].str.contains('Long', na=False)

# =========================
# STEP 4: DAILY DATASET
# =========================
daily = merged.groupby('date').agg(
    total_pnl=('Closed PnL','sum'),
    avg_pnl=('Closed PnL','mean'),
    trade_count=('Closed PnL','count'),
    value=('value','mean'),
    classification=('classification','first')
).reset_index()

# =========================
# STEP 5: SENTIMENT AGG
# =========================
sent_agg = merged.groupby('classification').agg(
    avg_pnl=('Closed PnL','mean'),
    win_rate=('is_profitable','mean'),
    trade_count=('Closed PnL','count'),
    avg_volume=('Size USD','mean')
).reset_index()

# =========================
# STEP 6: ACCOUNT FEATURES
# =========================
acc = merged.groupby('Account').agg(
    total_pnl=('Closed PnL','sum'),
    avg_pnl=('Closed PnL','mean'),
    win_rate=('is_profitable','mean'),
    trade_count=('Closed PnL','count'),
    avg_size_usd=('Size USD','mean'),
    pnl_std=('Closed PnL','std')
).reset_index()

acc['contrarian_ratio'] = np.random.rand(len(acc))  # placeholder

# =========================
# SAVE DATASETS
# =========================
merged.to_csv("merged.csv", index=False)
daily.to_csv("daily_trades.csv", index=False)
sent_agg.to_csv("sent_agg.csv", index=False)
acc.to_csv("acc_features.csv", index=False)

print("✅ 4 datasets generated")

# =========================
# STEP 7: SIMPLE ANALYSIS
# =========================
results = {}

# Sentiment performance
results['sentiment_perf'] = sent_agg.to_dict('records')

# Time series
daily = daily.sort_values('date')
daily['cumulative_pnl'] = daily['total_pnl'].cumsum()

results['timeseries'] = {
    'dates': daily['date'].astype(str).tolist(),
    'cumulative_pnl': daily['cumulative_pnl'].tolist(),
    'daily_pnl': daily['total_pnl'].tolist()
}

# =========================
# STEP 8: CLUSTERING
# =========================
features = ['win_rate','avg_pnl','trade_count','avg_size_usd']
X = acc[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
acc['cluster'] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

acc['x'] = coords[:,0]
acc['y'] = coords[:,1]

results['clustering'] = acc[['Account','cluster','x','y']].to_dict('records')

# =========================
# STEP 9: RANDOM FOREST
# =========================
rf_data = merged[merged['Closed PnL'] != 0].copy()

rf_data['target'] = (rf_data['Closed PnL'] > 0).astype(int)

X = rf_data[['value','Size USD']].fillna(0)
y = rf_data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

report = classification_report(y_test,y_pred, output_dict=True)

results['model'] = {
    'accuracy': report['accuracy']
}

# =========================
# SAVE JSON
# =========================
with open("analysis_results.json","w") as f:
    json.dump(results,f,indent=2)

print("✅ analysis_results.json created")