import os
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, precision_recall_fscore_support)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("⚠️  XGBoost not available, skipping.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    print("⚠️  LightGBM not available, skipping.")

# =========================
# CONFIG
# =========================
DATA_FOLDER = "data"
DASHBOARD_FOLDER = "dashboard"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

print("🚀 Starting Enhanced Crypto Sentiment Analysis Pipeline...")

# =========================
# STEP 1: LOAD RAW FILES
# =========================
print("📂 Loading data...")
trades = pd.read_csv(os.path.join(DATA_FOLDER, "historical_data.csv"))
sentiment = pd.read_csv(os.path.join(DATA_FOLDER, "fear_greed_index.csv"))

# =========================
# STEP 2: CLEAN & ENGINEER FEATURES
# =========================
print("🔧 Cleaning and engineering features...")
trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
trades['date'] = trades['Timestamp IST'].dt.date
trades['hour'] = trades['Timestamp IST'].dt.hour
trades['day_of_week'] = trades['Timestamp IST'].dt.dayofweek

sentiment['timestamp'] = pd.to_datetime(sentiment['timestamp'], unit='s')
sentiment['date'] = sentiment['timestamp'].dt.date
sentiment['value'] = pd.to_numeric(sentiment['value'], errors='coerce')

# Sentiment momentum features
sentiment = sentiment.sort_values('date')
sentiment['sent_ma_3'] = sentiment['value'].rolling(3, min_periods=1).mean()
sentiment['sent_ma_7'] = sentiment['value'].rolling(7, min_periods=1).mean()
sentiment['sent_change'] = sentiment['value'].diff()
sentiment['sent_volatility'] = sentiment['value'].rolling(7, min_periods=1).std()

# =========================
# STEP 3: MERGE
# =========================
print("🔗 Merging datasets...")
merge_cols = ['date', 'value', 'classification', 'sent_ma_3', 'sent_ma_7', 'sent_change', 'sent_volatility']
merged = pd.merge(trades, sentiment[merge_cols], on='date', how='left')

merged['is_profitable'] = merged['Closed PnL'] > 0
merged['is_long'] = merged['Direction'].str.contains('Long', na=False)
merged['is_short'] = merged['Direction'].str.contains('Short', na=False)
merged['is_open'] = merged['Direction'].str.contains('Open', na=False)
merged['is_close'] = merged['Direction'].str.contains('Close', na=False)
merged['leverage'] = np.where(merged['Start Position'] > 0,
                              merged['Size USD'] / merged['Start Position'], 0)
merged['fee_ratio'] = np.where(merged['Size USD'] > 0,
                               merged['Fee'] / merged['Size USD'], 0)
merged['pnl_per_dollar'] = np.where(merged['Size USD'] > 0,
                                     merged['Closed PnL'] / merged['Size USD'], 0)

# =========================
# STEP 4: DAILY AGGREGATION
# =========================
print("📊 Building daily aggregations...")
daily = merged.groupby('date').agg(
    total_pnl=('Closed PnL', 'sum'),
    avg_pnl=('Closed PnL', 'mean'),
    trade_count=('Closed PnL', 'count'),
    total_volume=('Size USD', 'sum'),
    avg_size=('Size USD', 'mean'),
    long_count=('is_long', 'sum'),
    short_count=('is_short', 'sum'),
    win_rate=('is_profitable', 'mean'),
    sentiment_value=('value', 'mean'),
    classification=('classification', 'first')
).reset_index()
daily = daily.sort_values('date')
daily['cumulative_pnl'] = daily['total_pnl'].cumsum()
daily['pnl_ma_7'] = daily['total_pnl'].rolling(7, min_periods=1).mean()

# =========================
# STEP 5: SENTIMENT PERFORMANCE
# =========================
print("📈 Analyzing sentiment performance...")
sent_agg = merged.groupby('classification').agg(
    avg_pnl=('Closed PnL', 'mean'),
    median_pnl=('Closed PnL', 'median'),
    total_pnl=('Closed PnL', 'sum'),
    win_rate=('is_profitable', 'mean'),
    trade_count=('Closed PnL', 'count'),
    avg_volume=('Size USD', 'mean'),
    avg_fee=('Fee', 'mean'),
    pnl_std=('Closed PnL', 'std')
).reset_index()
sent_agg['sharpe_like'] = sent_agg['avg_pnl'] / sent_agg['pnl_std'].replace(0, np.nan)

# =========================
# STEP 6: ACCOUNT FEATURES (richer)
# =========================
print("👤 Building account profiles...")
acc = merged.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    avg_pnl=('Closed PnL', 'mean'),
    median_pnl=('Closed PnL', 'median'),
    win_rate=('is_profitable', 'mean'),
    trade_count=('Closed PnL', 'count'),
    avg_size_usd=('Size USD', 'mean'),
    pnl_std=('Closed PnL', 'std'),
    total_fees=('Fee', 'sum'),
    avg_fee_ratio=('fee_ratio', 'mean'),
    long_pct=('is_long', 'mean'),
    short_pct=('is_short', 'mean'),
    avg_sentiment=('value', 'mean'),
    unique_coins=('Coin', 'nunique')
).reset_index()
acc['profit_factor'] = acc['total_pnl'] / acc['total_fees'].replace(0, np.nan)
acc['sharpe'] = acc['avg_pnl'] / acc['pnl_std'].replace(0, np.nan)

# Contrarian ratio: trading against sentiment
fear_trades = merged[merged['classification'].isin(['Fear', 'Extreme Fear'])]
greed_trades = merged[merged['classification'].isin(['Greed', 'Extreme Greed'])]

buy_in_fear = fear_trades.groupby('Account')['is_long'].sum()
sell_in_greed = greed_trades.groupby('Account')['is_short'].sum()
total_by_acc = merged.groupby('Account')['Closed PnL'].count()

acc['contrarian_ratio'] = ((buy_in_fear.reindex(acc['Account'], fill_value=0).values +
                            sell_in_greed.reindex(acc['Account'], fill_value=0).values) /
                           total_by_acc.reindex(acc['Account'], fill_value=1).values)

# =========================
# STEP 7: STRATEGY ANALYSIS
# =========================
print("⚔️ Comparing strategies...")
# Contrarian: buy in fear, short in greed
contrarian_mask = (
    (merged['classification'].isin(['Fear', 'Extreme Fear']) & merged['is_long']) |
    (merged['classification'].isin(['Greed', 'Extreme Greed']) & merged['is_short'])
)
# Momentum: buy in greed, short in fear
momentum_mask = (
    (merged['classification'].isin(['Greed', 'Extreme Greed']) & merged['is_long']) |
    (merged['classification'].isin(['Fear', 'Extreme Fear']) & merged['is_short'])
)

def strategy_stats(mask, name):
    subset = merged[mask]
    if len(subset) == 0:
        return {'strategy': name, 'trades': 0}
    return {
        'strategy': name,
        'trades': int(len(subset)),
        'avg_pnl': round(float(subset['Closed PnL'].mean()), 2),
        'total_pnl': round(float(subset['Closed PnL'].sum()), 2),
        'win_rate': round(float((subset['Closed PnL'] > 0).mean()), 4),
        'avg_size': round(float(subset['Size USD'].mean()), 2),
        'median_pnl': round(float(subset['Closed PnL'].median()), 2)
    }

strategies = [strategy_stats(contrarian_mask, 'Contrarian'),
              strategy_stats(momentum_mask, 'Momentum')]

# =========================
# STEP 8: HOURLY ANALYSIS
# =========================
print("🕐 Analyzing trading hours...")
hourly = merged.groupby('hour').agg(
    avg_pnl=('Closed PnL', 'mean'),
    trade_count=('Closed PnL', 'count'),
    win_rate=('is_profitable', 'mean')
).reset_index()

# =========================
# STEP 9: COIN PERFORMANCE
# =========================
print("🪙 Analyzing coin performance...")
coin_perf = merged.groupby('Coin').agg(
    avg_pnl=('Closed PnL', 'mean'),
    total_pnl=('Closed PnL', 'sum'),
    trade_count=('Closed PnL', 'count'),
    win_rate=('is_profitable', 'mean'),
    avg_volume=('Size USD', 'mean')
).reset_index()
coin_perf = coin_perf[coin_perf['trade_count'] >= 50].sort_values('total_pnl', ascending=False).head(15)

# =========================
# STEP 10: CLUSTERING (enhanced)
# =========================
print("🧩 Clustering accounts...")
cluster_features = ['win_rate', 'avg_pnl', 'trade_count', 'avg_size_usd',
                    'contrarian_ratio', 'long_pct', 'sharpe']
X_cluster = acc[cluster_features].fillna(0)
X_cluster = X_cluster.replace([np.inf, -np.inf], 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
acc['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
acc['x'] = coords[:, 0]
acc['y'] = coords[:, 1]

# Cluster profiles
cluster_profiles = []
for c in sorted(acc['cluster'].unique()):
    group = acc[acc['cluster'] == c]
    cluster_profiles.append({
        'cluster': int(c),
        'count': int(len(group)),
        'avg_pnl': round(float(group['avg_pnl'].mean()), 2),
        'avg_win_rate': round(float(group['win_rate'].mean()), 4),
        'avg_trade_count': round(float(group['trade_count'].mean()), 0),
        'avg_contrarian': round(float(group['contrarian_ratio'].mean()), 4),
        'avg_sharpe': round(float(group['sharpe'].mean()), 4)
    })

# =========================
# STEP 11: ML MODELS (multi-model comparison)
# =========================
print("🤖 Training ML models...")
rf_data = merged[merged['Closed PnL'] != 0].copy()
rf_data['target'] = (rf_data['Closed PnL'] > 0).astype(int)

feature_cols = ['value', 'Size USD', 'hour', 'day_of_week', 'sent_ma_3',
                'sent_ma_7', 'sent_change', 'sent_volatility', 'fee_ratio']
X = rf_data[feature_cols].fillna(0)
X = X.replace([np.inf, -np.inf], 0)
y = rf_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
}
if HAS_XGB:
    models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       random_state=42, use_label_encoder=False,
                                       eval_metric='logloss', verbosity=0)
if HAS_LGBM:
    models['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                         random_state=42, verbose=-1)

model_results = []
best_acc = 0
best_model_name = ""
best_model = None

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    except Exception:
        auc = None

    report = classification_report(y_test, y_pred, output_dict=True)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    acc_score = report['accuracy']
    result = {
        'name': name,
        'accuracy': round(acc_score, 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'auc_roc': auc,
        'cv_mean': round(float(cv_scores.mean()), 4),
        'cv_std': round(float(cv_scores.std()), 4)
    }
    model_results.append(result)

    if acc_score > best_acc:
        best_acc = acc_score
        best_model_name = name
        best_model = model

# Feature importance from best model
feat_imp = []
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    for fname, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        feat_imp.append({'feature': fname, 'importance': round(float(imp), 4)})

# Confusion matrix for best model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
cm_data = {
    'true_neg': int(cm[0][0]), 'false_pos': int(cm[0][1]),
    'false_neg': int(cm[1][0]), 'true_pos': int(cm[1][1])
}

# =========================
# STEP 12: KEY INSIGHTS
# =========================
print("💡 Generating insights...")
best_sent = sent_agg.loc[sent_agg['avg_pnl'].idxmax()]
worst_sent = sent_agg.loc[sent_agg['avg_pnl'].idxmin()]
best_hour = hourly.loc[hourly['avg_pnl'].idxmax()]
best_strat = max(strategies, key=lambda s: s.get('avg_pnl', 0))

top_accounts = acc.nlargest(5, 'total_pnl')[['Account', 'total_pnl', 'win_rate', 'trade_count', 'sharpe']].copy()
top_accounts['Account'] = top_accounts['Account'].str[:10] + '...'

insights = {
    'best_sentiment': str(best_sent['classification']),
    'best_sentiment_pnl': round(float(best_sent['avg_pnl']), 2),
    'worst_sentiment': str(worst_sent['classification']),
    'worst_sentiment_pnl': round(float(worst_sent['avg_pnl']), 2),
    'best_strategy': best_strat['strategy'],
    'best_strategy_pnl': best_strat.get('avg_pnl', 0),
    'best_hour': int(best_hour['hour']),
    'best_hour_pnl': round(float(best_hour['avg_pnl']), 2),
    'best_model': best_model_name,
    'best_model_accuracy': round(best_acc, 4),
    'total_trades': int(len(merged)),
    'total_accounts': int(acc.shape[0]),
    'total_coins': int(merged['Coin'].nunique()),
    'overall_win_rate': round(float(merged['is_profitable'].mean()), 4),
    'total_pnl': round(float(merged['Closed PnL'].sum()), 2),
    'date_range': f"{merged['date'].min()} to {merged['date'].max()}"
}

# =========================
# STEP 13: CORRELATION MATRIX
# =========================
corr_cols = ['Closed PnL', 'Size USD', 'value', 'Fee', 'hour']
corr_data = merged[corr_cols].dropna()
corr_matrix = corr_data.corr()
correlations = {
    'labels': corr_cols,
    'matrix': [[round(float(v), 4) for v in row] for row in corr_matrix.values]
}

# =========================
# BUILD RESULTS JSON
# =========================
print("📦 Building results JSON...")
results = {
    'insights': insights,
    'sentiment_perf': sent_agg.round(4).to_dict('records'),
    'timeseries': {
        'dates': daily['date'].astype(str).tolist(),
        'cumulative_pnl': [round(float(v), 2) for v in daily['cumulative_pnl']],
        'daily_pnl': [round(float(v), 2) for v in daily['total_pnl']],
        'pnl_ma_7': [round(float(v), 2) for v in daily['pnl_ma_7']],
        'sentiment': [round(float(v), 2) if pd.notna(v) else 0 for v in daily['sentiment_value']],
        'win_rate': [round(float(v), 4) for v in daily['win_rate']]
    },
    'hourly': hourly.round(4).to_dict('records'),
    'strategies': strategies,
    'coin_performance': coin_perf.round(2).to_dict('records'),
    'models': model_results,
    'best_model': best_model_name,
    'feature_importance': feat_imp,
    'confusion_matrix': cm_data,
    'clustering': acc[['Account', 'cluster', 'x', 'y', 'total_pnl', 'win_rate',
                        'trade_count', 'contrarian_ratio', 'sharpe']].round(4).to_dict('records'),
    'cluster_profiles': cluster_profiles,
    'correlations': correlations,
    'top_accounts': top_accounts.round(4).to_dict('records')
}

# Save outputs
merged.to_csv(os.path.join(DATA_FOLDER, "merged.csv"), index=False)
daily.to_csv(os.path.join(DATA_FOLDER, "daily_trades.csv"), index=False)
sent_agg.to_csv(os.path.join(DATA_FOLDER, "sent_agg.csv"), index=False)
acc.to_csv(os.path.join(DATA_FOLDER, "acc_features.csv"), index=False)

with open(os.path.join(DASHBOARD_FOLDER, "analysis_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("=" * 50)
print("✅ Pipeline complete!")
print(f"   📊 {len(merged)} trades analyzed")
print(f"   🤖 Best model: {best_model_name} ({best_acc:.1%} accuracy)")
print(f"   💰 Best sentiment: {insights['best_sentiment']} (avg PnL: ${insights['best_sentiment_pnl']:.2f})")
print(f"   ⚔️  Best strategy: {insights['best_strategy']}")
print(f"   📁 Results saved to dashboard/analysis_results.json")