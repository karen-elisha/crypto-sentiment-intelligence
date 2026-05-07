# 🚀 Crypto Sentiment Intelligence System

## 📌 Overview

This project analyzes trading performance using market sentiment (Fear & Greed Index).
It combines data processing, machine learning, and automated insights to identify optimal trading strategies.

https://karen-elisha.github.io/crypto-sentiment-intelligence/

---

## 🧠 Key Features

* 📊 Sentiment vs Profit Analysis
* ⚔️ Strategy Comparison (Contrarian vs Momentum)
* 🤖 Machine Learning (Random Forest for trade prediction)
* 🧩 Trader Clustering (K-Means)
* 🚀 Automated Insight Generation

---

## 🔄 Workflow

Raw Data → Processing → ML → Insights → JSON Output

---

## 📂 Project Structure

src/ → Core logic
data/ → Raw datasets
dashboard/ → Output JSON
automation/ → Execution script

---

## ▶️ How to Run

Install dependencies:
pip install -r requirements.txt

Run pipeline:
python src/analysis.py

OR (automated):
./automation/run.sh

---

## 📊 Example Output

{
"best_strategy": "Momentum",
"best_sentiment": "Extreme Greed",
"recommendation": "Use Momentum strategy during Extreme Greed"
}

---

## 💡 Key Insight

The system identifies profitable strategies based on market sentiment and provides actionable recommendations.

---

## 🚀 Future Improvements

* Real-time data integration
* API-based pipeline
* Live dashboard

---
