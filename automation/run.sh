#!/bin/bash

echo "🚀 Starting Crypto Sentiment Pipeline..."

echo "📊 Running analysis..."
python src/analysis.py

echo "✅ Pipeline completed successfully"

echo "📁 Output saved to dashboard/analysis_results.json"