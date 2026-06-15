#!/bin/bash
# Setup script for Elnatse-3
# Run this once to initialize the project

set -e  # Exit on error

echo "🚀 Elnatse-3 Setup Script"
echo "=========================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment (optional)
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Dependencies installed!"

echo ""
echo "📂 Project structure:"
echo "   ├── smhi_weather_harvest.py    ← Fetch weather data"
echo "   ├── fetch_svk_data.py          ← Fetch grid data"
echo "   ├── feature.py                 ← Engineer features"
echo "   ├── train.py                   ← Train model"
echo "   ├── live_detect.py             ← Run dashboard"
echo "   └── requirements.txt           ← Dependencies"

echo ""
echo "🔄 Next steps:"
echo "   1. Fetch data:"
echo "      python smhi_weather_harvest.py"
echo "      python fetch_svk_data.py"
echo ""
echo "   2. Process features:"
echo "      python feature.py"
echo ""
echo "   3. Train model:"
echo "      python train.py"
echo ""
echo "   4. Run dashboard:"
echo "      streamlit run live_detect.py"
echo ""
echo "📚 For details, see README.md"
echo ""
