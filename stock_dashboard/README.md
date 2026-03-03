# 📈 StockAI - Deep Learning Stock Market Predictor

A modern, high-performance web dashboard for stock market analysis and prediction, built with **React**, **Flask**, and **LSTM (Long Short-Term Memory)** neural networks. 

This project replaces the legacy Streamlit interface with a premium, interactive frontend featuring real-time data from Yahoo Finance and animated Plotly integrations.

---

## 📂 Project Structure

```text
stock_dashboard/
├── Latest_stock_price_model.keras  # Pre-trained LSTM model
├── netlify.toml                    # Deployment config for Netlify
├── README.md                       # This file
├── backend/                        # Python Flask API
│   ├── app.py                      # Main API server
│   └── requirements.txt            # Python dependencies
└── frontend/                       # React (Vite) Frontend
    ├── out/                        # PRODUCTION BUILD (Drop into Netlify)
    ├── src/                        # React source code
    │   ├── App.jsx                 # Dashboard logic & state
    │   ├── index.css               # Dark design system
    │   └── components/             # Reusable UI components
    └── vite.config.js              # Build & Proxy configuration
```

---

## 🚀 Getting Started (Local Development)

### 1. Backend Setup (Flask)
The backend handles data fetching from Yahoo Finance and runs the ML model.

```bash
cd backend
pip install -r requirements.txt
python app.py
```
*Backend will run at http://127.0.0.1:5000*

### 2. Frontend Setup (React)
The frontend provides the interactive dashboard.

```bash
cd frontend
npm install
npm run dev
```
*Frontend will run at http://localhost:5173*

---

## 🌐 Deployment to Netlify

This project is configured for seamless Netlify deployment.

### Method A: Manual (Drag & Drop)
1. Run `npm run build` inside the `frontend` folder.
2. Drag and drop the `stock_dashboard/frontend/out` folder into the Netlify "Deploys" area.

### Method B: Git Integration
1. Connect your GitHub repository to Netlify.
2. Netlify will detect `netlify.toml` and use the following settings:
   - **Base directory**: `stock_dashboard`
   - **Build command**: `cd frontend && npm run build`
   - **Publish directory**: `stock_dashboard/frontend/out`

---

## ✨ Features
- **Interactive Charts**: Moving Averages (100, 200, 250 days) and Candlestick overviews.
- **Deep Learning Predictions**: Visualizes Original vs. Predicted prices using LSTM.
- **Performance Metrics**: Calculates MAE, RMSE, and MAPE errors for model accuracy.
- **Modern UI**: Dark-themed glassmorphism design with responsive support.
- **Data Export**: Paginated and searchable stock data tables.

---

## 🛠️ Technical Stack
- **Frontend**: React (Hooks, Context), Vite, Framer Motion, Lucide Icons, Plotly.js.
- **Backend**: Flask, Flask-CORS.
- **Data/ML**: YFinance, NumPy, Pandas, Scikit-learn, TensorFlow/Keras.
