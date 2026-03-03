import axios from 'axios';

// In dev: Vite proxy forwards /api → Flask (see vite.config.js)
// In production: set VITE_API_URL to your deployed backend (e.g. Render.com)
const BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

export const fetchStockData = (ticker, start, end) =>
  axios.get(`${BASE}/stock-data`, { params: { ticker, start, end } }).then(r => r.data);

export const fetchMovingAverage = (ticker, start, end, windows) =>
  axios.get(`${BASE}/moving-average`, { params: { ticker, start, end, windows: windows.join(',') } }).then(r => r.data);

export const fetchPrediction = (ticker, start, end) =>
  axios.get(`${BASE}/predict`, { params: { ticker, start, end } }).then(r => r.data);
