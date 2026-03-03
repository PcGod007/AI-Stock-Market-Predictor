import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster, toast } from 'react-hot-toast';
import { Activity, BarChart2, Table2, TrendingUp, Brain, Newspaper, ChevronDown, SlidersHorizontal } from 'lucide-react';

import Navbar from './components/Navbar';
import ControlPanel from './components/ControlPanel';
import StockTable from './components/StockTable';
import ComparisonTable from './components/ComparisonTable';
import NewsPanel from './components/NewsPanel';
import { MAChart, PredictionChart, OverviewChart } from './components/ChartView';
import { fetchStockData, fetchMovingAverage, fetchPrediction, fetchNews } from './services/api';
import StockLogo3D from './components/StockLogo3D';

const TABS = [
  { id: 'overview', label: 'Overview', icon: Activity },
  { id: 'table', label: 'Stock Data', icon: Table2 },
  { id: 'ma', label: 'Moving Avg', icon: TrendingUp },
  { id: 'prediction', label: 'Prediction', icon: Brain },
  { id: 'comparison', label: 'Comparison', icon: BarChart2 },
  { id: 'news', label: 'News', icon: Newspaper },
];

function StatBadge({ label, value, sub, color }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${color || ''}`}>{value}</span>
      {sub && <span className={`stat-change ${sub.startsWith('+') ? 'up' : sub.startsWith('-') ? 'down' : ''}`}>{sub}</span>}
    </div>
  );
}

export default function App() {
  const [loading, setLoading] = useState(false);
  const [ticker, setTicker] = useState('');
  const [stockData, setStockData] = useState(null);
  const [maData, setMaData] = useState(null);
  const [predData, setPredData] = useState(null);
  const [newsData, setNewsData] = useState(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [maWindows, setMaWindows] = useState([100]);
  const [activeTab, setActiveTab] = useState('overview');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const hasData = !!stockData;
  const hasPred = !!predData;
  const hasMa = !!maData;

  /* ── Stats derived from stock data ── */
  const stats = stockData ? (() => {
    const d = stockData;
    const n = d.length;
    const last = d[n - 1];
    const prev = d[n - 2] || last;
    const pct = ((last.Close - prev.Close) / prev.Close * 100);
    const chgStr = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
    // 52W: only last 365 calendar days
    const cutoff = new Date(last.Date);
    cutoff.setFullYear(cutoff.getFullYear() - 1);
    const d52 = d.filter(r => new Date(r.Date) >= cutoff);
    const maxH = Math.max(...d52.map(r => r.High)).toFixed(2);
    const minL = Math.min(...d52.map(r => r.Low)).toFixed(2);
    const avgVol = (d.reduce((s, r) => s + Number(r.Volume), 0) / n);
    const volStr = avgVol >= 1e6 ? `${(avgVol / 1e6).toFixed(1)}M` : `${(avgVol / 1e3).toFixed(0)}K`;
    return { last, pct, chgStr, maxH, minL, volStr, n };
  })() : null;

  /* ── Handlers ── */
  const handleFetch = useCallback(async ({ ticker, start, end }) => {
    setLoading(true);
    setTicker(ticker);
    try {
      const res = await fetchStockData(ticker, start, end);
      if (res.error) { toast.error(res.error); return; }
      setStockData(res.data);
      setMaData(null);
      setPredData(null);
      setActiveTab('overview');
      toast.success(`Loaded ${res.data.length} rows for ${ticker}`);
    } catch (e) {
      toast.error(e?.response?.data?.error || 'Failed to fetch stock data.');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleShowMA = useCallback(async ({ ticker, start, end, windows }) => {
    setLoading(true);
    setMaWindows(windows);
    try {
      const res = await fetchMovingAverage(ticker, start, end, windows);
      if (res.error) { toast.error(res.error); return; }
      setMaData(res);
      setActiveTab('ma');
      toast.success(`Moving average${windows.length > 1 ? 's' : ''} computed (${windows.join(', ')}d)`);
    } catch (e) {
      toast.error(e?.response?.data?.error || 'Failed to fetch MA data.');
    } finally {
      setLoading(false);
    }
  }, []);

  const handlePredict = useCallback(async ({ ticker, start, end }) => {
    setLoading(true);
    setNewsLoading(true);
    try {
      const res = await fetchPrediction(ticker, start, end);
      if (res.error) { toast.error(res.error); return; }
      setPredData(res);
      // news_articles now come bundled with the prediction response
      setNewsData(res.news_articles || []);
      setActiveTab('prediction');
      toast.success('AI prediction complete!');
    } catch (e) {
      toast.error(e?.response?.data?.error || 'Prediction failed — ensure the Flask API is running.');
    } finally {
      setLoading(false);
      setNewsLoading(false);
    }
  }, []);

  /* ── Tab content ── */
  const tabContent = () => {
    if (!hasData && activeTab !== 'overview') return (
      <div className="empty-state">
        <div className="empty-icon">📡</div>
        <div className="empty-title">No data loaded</div>
        <p className="empty-sub">Enter a ticker and click "Fetch Stock Data" to begin.</p>
      </div>
    );

    switch (activeTab) {
      case 'overview':
        return hasData ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <OverviewChart data={stockData} />
            <StockTable data={stockData.slice(-200)} />
          </div>
        ) : (
          <div className="empty-state">
            <StockLogo3D
              size={160}
              modelScale={1.0}
              autoRotate={true}
              drag={true}
              style={{ borderRadius: '16px', marginBottom: '1rem' }}
            />
            <div className="empty-title" style={{ fontSize: '1.15rem' }}>Welcome to StockAI</div>
            <p className="empty-sub">Enter a stock ticker and date range in the sidebar to get started.</p>
          </div>
        );

      case 'table':
        return <StockTable data={stockData} />;

      case 'ma':
        return hasMa ? (
          <MAChart data={maData} windows={maWindows} />
        ) : (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <div className="empty-title">No MA data</div>
            <p className="empty-sub">Select windows and click "Show MA Chart" in the sidebar.</p>
          </div>
        );

      case 'prediction':
        return hasPred ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <PredictionChart data={predData} />
            {/* ─── Reasoning Section ─── */}
            {predData.reasoning && (() => {
              const v = predData.reasoning.verdict;
              const verdictColor = v === 'bullish' ? '#3fb950' : v === 'neutral' ? '#d29922' : '#f85149';
              const verdictBg = v === 'bullish' ? 'rgba(63,185,80,.08)' : v === 'neutral' ? 'rgba(210,153,34,.08)' : 'rgba(248,81,73,.08)';
              const verdictBorder = v === 'bullish' ? 'rgba(63,185,80,.2)' : v === 'neutral' ? 'rgba(210,153,34,.2)' : 'rgba(248,81,73,.2)';
              const verdictIcon = v === 'bullish' ? '🟢' : v === 'neutral' ? '🟡' : '🔴';
              const pct = predData.reasoning.pct_change;
              const pctColor = pct >= 0 ? '#3fb950' : '#f85149';
              return (
                <div style={{ padding: '1.5rem', borderRadius: '12px', background: verdictBg, border: `1px solid ${verdictBorder}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '.6rem', marginBottom: '1rem' }}>
                    <span style={{ fontSize: '1.3rem' }}>{verdictIcon}</span>
                    <h3 style={{ margin: 0, color: '#e6edf3', fontSize: '1.05rem' }}>
                      AI Verdict: <span style={{ color: verdictColor, textTransform: 'capitalize' }}>
                        {v}
                      </span>
                    </h3>
                    <span style={{
                      background: pct >= 0 ? 'rgba(63,185,80,.15)' : 'rgba(248,81,73,.15)',
                      color: pctColor,
                      border: `1px solid ${pctColor}44`,
                      borderRadius: '6px', padding: '2px 10px', fontWeight: 700, fontSize: '.9rem'
                    }}>
                      {pct >= 0 ? '+' : ''}{pct}%
                    </span>
                  </div>
                  <p style={{ margin: '0 0 1rem', color: '#8b949e', fontSize: '.92rem', lineHeight: 1.6 }}>
                    {predData.reasoning.summary}
                  </p>
                  {predData.reasoning.factors.length > 0 && (
                    <ul style={{ margin: 0, padding: '0 0 0 1.2rem', display: 'flex', flexDirection: 'column', gap: '.4rem' }}>
                      {predData.reasoning.factors.map((f, i) => (
                        <li key={i} style={{ color: '#c9d1d9', fontSize: '.88rem', lineHeight: 1.55 }}>{f}</li>
                      ))}
                    </ul>
                  )}
                  {predData.reasoning.indicators && (
                    <div style={{
                      marginTop: '1rem', paddingTop: '.75rem', borderTop: '1px solid rgba(56,139,253,.12)',
                      display: 'flex', flexWrap: 'wrap', gap: '.5rem'
                    }}>
                      {Object.entries(predData.reasoning.indicators).filter(([, v]) => v != null).map(([k, v]) => (
                        <span key={k} style={{
                          background: 'rgba(56,139,253,.08)', border: '1px solid rgba(56,139,253,.2)',
                          borderRadius: '6px', padding: '3px 10px', fontSize: '.8rem', color: '#8b949e'
                        }}>
                          <strong style={{ color: '#e6edf3' }}>{k.replace(/_/g, ' ')}:</strong> {typeof v === 'number' ? v.toFixed(2) : v}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )
            })()}
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🤖</div>
            <div className="empty-title">No prediction yet</div>
            <p className="empty-sub">Click "Run AI Prediction" in the sidebar to run the LSTM model.</p>
          </div>
        );

      case 'comparison':
        return hasPred ? (
          <ComparisonTable data={predData} />
        ) : (
          <div className="empty-state">
            <div className="empty-icon">🔬</div>
            <div className="empty-title">Run prediction first</div>
            <p className="empty-sub">The comparison table shows original vs predicted values.</p>
          </div>
        );

      case 'news':
        return hasPred ? (
          <NewsPanel articles={newsData} loading={newsLoading} />
        ) : (
          <div className="empty-state">
            <div className="empty-icon">📰</div>
            <div className="empty-title">Run AI Prediction first</div>
            <p className="empty-sub">News is fetched automatically when you run the AI Prediction.</p>
          </div>
        );

      default: return null;
    }
  };

  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          style: { background: '#0f1f38', color: '#e6edf3', border: '1px solid rgba(56,139,253,.2)', fontFamily: 'Inter' },
          success: { iconTheme: { primary: '#3fb950', secondary: '#0f1f38' } },
          error: { iconTheme: { primary: '#f85149', secondary: '#0f1f38' } },
        }}
      />
      <Navbar ticker={ticker} dataLoaded={hasData} />

      <div className={`app-layout${sidebarOpen ? ' sidebar-open' : ''}`}>
        {/* ── Mobile Sidebar Toggle ── */}
        <button
          className="sidebar-toggle"
          onClick={() => setSidebarOpen(o => !o)}
          aria-label="Toggle controls"
        >
          <SlidersHorizontal size={15} />
          Controls
          <ChevronDown size={14} className="sidebar-toggle-icon" style={{ marginLeft: 'auto', transform: sidebarOpen ? 'rotate(180deg)' : 'none', transition: 'transform .3s' }} />
        </button>

        {/* ── Sidebar ── */}
        <ControlPanel
          onFetch={handleFetch}
          onShowMA={handleShowMA}
          onPredict={handlePredict}
          loading={loading}
          hasData={hasData}
        />

        {/* ── Main ── */}
        <main className="main-content">
          {/* Hero */}
          <div className="hero">
            <h1 className="hero-title">
              {hasData ? (
                <>Stock Analysis · <span className="hero-ticker">{ticker}</span></>
              ) : 'AI Stock Market Predictor'}
            </h1>
            <p className="hero-sub">
              {hasData
                ? `${stats.n.toLocaleString()} trading days · powered by LSTM deep learning`
                : 'Powered by LSTM deep learning · real-time data via Yahoo Finance'}
            </p>
          </div>

          {/* Stats */}
          {hasData && (
            <motion.div
              className="stats-row"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: .4 }}
            >
              <StatBadge
                label="Latest Close"
                value={`$${Number(stats.last.Close).toFixed(2)}`}
                sub={stats.chgStr}
                color={stats.pct >= 0 ? 'green' : 'red'}
              />
              <StatBadge label="52W High" value={`$${stats.maxH}`} color="green" />
              <StatBadge label="52W Low" value={`$${stats.minL}`} color="red" />
              <StatBadge label="Avg Volume" value={stats.volStr} color="blue" />
            </motion.div>
          )}

          {/* Card with tabs */}
          <motion.div
            className="card"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: .4, delay: .1 }}
          >
            {/* Tab bar */}
            <div className="tabs-row">
              {TABS.map(t => {
                const Icon = t.icon;
                const disabled =
                  !hasData && t.id !== 'overview' ||
                  (t.id === 'comparison' && !hasPred) ||
                  (t.id === 'prediction' && !hasPred);
                return (
                  <button
                    key={t.id}
                    className={`tab-btn ${activeTab === t.id ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
                    onClick={() => !disabled && setActiveTab(t.id)}
                    style={{ opacity: disabled ? .4 : 1, cursor: disabled ? 'not-allowed' : 'pointer' }}
                  >
                    <Icon size={14} />{t.label}
                  </button>
                );
              })}
            </div>

            {/* Tab body */}
            <div className="card-body">
              {loading
                ? <div className="spinner-wrap">
                  <div className="spinner" />
                  <span className="spinner-text">Crunching numbers…</span>
                </div>
                : <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: .22 }}
                  >
                    {tabContent()}
                  </motion.div>
                </AnimatePresence>
              }
            </div>
          </motion.div>
        </main>
      </div>
    </>
  );
}
