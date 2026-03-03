import { useState } from 'react';
import { Search, Calendar, TrendingUp, BarChart2, Activity, Table, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

const MA_OPTIONS = [
    { label: '100d', value: 100 },
    { label: '200d', value: 200 },
    { label: '250d', value: 250 },
];

export default function ControlPanel({ onFetch, onShowMA, onPredict, loading, hasData }) {
    const today = new Date().toISOString().slice(0, 10);
    const twentyYrsAgo = `${new Date().getFullYear() - 20}-01-01`;

    const [ticker, setTicker] = useState('GOOG');
    const [start, setStart] = useState(twentyYrsAgo);
    const [end, setEnd] = useState(today);
    const [selMAs, setSelMAs] = useState([100]);
    const [activeView, setActiveView] = useState(null);

    const toggleMA = (v) =>
        setSelMAs(prev => prev.includes(v) ? prev.filter(x => x !== v) : [...prev, v]);

    const handleFetch = () => {
        setActiveView('table');
        onFetch({ ticker, start, end });
    };

    const handleMA = () => {
        setActiveView('ma');
        onShowMA({ ticker, start, end, windows: selMAs });
    };

    const handlePredict = () => {
        setActiveView('predict');
        onPredict({ ticker, start, end });
    };

    const isDisabled = loading || !ticker.trim();

    return (
        <aside className="sidebar">
            {/* ── Stock Input ── */}
            <div className="sidebar-section">
                <span className="sidebar-label">📡 Data Source</span>

                <div className="form-group">
                    <label className="form-label">Stock Ticker</label>
                    <div className="input-wrapper">
                        <Search className="input-icon" size={14} />
                        <input
                            className="form-input"
                            value={ticker}
                            onChange={e => setTicker(e.target.value.toUpperCase())}
                            placeholder="e.g. GOOG, AAPL, TSLA"
                            onKeyDown={e => e.key === 'Enter' && !isDisabled && handleFetch()}
                        />
                    </div>
                </div>

                <div className="form-group">
                    <label className="form-label">Start Date</label>
                    <div className="input-wrapper">
                        <Calendar className="input-icon" size={14} />
                        <input type="date" className="form-input" value={start} onChange={e => setStart(e.target.value)} />
                    </div>
                </div>

                <div className="form-group">
                    <label className="form-label">End Date</label>
                    <div className="input-wrapper">
                        <Calendar className="input-icon" size={14} />
                        <input type="date" className="form-input" value={end} onChange={e => setEnd(e.target.value)} />
                    </div>
                </div>

                <motion.button
                    className="btn btn-primary btn-full"
                    onClick={handleFetch}
                    disabled={isDisabled}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: .97 }}
                >
                    {loading && activeView === 'table'
                        ? <><RefreshCw size={14} className="spin-icon" /> Fetching…</>
                        : <><Activity size={14} /> Fetch Stock Data</>}
                </motion.button>
            </div>

            <div className="divider" />

            {/* ── Moving Average ── */}
            <div className="sidebar-section">
                <span className="sidebar-label">📊 Moving Averages</span>
                <div className="ma-chips">
                    {MA_OPTIONS.map(opt => (
                        <button
                            key={opt.value}
                            className={`ma-chip ${selMAs.includes(opt.value) ? 'active' : ''}`}
                            onClick={() => toggleMA(opt.value)}
                        >
                            {opt.label}
                        </button>
                    ))}
                </div>
                <motion.button
                    className="btn btn-outline btn-full"
                    onClick={handleMA}
                    disabled={isDisabled || !hasData}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: .97 }}
                    style={{ marginTop: '.25rem' }}
                >
                    {loading && activeView === 'ma'
                        ? <><RefreshCw size={14} className="spin-icon" /> Loading…</>
                        : <><TrendingUp size={14} /> Show MA Chart</>}
                </motion.button>
            </div>

            <div className="divider" />

            {/* ── Prediction ── */}
            <div className="sidebar-section">
                <span className="sidebar-label">🤖 AI Prediction</span>
                <p style={{ fontSize: '.78rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                    Uses the trained LSTM model to predict close prices on the test split (last 30%).
                </p>
                <motion.button
                    className="btn btn-primary btn-full"
                    onClick={handlePredict}
                    disabled={isDisabled || !hasData}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: .97 }}
                    style={{ background: 'linear-gradient(135deg,#3fb950,#238636)' }}
                >
                    {loading && activeView === 'predict'
                        ? <><RefreshCw size={14} className="spin-icon" /> Predicting…</>
                        : <>🧠 Run AI Prediction</>}
                </motion.button>
            </div>

            <div className="divider" />

            {/* ── View Toggle ── */}
            {hasData && (
                <div className="sidebar-section">
                    <span className="sidebar-label">📋 Views</span>
                    <div className="btn-group">
                        <button className={`btn-tab ${activeView === 'table' ? 'active' : ''}`}
                            onClick={() => setActiveView('table')}>
                            <Table size={13} /> Stock Data
                        </button>
                        <button className={`btn-tab ${activeView === 'ma' ? 'active' : ''}`}
                            onClick={() => setActiveView('ma')}>
                            <BarChart2 size={13} /> MA Chart
                        </button>
                    </div>
                </div>
            )}
        </aside>
    );
}
