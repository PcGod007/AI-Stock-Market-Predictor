import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const PAGE_SIZE = 25;

function fmt(v) {
    return Number(v).toLocaleString('en-US', { minimumFractionDigits: 4, maximumFractionDigits: 4 });
}

export default function ComparisonTable({ data }) {
    const [page, setPage] = useState(1);
    const rows = data.dates.map((d, i) => ({
        date: d,
        original: data.original[i],
        predicted: data.predicted[i],
        error: data.predicted[i] - data.original[i],
        pct: ((data.predicted[i] - data.original[i]) / data.original[i] * 100),
    }));

    const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
    const page_ = Math.min(page, totalPages);
    const slice = rows.slice((page_ - 1) * PAGE_SIZE, page_ * PAGE_SIZE);

    const mae = useMemo(() => rows.reduce((s, r) => s + Math.abs(r.error), 0) / rows.length, [rows]);
    const rmse = useMemo(() => Math.sqrt(rows.reduce((s, r) => s + r.error ** 2, 0) / rows.length), [rows]);
    const mape = useMemo(() => rows.reduce((s, r) => s + Math.abs(r.pct), 0) / rows.length, [rows]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Accuracy metrics row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '.75rem' }}>
                {[
                    { label: 'MAE', value: `$${mae.toFixed(4)}`, color: 'var(--accent-blue)' },
                    { label: 'RMSE', value: `$${rmse.toFixed(4)}`, color: 'var(--accent-purple)' },
                    { label: 'MAPE', value: `${mape.toFixed(2)}%`, color: 'var(--accent-amber)' },
                ].map(m => (
                    <div key={m.label} className="stat-card" style={{ textAlign: 'center' }}>
                        <span className="stat-label">{m.label}</span>
                        <span className="stat-value" style={{ fontSize: '1.1rem', color: m.color }}>{m.value}</span>
                    </div>
                ))}
            </div>

            {/* Pagination */}
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div className="pagination">
                    <button className="page-btn" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page_ === 1}>
                        <ChevronLeft size={13} />
                    </button>
                    <span className="page-info">{page_} / {totalPages} · {rows.length} rows</span>
                    <button className="page-btn" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page_ === totalPages}>
                        <ChevronRight size={13} />
                    </button>
                </div>
            </div>

            {/* Table */}
            {/* Future Forecast Section (if stays on final page or via toggle) */}
            {data.future_dates && (
                <div style={{ marginTop: '1.5rem', border: '1px solid rgba(188,140,255, .2)', borderRadius: '12px', padding: '1.25rem', background: 'rgba(188,140,255, .03)' }}>
                    <h3 style={{ margin: '0 0 1rem 0', display: 'flex', alignItems: 'center', gap: '.5rem', color: '#bc8cff', fontSize: '1.1rem' }}>
                        🔮 Future Price Forecast (Next 30 Days)
                    </h3>
                    <div className="table-container" style={{ maxHeight: '40vh' }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Predicted Close ($)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.future_dates.map((d, i) => (
                                    <tr key={d}>
                                        <td className="td-mono" style={{ color: 'var(--text-secondary)' }}>{d}</td>
                                        <td className="td-mono" style={{ color: '#bc8cff', fontWeight: 'bold' }}>
                                            ${fmt(data.future_predicted[i])}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* ─── Educational Reasoning Section ─── */}
            <div style={{
                marginTop: '1.5rem',
                padding: '1.5rem',
                borderRadius: '12px',
                background: 'rgba(56,139,253,.05)',
                border: '1px solid rgba(56,139,253,.15)',
                lineHeight: '1.6'
            }}>
                <h3 style={{ margin: '0 0 1rem 0', color: '#e6edf3', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '.5rem' }}>
                    <span style={{ fontSize: '1.2rem' }}>💡</span> How the AI Prediction Works
                </h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                    <div>
                        <h4 style={{ color: '#388bfd', margin: '0 0 .5rem 0', fontSize: '.95rem' }}>1. Pattern Recognition (LSTM)</h4>
                        <p style={{ margin: 0, fontSize: '.9rem', color: '#8b949e' }}>
                            We use a <strong>Long Short-Term Memory (LSTM)</strong> neural network. Unlike simple models, it can "remember" long-term trends and short-term fluctuations simultaneously by analyzing the last 100 days of price data.
                        </p>
                    </div>

                    <div>
                        <h4 style={{ color: '#bc8cff', margin: '0 0 .5rem 0', fontSize: '.95rem' }}>2. Sliding Window Forecast</h4>
                        <p style={{ margin: 0, fontSize: '.9rem', color: '#8b949e' }}>
                            To predict 30 days ahead, the AI uses its own predictions as input for the next day. This creates a "sliding window" that projects momentum into the future based strictly on numerical patterns.
                        </p>
                    </div>

                    <div>
                        <h4 style={{ color: '#ff7b72', margin: '0 0 .5rem 0', fontSize: '.95rem' }}>3. Important Limitations</h4>
                        <p style={{ margin: 0, fontSize: '.9rem', color: '#8b949e' }}>
                            This is <strong>Technical Analysis</strong>. The AI does not know about news, earnings reports, or market events. Predictions become less certain the further they project into the future due to compound error.
                        </p>
                    </div>
                </div>

                <div style={{ marginTop: '1rem', borderTop: '1px solid rgba(56,139,253, .1)', paddingTop: '1rem', fontSize: '.85rem', color: '#8b949e', fontStyle: 'italic' }}>
                    Note: Predictions are for educational purposes only. Always perform your own due diligence before making financial decisions.
                </div>
            </div>
        </div>
    );
}
