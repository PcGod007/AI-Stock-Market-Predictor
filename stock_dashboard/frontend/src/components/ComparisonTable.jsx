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
            <div className="table-container" style={{ maxHeight: '52vh', overflowY: 'auto' }}>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Original ($)</th>
                            <th>Predicted ($)</th>
                            <th>Error ($)</th>
                            <th>Error %</th>
                        </tr>
                    </thead>
                    <tbody>
                        <AnimatePresence mode="wait">
                            {slice.map((r, i) => (
                                <motion.tr
                                    key={r.date}
                                    initial={{ opacity: 0, x: -6 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * .01, duration: .16 }}
                                >
                                    <td className="td-mono" style={{ color: 'var(--text-secondary)' }}>{r.date}</td>
                                    <td className="td-mono">{fmt(r.original)}</td>
                                    <td className="td-mono" style={{ color: 'var(--accent-blue)' }}>{fmt(r.predicted)}</td>
                                    <td className={`td-mono ${r.error >= 0 ? 'td-up' : 'td-down'}`}>
                                        {r.error >= 0 ? '+' : ''}{fmt(r.error)}
                                    </td>
                                    <td>
                                        <span className={`badge ${Math.abs(r.pct) < 3 ? 'badge-green' : Math.abs(r.pct) < 6 ? 'badge-blue' : 'badge-red'}`}>
                                            {r.pct >= 0 ? '+' : ''}{r.pct.toFixed(2)}%
                                        </span>
                                    </td>
                                </motion.tr>
                            ))}
                        </AnimatePresence>
                    </tbody>
                </table>
            </div>
        </div>
    );
}
