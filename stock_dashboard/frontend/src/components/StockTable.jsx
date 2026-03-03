import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ChevronLeft, ChevronRight } from 'lucide-react';

const PAGE_SIZE = 20;

function fmt(val, decimals = 2) {
    if (val == null || val === '') return '—';
    return Number(val).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function VolFmt({ val }) {
    const n = Number(val);
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K`;
    return n.toLocaleString();
}

export default function StockTable({ data }) {
    const [query, setQuery] = useState('');
    const [page, setPage] = useState(1);
    const [sortKey, setSortKey] = useState('Date');
    const [sortAsc, setSortAsc] = useState(false);

    const filtered = useMemo(() => {
        const q = query.toLowerCase();
        return data.filter(r => r.Date.includes(q));
    }, [data, query]);

    const sorted = useMemo(() => {
        return [...filtered].sort((a, b) => {
            const av = a[sortKey], bv = b[sortKey];
            return sortAsc
                ? (av < bv ? -1 : av > bv ? 1 : 0)
                : (av > bv ? -1 : av < bv ? 1 : 0);
        });
    }, [filtered, sortKey, sortAsc]);

    const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
    const page_ = Math.min(page, totalPages);
    const rows = sorted.slice((page_ - 1) * PAGE_SIZE, page_ * PAGE_SIZE);

    const handleSort = key => {
        if (sortKey === key) setSortAsc(a => !a);
        else { setSortKey(key); setSortAsc(true); }
    };

    const SortInd = ({ col }) => (
        <span style={{ marginLeft: 4, opacity: sortKey === col ? 1 : .3 }}>
            {sortKey === col && !sortAsc ? '↑' : '↓'}
        </span>
    );

    // Mini stats
    const closes = data.map(r => r.Close);
    const lastClose = closes[closes.length - 1];
    const prevClose = closes[closes.length - 2];
    const pctChange = prevClose ? ((lastClose - prevClose) / prevClose * 100) : 0;
    const isUp = pctChange >= 0;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="table-toolbar">
                <div className="search-wrapper">
                    <Search className="search-icon" size={14} />
                    <input
                        className="search-input"
                        placeholder="Filter by date…"
                        value={query}
                        onChange={e => { setQuery(e.target.value); setPage(1); }}
                    />
                </div>
                <div className="pagination">
                    <button className="page-btn" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page_ === 1}>
                        <ChevronLeft size={13} />
                    </button>
                    <span className="page-info">
                        {page_} / {totalPages} &nbsp;·&nbsp; {sorted.length} rows
                    </span>
                    <button className="page-btn" onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page_ === totalPages}>
                        <ChevronRight size={13} />
                    </button>
                </div>
            </div>

            <div className="table-container" style={{ maxHeight: '55vh', overflowY: 'auto' }}>
                <table className="data-table">
                    <thead>
                        <tr>
                            {['Date', 'Open', 'High', 'Low', 'Close', 'Volume'].map(col => (
                                <th key={col} onClick={() => handleSort(col)} style={{ cursor: 'pointer', userSelect: 'none' }}>
                                    {col}<SortInd col={col} />
                                </th>
                            ))}
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        <AnimatePresence mode="wait">
                            {rows.map((row, i) => {
                                const prev = sorted[(page_ - 1) * PAGE_SIZE + i - 1];
                                const chg = prev ? ((row.Close - prev.Close) / prev.Close * 100) : null;
                                return (
                                    <motion.tr
                                        key={row.Date}
                                        initial={{ opacity: 0, y: 4 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: i * .012, duration: .18 }}
                                    >
                                        <td className="td-mono" style={{ color: 'var(--text-secondary)' }}>{row.Date}</td>
                                        <td className="td-mono">{fmt(row.Open)}</td>
                                        <td className="td-mono td-up">{fmt(row.High)}</td>
                                        <td className="td-mono td-down">{fmt(row.Low)}</td>
                                        <td className="td-mono" style={{ fontWeight: 600 }}>{fmt(row.Close)}</td>
                                        <td className="td-mono" style={{ color: 'var(--text-muted)' }}>
                                            <VolFmt val={row.Volume} />
                                        </td>
                                        <td>
                                            {chg != null && (
                                                <span className={`badge ${chg >= 0 ? 'badge-green' : 'badge-red'}`}>
                                                    {chg >= 0 ? '+' : ''}{chg.toFixed(2)}%
                                                </span>
                                            )}
                                        </td>
                                    </motion.tr>
                                );
                            })}
                        </AnimatePresence>
                    </tbody>
                </table>
            </div>
        </div>
    );
}
