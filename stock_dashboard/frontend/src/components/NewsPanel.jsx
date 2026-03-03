/* NewsPanel.jsx — Displays real-time news with sentiment badges */
import { motion } from 'framer-motion';
import { ExternalLink, TrendingUp, TrendingDown, Minus } from 'lucide-react';

const SENTIMENT_STYLES = {
    bullish: { color: '#3fb950', bg: 'rgba(63,185,80,.1)', icon: TrendingUp, label: 'Bullish' },
    bearish: { color: '#f85149', bg: 'rgba(248,81,73,.1)', icon: TrendingDown, label: 'Bearish' },
    neutral: { color: '#8b949e', bg: 'rgba(139,148,158,.1)', icon: Minus, label: 'Neutral' },
};

function timeAgo(iso) {
    if (!iso) return '';
    const diff = (Date.now() - new Date(iso)) / 1000;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

export default function NewsPanel({ articles, loading }) {
    if (loading) {
        return (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#8b949e' }}>
                <div className="spinner" style={{ margin: '0 auto 1rem' }} />
                Fetching latest news…
            </div>
        );
    }

    if (!articles || articles.length === 0) {
        return (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#8b949e' }}>
                No news available for this ticker.
            </div>
        );
    }

    const sentimentCount = articles.reduce((acc, a) => {
        acc[a.sentiment] = (acc[a.sentiment] || 0) + 1;
        return acc;
    }, {});

    const overallSentiment = (sentimentCount.bullish || 0) > (sentimentCount.bearish || 0)
        ? 'bullish' : (sentimentCount.bearish || 0) > (sentimentCount.bullish || 0)
            ? 'bearish' : 'neutral';

    const st = SENTIMENT_STYLES[overallSentiment];
    const Icon = st.icon;

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Overall Market Sentiment */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: '.75rem',
                padding: '1rem 1.25rem', borderRadius: '10px',
                background: st.bg, border: `1px solid ${st.color}33`
            }}>
                <Icon size={20} color={st.color} />
                <div>
                    <div style={{ color: st.color, fontWeight: 700, fontSize: '1rem' }}>
                        News Sentiment: {st.label}
                    </div>
                    <div style={{ color: '#8b949e', fontSize: '.85rem' }}>
                        {sentimentCount.bullish || 0} bullish · {sentimentCount.bearish || 0} bearish · {sentimentCount.neutral || 0} neutral from last {articles.length} articles
                    </div>
                </div>
            </div>

            {/* Article Cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem' }}>
                {articles.map((a, i) => {
                    const s = SENTIMENT_STYLES[a.sentiment];
                    const ArticleIcon = s.icon;
                    return (
                        <motion.a
                            key={i}
                            href={a.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.05, duration: 0.2 }}
                            style={{
                                display: 'block', textDecoration: 'none', padding: '1rem 1.25rem',
                                borderRadius: '10px', border: `1px solid rgba(56,139,253,.12)`,
                                background: 'rgba(56,139,253,.03)', cursor: 'pointer',
                                transition: 'border-color .2s, background .2s',
                            }}
                            onMouseEnter={e => { e.currentTarget.style.borderColor = `${s.color}55`; e.currentTarget.style.background = s.bg; }}
                            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(56,139,253,.12)'; e.currentTarget.style.background = 'rgba(56,139,253,.03)'; }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                                <div style={{ flex: 1 }}>
                                    <div style={{ color: '#e6edf3', fontWeight: 600, fontSize: '.95rem', marginBottom: '.35rem', lineHeight: 1.4 }}>
                                        {a.title}
                                    </div>
                                    {a.description && (
                                        <div style={{
                                            color: '#8b949e', fontSize: '.85rem', lineHeight: 1.5,
                                            overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical'
                                        }}>
                                            {a.description}
                                        </div>
                                    )}
                                    <div style={{ marginTop: '.5rem', display: 'flex', alignItems: 'center', gap: '.75rem' }}>
                                        <span style={{
                                            background: s.bg, color: s.color, border: `1px solid ${s.color}44`,
                                            borderRadius: '6px', padding: '2px 8px', fontSize: '.78rem', fontWeight: 600,
                                            display: 'flex', alignItems: 'center', gap: '3px'
                                        }}>
                                            <ArticleIcon size={11} /> {s.label}
                                        </span>
                                        {a.source && <span style={{ color: '#8b949e', fontSize: '.8rem' }}>{a.source}</span>}
                                        <span style={{ color: '#8b949e', fontSize: '.8rem' }}>{timeAgo(a.published_at)}</span>
                                    </div>
                                </div>
                                <ExternalLink size={14} color="#8b949e" style={{ flexShrink: 0, marginTop: '3px' }} />
                            </div>
                        </motion.a>
                    );
                })}
            </div>
        </div>
    );
}
