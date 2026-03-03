import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

// Responsive chart height: smaller on mobile, full on desktop
const chartH = () => window.innerWidth <= 480 ? 260 : window.innerWidth <= 768 ? 330 : window.innerWidth <= 1024 ? 390 : 460;

const LAYOUT_BASE = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, sans-serif', color: '#8b949e', size: 12 },
    margin: { l: 60, r: 20, t: 20, b: 50 },
    xaxis: {
        gridcolor: 'rgba(56,139,253,.08)',
        linecolor: 'rgba(56,139,253,.15)',
        tickcolor: 'rgba(56,139,253,.15)',
        zerolinecolor: 'rgba(56,139,253,.1)',
    },
    yaxis: {
        gridcolor: 'rgba(56,139,253,.08)',
        linecolor: 'rgba(56,139,253,.15)',
        tickcolor: 'rgba(56,139,253,.15)',
        zerolinecolor: 'rgba(56,139,253,.1)',
        tickprefix: '$',
    },
    legend: {
        bgcolor: 'rgba(10,22,40,.7)',
        bordercolor: 'rgba(56,139,253,.2)',
        borderwidth: 1,
        font: { size: 11 },
    },
    hovermode: 'x unified',
    hoverlabel: {
        bgcolor: '#0a1628',
        bordercolor: 'rgba(56,139,253,.5)',
        font: { family: 'Inter, sans-serif', color: '#e6edf3', size: 12 },
    },
};

const CONFIG = {
    displayModeBar: true,
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
};

const MA_COLORS = {
    100: '#388bfd',
    200: '#bc8cff',
    250: '#d29922',
};

/* ──────────────── Moving Average Chart ──────────────────── */
export function MAChart({ data, windows }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!ref.current || !data) return;

        const traces = [
            {
                x: data.dates,
                y: data.close,
                name: 'Close Price',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#e6edf3', width: 1.5 },
                opacity: 0.6,
            },
            ...windows.map(w => ({
                x: data.dates,
                y: data.mas[String(w)],
                name: `MA ${w}d`,
                type: 'scatter',
                mode: 'lines',
                line: { color: MA_COLORS[w] || '#3fb950', width: 2 },
                connectgaps: false,
            })),
        ];

        Plotly.react(ref.current, traces, {
            ...LAYOUT_BASE,
            title: {
                text: `Close Price + Moving Average${windows.length > 1 ? 's' : ''}`,
                font: { family: 'Space Grotesk, sans-serif', size: 15, color: '#e6edf3' },
            },
        }, CONFIG);
    }, [data, windows]);

    return <div ref={ref} style={{ width: '100%', height: chartH() }} />;
}

/* ──────────────── Prediction Chart ─────────────────────── */
export function PredictionChart({ data }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!ref.current || !data) return;

        const traces = [
            {
                x: data.close_full.dates,
                y: data.close_full.values,
                name: 'Full Close Price',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#8b949e', width: 1.2 },
                opacity: 0.5,
            },
            {
                x: data.dates,
                y: data.original,
                name: 'Original (Test)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#388bfd', width: 2 },
            },
            {
                x: data.dates,
                y: data.predicted,
                name: 'AI Predicted (Test)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#3fb950', width: 2, dash: 'dot' },
                fill: 'tonexty',
                fillcolor: 'rgba(63,185,80,.06)',
            },
            {
                x: data.future_dates,
                y: data.future_predicted,
                name: 'Future Forecast (30d)',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#bc8cff', width: 3 },
                marker: { size: 4 },
                connectgaps: true,
            },
        ];

        Plotly.react(ref.current, traces, {
            ...LAYOUT_BASE,
            title: {
                text: 'Original vs Predicted Close Price (LSTM)',
                font: { family: 'Space Grotesk, sans-serif', size: 15, color: '#e6edf3' },
            },
        }, CONFIG);
    }, [data]);

    return <div ref={ref} style={{ width: '100%', height: 460 }} />;
}

/* ──────────────── Overview Candlestick Chart ────────────── */
export function OverviewChart({ data }) {
    const ref = useRef(null);

    useEffect(() => {
        if (!ref.current || !data || data.length === 0) return;

        const traces = [
            {
                x: data.map(r => r.Date),
                close: data.map(r => r.Close),
                open: data.map(r => r.Open),
                high: data.map(r => r.High),
                low: data.map(r => r.Low),
                type: 'candlestick',
                name: 'OHLC',
                increasing: { line: { color: '#3fb950' }, fillcolor: 'rgba(63,185,80,.6)' },
                decreasing: { line: { color: '#f85149' }, fillcolor: 'rgba(248,81,73,.6)' },
            },
        ];

        Plotly.react(ref.current, traces, {
            ...LAYOUT_BASE,
            title: {
                text: 'Price Overview (Candlestick)',
                font: { family: 'Space Grotesk, sans-serif', size: 15, color: '#e6edf3' },
            },
            xaxis: {
                ...LAYOUT_BASE.xaxis,
                rangeselector: {
                    buttons: [
                        { count: 1, label: '1M', step: 'month', stepmode: 'backward' },
                        { count: 3, label: '3M', step: 'month', stepmode: 'backward' },
                        { count: 6, label: '6M', step: 'month', stepmode: 'backward' },
                        { count: 1, label: '1Y', step: 'year', stepmode: 'backward' },
                        { count: 3, label: '3Y', step: 'year', stepmode: 'backward' },
                        { step: 'all', label: 'ALL' },
                    ],
                    bgcolor: 'rgba(15,31,56,.8)',
                    activecolor: 'rgba(56,139,253,.35)',
                    bordercolor: 'rgba(56,139,253,.2)',
                    borderwidth: 1,
                    font: { color: '#8b949e', size: 11 },
                },
                rangeslider: {
                    visible: true,
                    bgcolor: 'rgba(15,31,56,.6)',
                    bordercolor: 'rgba(56,139,253,.15)',
                    thickness: 0.07,
                },
                // Default view: ~last 1 year (252 trading days)
                range: [
                    data[Math.max(0, data.length - 252)].Date,
                    data[data.length - 1].Date
                ],
                type: 'date',
            },
        }, CONFIG);
    }, [data]);

    return <div ref={ref} style={{ width: '100%', height: chartH() }} />;
}
