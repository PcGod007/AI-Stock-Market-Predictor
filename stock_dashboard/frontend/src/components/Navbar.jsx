import { TrendingUp } from 'lucide-react';

export default function Navbar({ ticker, dataLoaded }) {
    return (
        <nav className="navbar">
            <div className="navbar-brand">
                <div className="logo-icon">📈</div>
                <span>Stock<em>AI</em></span>
            </div>
            <div className="navbar-status">
                {dataLoaded && (
                    <>
                        <div className="status-dot" />
                        <span>Live · {ticker.toUpperCase()}</span>
                    </>
                )}
            </div>
        </nav>
    );
}
