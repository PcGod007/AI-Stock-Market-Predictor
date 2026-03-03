import StockLogo3D from './StockLogo3D';

export default function Navbar({ ticker, dataLoaded }) {
    return (
        <nav className="navbar">
            <div className="navbar-brand">
                <div className="logo-icon">
                    <StockLogo3D
                        size={36}
                        modelScale={0.8}
                        autoRotate={true}
                        drag={true}
                    />
                </div>
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
