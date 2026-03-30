import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './dashboard.css';

function Dashboard() {
    const navigate = useNavigate();

    const [stats, setStats] = useState({
        total_scans: 0,
        wallet_scans: 0,
        coin_scans: 0,
        high_risk_detected: 0,
    });
    const [recentWallets, setRecentWallets] = useState([]);

    useEffect(() => {
        fetchStats();
        fetchRecentWallets();
    }, []);

    async function fetchStats() {
        try {
            const res = await fetch("http://127.0.0.1:8000/stats");
            const data = await res.json();
            setStats(data);
        } catch (err) {
            console.error("Failed to load stats:", err);
        }
    }

    async function fetchRecentWallets() {
        try {
            const res = await fetch("http://127.0.0.1:8000/recent-wallet-scans?limit=5");
            const data = await res.json();
            setRecentWallets(data);
        } catch (err) {
            console.error("Failed to load recent scans:", err);
        }
    }

    function truncateAddr(addr) {
        if (!addr || addr.length <= 14) return addr;
        return addr.slice(0, 8) + '...' + addr.slice(-6);
    }

    function formatDate(iso) {
        const d = new Date(iso);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    const top_left_title = (
        <header className="dashboard-header">
            <h1>Security Dashboard</h1>
            <h5>Monitor threats and analyze Web3 transactions</h5>
        </header>
    );

    const top_right_buttons = (
        <div className='top-left-buttons'>
            <button className='analyze_wallet' onClick={() => navigate('/wallet-analyzer')}>Analyze Wallet</button>
            <button className='report_scam' onClick={() => navigate('/scam-report')}>Report Scam</button>
        </div>
    );

    const boxes = (
        <div className='dash-boxes'>
            <div>
                <h3>Total Scans</h3>
                <h2>{stats.total_scans}</h2>
            </div>
            <div>
                <h3>Wallet Scans</h3>
                <h2>{stats.wallet_scans}</h2>
            </div>
            <div>
                <h3>Coin Scans</h3>
                <h2>{stats.coin_scans}</h2>
            </div>
            <div>
                <h3>High Risk Detected</h3>
                <h2>{stats.high_risk_detected}</h2>
            </div>
        </div>
    );

    const quick_actions = (
        <div className='quick-actions'>
            <h2>Quick Actions</h2>
            <div className='wallet_risk_check' onClick={() => navigate('/wallet-analyzer')}>
                <h4>Wallet Risk Check</h4>
                <h6>Analyze any wallet address</h6>
            </div>
            <div className='token-analyzer' onClick={() => navigate('/coin-checker')}>
                <h4>Token Analyzer</h4>
                <h6>Check Coin Legitimacy</h6>
            </div>
            <div className='report-scam' onClick={() => navigate('/scam-report')}>
                <h4>Report a Scam</h4>
                <h6>Help the community</h6>
            </div>
        </div>
    );

    const recent_reported_scams = (
        <div className='recent-reported-scams'>
            <h4>Recent Wallet Scans</h4>
            {recentWallets.length === 0 ? (
                <p className='empty-state'>No scans yet.</p>
            ) : (
                <div className='dash-scan-list'>
                    {recentWallets.map((scan) => (
                        <div key={scan.id} className='dash-scan-item'>
                            <span className='dash-scan-addr'>{truncateAddr(scan.address)}</span>
                            <span className={`dash-scan-badge ${scan.suspicious ? 'badge-danger' : 'badge-safe'}`}>
                                {scan.suspicious ? 'Suspicious' : 'Normal'}
                            </span>
                            <span className='dash-scan-date'>{formatDate(scan.scanned_at)}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );

    const footer = (
        <div className='footer'>
            <h2>Stay Protected in Web3</h2>
            <h6>AI-powered wallet and token analysis with Supabase-backed history</h6>
            <button className="start-scanning" onClick={() => navigate('/wallet-analyzer')}>Start Scanning</button>
        </div>
    );

    return (
        <div className='dashboard'>
            {top_left_title}
            {top_right_buttons}
            {boxes}
            {quick_actions}
            {recent_reported_scams}
            {footer}
        </div>
    );
}

export default Dashboard;
