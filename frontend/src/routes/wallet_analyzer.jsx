import React, { useState, useEffect } from 'react';
import { API_BASE } from '../shared/api/config';
import './wallet_analyzer.css';

function WalletAnalyzer() {
    const [address, setAddress] = useState("");
    const [blockchain, setBlockchain] = useState("ethereum");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [recentScans, setRecentScans] = useState([]);

    useEffect(() => {
        fetchRecent();
    }, []);

    async function fetchRecent() {
        try {
            const res = await fetch(`${API_BASE}/recent-wallet-scans?limit=10`);
            const data = await res.json();
            setRecentScans(data);
        } catch (err) {
            console.error("Failed to load recent wallet scans:", err);
        }
    }

    async function analyze() {
        if (!address) {
            alert("Please enter wallet address");
            return;
        }
        setLoading(true);
        setResult(null);
        try {
            const response = await fetch(`${API_BASE}/analyze-wallet`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ address: address, blockchain: blockchain }),
            });
            const data = await response.json();
            setResult(data);
            // Refresh recent scans after a new analysis is stored
            fetchRecent();
        } catch (error) {
            console.error("Error, failed to analyze wallet: ", error);
            alert("Failed to analyze wallet");
        } finally {
            setLoading(false);
        }
    }

    function formatDate(iso) {
        const d = new Date(iso);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function truncateAddr(addr) {
        if (!addr || addr.length <= 14) return addr;
        return addr.slice(0, 8) + '...' + addr.slice(-6);
    }

    const top_left_title = (
        <header className="dashboard-header">
            <h1>Wallet Analyzer</h1>
            <h5>AI-powered transaction analysis for suspicious patterns</h5>
        </header>
    );

    const address_checker = (
        <div className='address-checker'>
            <div className='add-input'>
                <h6>Wallet Address</h6>
                <input
                    type="text"
                    placeholder='Enter Wallet Address'
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                />
            </div>

            <div className='blockchain-select'>
                <h6>Blockchain</h6>
                <select id="Blockchain" value={blockchain} onChange={(e) => setBlockchain(e.target.value)}>
                    <option value="ethereum">Ethereum</option>
                    <option value="BNB_chain">BNB Chain</option>
                    <option value="Polygon">Polygon</option>
                    <option value="Solana">Solana</option>
                    <option value="Tron">Tron</option>
                    <option value="Litecoin">Litecoin</option>
                    <option value="Dogecoin">Dogecoin</option>
                    <option value="Ripple">Ripple</option>
                    <option value="Stellar">Stellar</option>
                    <option value="Cardano">Cardano</option>
                    <option value="EOS">EOS</option>
                    <option value="Tezos">Tezos</option>
                    <option value="Other">Other</option>
                </select>
            </div>

            <div className='analyze-button'>
                <h6>&nbsp;</h6>
                <button onClick={analyze} disabled={loading}>
                    {loading ? 'Analyzing...' : 'Analyze'}
                </button>
            </div>
        </div>
    );

    const result_display = result && (
        <div className={`analysis-result ${result.suspicious ? 'result-danger' : 'result-safe'}`}>
            <div className='result-header'>
                <h3>{result.suspicious ? 'Suspicious Activity Detected' : 'No Suspicious Activity'}</h3>
                <span className='result-badge'>{result.suspicious ? 'Suspicious' : 'Normal'}</span>
            </div>
            <div className='result-details'>
                <div>
                    <h6>Confidence Score</h6>
                    <h4>{Math.round(result.score * 100)}%</h4>
                </div>
                <div>
                    <h6>Address</h6>
                    <h4 className='truncate'>{result.address}</h4>
                </div>
                <div>
                    <h6>Blockchain</h6>
                    <h4>{result.blockchain}</h4>
                </div>
            </div>
        </div>
    );

    const recent_checks = (
        <div className='recent-checks'>
            <h4>Recent Analyses</h4>
            {recentScans.length === 0 ? (
                <p className='empty-state'>No analyses yet. Analyze a wallet to get started.</p>
            ) : (
                <div className='scan-list'>
                    {recentScans.map((scan) => (
                        <div key={scan.id} className={`scan-item ${scan.suspicious ? 'scan-suspicious' : 'scan-normal'}`}>
                            <div className='scan-address'>{truncateAddr(scan.address)}</div>
                            <div className='scan-chain'>{scan.blockchain}</div>
                            <div className='scan-score'>{Math.round(scan.score * 100)}%</div>
                            <div className={`scan-badge ${scan.suspicious ? 'badge-danger' : 'badge-safe'}`}>
                                {scan.suspicious ? 'Suspicious' : 'Normal'}
                            </div>
                            <div className='scan-date'>{formatDate(scan.scanned_at)}</div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );

    return (
        <div className='wallet-analyzer'>
            {top_left_title}
            {address_checker}
            {result_display}
            {recent_checks}
        </div>
    );
}

export default WalletAnalyzer;
