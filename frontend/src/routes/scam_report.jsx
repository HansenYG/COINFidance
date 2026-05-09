import React, { useEffect, useState, useCallback } from 'react';
import { API_BASE } from '../shared/api/config';
import './scam_report.css';


function ScamReport() {
    const [stats, setStats] = useState({
        total_reports: 0,
        verified: 0,
        under_review: 0,
        total_lost: 0,
    });
    const [reports, setReports] = useState([]);
    const [filters, setFilters] = useState({ q: '', status: 'all_s', blockchain: 'all_t' });
    const [showForm, setShowForm] = useState(false);
    const [form, setForm] = useState({
        address: '',
        blockchain: 'ethereum',
        description: '',
        amount_lost: 0,
    });

    const loadStats = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/scam-reports/stats`);
            if (res.ok) setStats(await res.json());
        } catch (err) { console.error(err); }
    }, []);

    const loadReports = useCallback(async () => {
        try {
            const params = new URLSearchParams();
            if (filters.q) params.set('q', filters.q);
            if (filters.status && filters.status !== 'all_s') params.set('status', filters.status);
            if (filters.blockchain && filters.blockchain !== 'all_t') params.set('blockchain', filters.blockchain);
            const res = await fetch(`${API_BASE}/scam-reports?${params.toString()}`);
            if (res.ok) setReports(await res.json());
        } catch (err) { console.error(err); }
    }, [filters]);

    useEffect(() => { loadStats(); loadReports(); }, [loadStats, loadReports]);

    async function submitReport(e) {
        e.preventDefault();
        try {
            const res = await fetch(`${API_BASE}/scam-reports`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...form, amount_lost: Number(form.amount_lost) || 0 }),
            });
            if (res.ok) {
                setForm({ address: '', blockchain: 'ethereum', description: '', amount_lost: 0 });
                setShowForm(false);
                loadStats(); loadReports();
            }
        } catch (err) { console.error(err); }
    }

    const top_left_title = (
        <header className="dashboard-header">
            <h1>Community Scam Reports</h1>
            <h5>Help protect the community by reporting and verifying scams</h5>
        </header>
    );

    const top_right_buttons = (
        <div className='top-left-buttons'>
            <button className='report_scam' onClick={() => setShowForm((s) => !s)}>
                {showForm ? 'Cancel' : 'Report Scam'}
            </button>
        </div>
    );

    const boxes = (
        <div className='dash-boxes'>
            <div><h3>Total Reports</h3><h2>{stats.total_reports}</h2></div>
            <div><h3>Verified</h3><h2>{stats.verified}</h2></div>
            <div><h3>Under Review</h3><h2>{stats.under_review}</h2></div>
            <div><h3>Total Lost</h3><h2>${Number(stats.total_lost).toLocaleString()}</h2></div>
        </div>
    );

    const reportForm = showForm && (
        <form className='scam-form' onSubmit={submitReport}>
            <input type='text' placeholder='Wallet / token address'
                value={form.address}
                onChange={(e) => setForm({ ...form, address: e.target.value })} required />
            <select value={form.blockchain}
                onChange={(e) => setForm({ ...form, blockchain: e.target.value })}>
                <option value='ethereum'>Ethereum</option>
                <option value='BNB_chain'>BNB Chain</option>
                <option value='Polygon'>Polygon</option>
                <option value='Solana'>Solana</option>
                <option value='Tron'>Tron</option>
                <option value='Other'>Other</option>
            </select>
            <input type='number' placeholder='Amount lost (USD)' min='0' step='0.01'
                value={form.amount_lost}
                onChange={(e) => setForm({ ...form, amount_lost: e.target.value })} />
            <textarea placeholder='Describe what happened...'
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })} />
            <button type='submit'>Submit Report</button>
        </form>
    );

    const scam_searchbar = (
        <div className='scam-searchbar'>
            <div className='add-input'>
                <h6>Search</h6>
                <input type="text" placeholder='Search by address or description'
                    value={filters.q}
                    onChange={(e) => setFilters({ ...filters, q: e.target.value })} />
            </div>
            <div className='status-select'>
                <h6>Status</h6>
                <select value={filters.status}
                    onChange={(e) => setFilters({ ...filters, status: e.target.value })}>
                    <option value="all_s">All Status</option>
                    <option value="verified">Verified</option>
                    <option value="under_review">Under Review</option>
                    <option value="pending">Pending</option>
                </select>
            </div>
            <div className='blockchain-select'>
                <h6>Blockchain</h6>
                <select value={filters.blockchain}
                    onChange={(e) => setFilters({ ...filters, blockchain: e.target.value })}>
                    <option value="all_t">All Blockchains</option>
                    <option value="ethereum">Ethereum</option>
                    <option value="BNB_chain">BNB Chain</option>
                    <option value="Polygon">Polygon</option>
                    <option value="Solana">Solana</option>
                    <option value="Tron">Tron</option>
                    <option value="Other">Other</option>
                </select>
            </div>
            <div className='search-button'>
                <h6>&nbsp;</h6>
                <button onClick={loadReports}>Search</button>
            </div>
        </div>
    );

    const reportList = (
        <div className='report-list'>
            {reports.length === 0 ? (
                <div className='footer'><h2>No Reports Found</h2></div>
            ) : (
                reports.map((r) => (
                    <div key={r.id} className='report-card'>
                        <div className='report-row'>
                            <span className='report-addr'>{r.address}</span>
                            <span className={`report-badge badge-${r.status}`}>{r.status}</span>
                        </div>
                        <div className='report-row report-meta'>
                            <span>{r.blockchain}</span>
                            <span>${Number(r.amount_lost).toLocaleString()}</span>
                            <span>{new Date(r.reported_at).toLocaleDateString()}</span>
                        </div>
                        {r.description && <p className='report-desc'>{r.description}</p>}
                    </div>
                ))
            )}
        </div>
    );

    return (
        <div className='scam-report'>
            {top_left_title}
            {top_right_buttons}
            {boxes}
            {reportForm}
            {scam_searchbar}
            {reportList}
        </div>
    );
}

export default ScamReport;
