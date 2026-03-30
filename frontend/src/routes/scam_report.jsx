import React from 'react';
import './scam_report.css';

function search() {
    console.log('searching')
}

function report() {
    console.log('reporting')
}

function ScamReport() {
    let total_reports = 0;
    let verified = 0;
    let under_review = 0;
    let total_lost = 0;

    const top_left_title = (
    <header className="dashboard-header">
        <h1>Community Scam Reports</h1>
        <h5>Help protect the community by reporting and verifying scams</h5>
    </header>
    );

    const top_right_buttons = (
        <div className='top-left-buttons'>
            <button className='report_scam' onClick={() => report()}>Report Scam</button>
        </div>
    );

    const boxes = (
        <div className='dash-boxes'>
            <div>
                <h3>Total Reports</h3>
                <h2>{total_reports}</h2>
            </div>
            <div>
                <h3>Verified</h3>
                <h2>{verified}</h2>
            </div>
            <div>
                <h3>Under Review</h3>
                <h2>{under_review}</h2>
            </div>
            <div>
                <h3>Total Lost</h3>
                <h2>{total_lost}</h2>
            </div>
        </div>
    );

    const scam_searchbar = (
        <div className='scam-searchbar'>
            <div className='add-input'>
                <h6>Token Address</h6>
                <input type="text" placeholder='Search by address or description' />
            </div>

            <div className='status-select'>
                <h6>Status</h6>
                <select id="status">
                    <option value="all_s">All Status</option>
                    <option value="verified">Verified</option>
                    <option value="under_review">Under Review</option>
                    <option value="pending">Pending</option>
                </select>
            </div>

            <div className='blockchain-select'>
                <h6>Blockchain</h6>
                <select id="types">
                    <option value="all_t">All Blockchains</option>
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

            <div className='search-button'>
                <h6>&nbsp;</h6>
                <button onClick={() => search()}>Search</button>
            </div>
        </div>
    )

    const footer = (
        <div className='footer'>
            <h2>No Reports Found</h2>
        </div>
    );

    return (
        <div className='scam-report'>
            {top_left_title}
            {top_right_buttons}
            {boxes}
            {scam_searchbar}
            {footer}
        </div>
    );
}

export default ScamReport;

