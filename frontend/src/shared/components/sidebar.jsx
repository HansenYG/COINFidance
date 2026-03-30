import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './sidebar.css';

function Sidebar(){
    const navigate = useNavigate();
    const location = useLocation();

    const menuItems = [
        { path: '/dashboard', label: 'Dashboard' },
        { path: '/wallet-analyzer', label: 'Wallet Analyzer' },
        { path: '/coin-checker', label: 'Coin Checker' },
        { path: '/scam-report', label: 'Scam Report' },
        { path: '/com-hub', label: 'Community Hub' },
        { path: '/deepfake-detector', label: 'Deepfake Detector' },
        { path: '/news', label: 'News' },
    ];

    const subscription = (
        <div className='subscription'>
            <h6>Pro Subscription</h6>
            <p>Advanced AI detection</p>
        </div>
    )

    const isActive = (path) => {
        if (path === '/dashboard') {
            return location.pathname === '/' || location.pathname === '/dashboard';
        }
        return location.pathname === path;
    };

    return (
        <div className="sidebar">
            <h3>COINFidance</h3>
            {menuItems.map((item) => (
                <div
                    key={item.path}
                    className={`sidebar-item ${isActive(item.path) ? 'active' : ''}`}
                    onClick={() => navigate(item.path)}
                >
                    <h3>{item.label}</h3>
                </div>
            ))}
        </div>
    );
}

export default Sidebar;