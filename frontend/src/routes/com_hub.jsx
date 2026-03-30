import React from 'react';
import './com_hub.css';

function search() {
    console.log('searching')
}

function create() {
    console.log('creating')
}

function ComHub() {
    let total_posts = 0;
    let active_today = 0;
    let scam_alerts = 0;
    let com_tips = 0;

    const top_left_title = (
    <header className="dashboard-header">
        <h1>Community Hub</h1>
        <h5>Share knowledge and stay informed about the latest scam methods</h5>
    </header>
    );

    const top_right_buttons = (
        <div className='top-left-buttons'>
            <button className='report_scam' onClick={() => create()}>Create Post</button>
        </div>
    );

    const boxes = (
        <div className='dash-boxes'>
            <div>
                <h3>Total Posts</h3>
                <h2>{total_posts}</h2>
            </div>
            <div>
                <h3>Active Today</h3>
                <h2>{active_today}</h2>
            </div>
            <div>
                <h3>Scam Alerts</h3>
                <h2>{scam_alerts}</h2>
            </div>
            <div>
                <h3>Community Tips</h3>
                <h2>{com_tips}</h2>
            </div>
        </div>
    );

    const post_searchbar = (
        <div className='post-searchbar'>
            <div className='add-input'>
                <h6>Search</h6>
                <input type="text" placeholder='Search by address or description' />
            </div>

            <div className='category-select'>
                <h6>Category</h6>
                <select id="category">
                    <option value="all_c">All Categories</option>
                    <option value="scam_alerts">Scam Alerts</option>
                    <option value="tips">Community Tips</option>
                    <option value="discussions">Discussions</option>
                    <option value="news">News</option>
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
            <h2>No Posts yet</h2>
        </div>
    );

    return (
        <div className='com-hub'>
            {top_left_title}
            {top_right_buttons}
            {boxes}
            {post_searchbar}
            {footer}
        </div>
    );
}

export default ComHub;

