import React, { useEffect, useState, useCallback } from 'react';
import { API_BASE } from '../shared/api/config';
import './com_hub.css';


function ComHub() {
    const [stats, setStats] = useState({
        total_posts: 0,
        active_today: 0,
        scam_alerts: 0,
        com_tips: 0,
    });
    const [posts, setPosts] = useState([]);
    const [filters, setFilters] = useState({ q: '', category: 'all_c' });
    const [showForm, setShowForm] = useState(false);
    const [form, setForm] = useState({ title: '', body: '', category: 'discussions' });

    const loadStats = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/community-posts/stats`);
            if (res.ok) setStats(await res.json());
        } catch (err) { console.error(err); }
    }, []);

    const loadPosts = useCallback(async () => {
        try {
            const params = new URLSearchParams();
            if (filters.q) params.set('q', filters.q);
            if (filters.category && filters.category !== 'all_c') params.set('category', filters.category);
            const res = await fetch(`${API_BASE}/community-posts?${params.toString()}`);
            if (res.ok) setPosts(await res.json());
        } catch (err) { console.error(err); }
    }, [filters]);

    useEffect(() => { loadStats(); loadPosts(); }, [loadStats, loadPosts]);

    async function submitPost(e) {
        e.preventDefault();
        try {
            const res = await fetch(`${API_BASE}/community-posts`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            });
            if (res.ok) {
                setForm({ title: '', body: '', category: 'discussions' });
                setShowForm(false);
                loadStats(); loadPosts();
            }
        } catch (err) { console.error(err); }
    }

    const top_left_title = (
        <header className="dashboard-header">
            <h1>Community Hub</h1>
            <h5>Share knowledge and stay informed about the latest scam methods</h5>
        </header>
    );

    const top_right_buttons = (
        <div className='top-left-buttons'>
            <button className='report_scam' onClick={() => setShowForm((s) => !s)}>
                {showForm ? 'Cancel' : 'Create Post'}
            </button>
        </div>
    );

    const boxes = (
        <div className='dash-boxes'>
            <div><h3>Total Posts</h3><h2>{stats.total_posts}</h2></div>
            <div><h3>Active Today</h3><h2>{stats.active_today}</h2></div>
            <div><h3>Scam Alerts</h3><h2>{stats.scam_alerts}</h2></div>
            <div><h3>Community Tips</h3><h2>{stats.com_tips}</h2></div>
        </div>
    );

    const postForm = showForm && (
        <form className='post-form' onSubmit={submitPost}>
            <input type='text' placeholder='Post title' required
                value={form.title}
                onChange={(e) => setForm({ ...form, title: e.target.value })} />
            <select value={form.category}
                onChange={(e) => setForm({ ...form, category: e.target.value })}>
                <option value='discussions'>Discussion</option>
                <option value='scam_alerts'>Scam Alert</option>
                <option value='tips'>Community Tip</option>
                <option value='news'>News</option>
            </select>
            <textarea placeholder='Write your post...'
                value={form.body}
                onChange={(e) => setForm({ ...form, body: e.target.value })} />
            <button type='submit'>Publish</button>
        </form>
    );

    const post_searchbar = (
        <div className='post-searchbar'>
            <div className='add-input'>
                <h6>Search</h6>
                <input type="text" placeholder='Search posts'
                    value={filters.q}
                    onChange={(e) => setFilters({ ...filters, q: e.target.value })} />
            </div>
            <div className='category-select'>
                <h6>Category</h6>
                <select value={filters.category}
                    onChange={(e) => setFilters({ ...filters, category: e.target.value })}>
                    <option value="all_c">All Categories</option>
                    <option value="scam_alerts">Scam Alerts</option>
                    <option value="tips">Community Tips</option>
                    <option value="discussions">Discussions</option>
                    <option value="news">News</option>
                </select>
            </div>
            <div className='search-button'>
                <h6>&nbsp;</h6>
                <button onClick={loadPosts}>Search</button>
            </div>
        </div>
    );

    const postList = (
        <div className='post-list'>
            {posts.length === 0 ? (
                <div className='footer'><h2>No Posts yet</h2></div>
            ) : (
                posts.map((p) => (
                    <div key={p.id} className='post-card'>
                        <div className='post-row'>
                            <h3>{p.title}</h3>
                            <span className={`post-badge cat-${p.category}`}>{p.category.replace('_', ' ')}</span>
                        </div>
                        {p.body && <p className='post-body'>{p.body}</p>}
                        <div className='post-meta'>
                            <span>{p.author || 'anonymous'}</span>
                            <span>{new Date(p.created_at).toLocaleString()}</span>
                        </div>
                    </div>
                ))
            )}
        </div>
    );

    return (
        <div className='com-hub'>
            {top_left_title}
            {top_right_buttons}
            {boxes}
            {postForm}
            {post_searchbar}
            {postList}
        </div>
    );
}

export default ComHub;
