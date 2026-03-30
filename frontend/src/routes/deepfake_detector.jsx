import React, { useState, useRef } from 'react';
import './deepfake_detector.css';

function DeepfakeDetector() {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef(null);

    function handleFile(selectedFile) {
        if (!selectedFile) return;
        setFile(selectedFile);
        setResult(null);
        setError(null);

        // Generate preview for images and videos
        const url = URL.createObjectURL(selectedFile);
        const isVideo = selectedFile.type.startsWith('video/');
        setPreview({ url, isVideo, name: selectedFile.name, size: selectedFile.size });
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(true);
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
    }

    function handleInputChange(e) {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    }

    function clearFile() {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
        if (inputRef.current) inputRef.current.value = '';
    }

    async function analyze() {
        if (!file) {
            setError('Please upload a file first.');
            return;
        }
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('http://127.0.0.1:8000/detect-deepfake', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => null);
                throw new Error(errData?.detail || `Server error ${response.status}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error('Deepfake detection failed:', err);
            setError(err.message || 'Failed to analyze media.');
        } finally {
            setLoading(false);
        }
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function getScoreColor(score) {
        if (score <= 0.3) return '#22c55e';  // green - real
        if (score <= 0.6) return '#eab308';  // yellow - uncertain
        return '#ef4444';                     // red - fake
    }

    // ── JSX sections ─────────────────────────────────────────────────────

    const header = (
        <header className="dashboard-header">
            <h1>Deepfake Detection Engine</h1>
            <h5>AI-powered analysis to detect manipulated media content</h5>
        </header>
    );

    const uploadArea = (
        <div
            className={`upload-media ${dragActive ? 'drag-active' : ''} ${preview ? 'has-file' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => !preview && inputRef.current?.click()}
        >
            <input
                ref={inputRef}
                type="file"
                accept="image/*,video/*"
                onChange={handleInputChange}
                style={{ display: 'none' }}
            />

            {!preview ? (
                <div className="upload-prompt">
                    <div className="upload-icon">&#8682;</div>
                    <h2>Drop media here or click to upload</h2>
                    <h4>Supports images (PNG, JPG, WEBP) and videos (MP4, AVI, MOV)</h4>
                </div>
            ) : (
                <div className="file-preview">
                    <div className="preview-media">
                        {preview.isVideo ? (
                            <video src={preview.url} controls muted style={{ maxHeight: '200px', borderRadius: '8px' }} />
                        ) : (
                            <img src={preview.url} alt="Preview" style={{ maxHeight: '200px', borderRadius: '8px' }} />
                        )}
                    </div>
                    <div className="preview-info">
                        <p className="file-name">{preview.name}</p>
                        <p className="file-size">{formatFileSize(preview.size)}</p>
                    </div>
                    <div className="preview-actions">
                        <button className="analyze-btn" onClick={(e) => { e.stopPropagation(); analyze(); }} disabled={loading}>
                            {loading ? 'Analyzing...' : 'Analyze Media'}
                        </button>
                        <button className="clear-btn" onClick={(e) => { e.stopPropagation(); clearFile(); }} disabled={loading}>
                            Clear
                        </button>
                    </div>
                </div>
            )}
        </div>
    );

    const resultPanel = result && (
        <div className="result-panel">
            <h3>Analysis Results</h3>
            <div className="score-display">
                <div className="score-circle" style={{ borderColor: getScoreColor(result.fake_score) }}>
                    <span className="score-value" style={{ color: getScoreColor(result.fake_score) }}>
                        {Math.round(result.fake_score * 100)}%
                    </span>
                    <span className="score-subtitle">fake probability</span>
                </div>
                <div className="score-details">
                    <div className={`verdict ${result.label === 'Fake' ? 'verdict-fake' : 'verdict-real'}`}>
                        {result.label}
                    </div>
                    <p>Frames analysed: <strong>{result.frame_count}</strong></p>
                    <p>Confidence: <strong>{Math.round(result.fake_score > 0.5 ? result.fake_score * 100 : (1 - result.fake_score) * 100)}%</strong></p>
                </div>
            </div>
            {result.per_frame && result.per_frame.length > 1 && (
                <div className="frame-scores">
                    <h4>Per-frame scores</h4>
                    <div className="frame-bar-container">
                        {result.per_frame.map((s, i) => (
                            <div key={i} className="frame-bar" style={{ height: `${Math.max(s * 100, 2)}%`, backgroundColor: getScoreColor(s) }} title={`Frame ${i + 1}: ${Math.round(s * 100)}%`} />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );

    const errorDisplay = error && (
        <div className="error-banner">
            <p>{error}</p>
        </div>
    );

    const boxes = (
        <div className="dash-boxes">
            <div>
                <h2>Facial Analysis</h2>
                <h3>Detects unnatural face movements and blending artifacts</h3>
            </div>
            <div>
                <h2>Visual Inspection</h2>
                <h3>Analyzes lighting, shadows, and compression patterns</h3>
            </div>
            <div>
                <h2>Trust Verification</h2>
                <h3>Provides authenticity score with detailed findings</h3>
            </div>
        </div>
    );

    return (
        <div className="deepfake-detector">
            {header}
            {uploadArea}
            {errorDisplay}
            {resultPanel}
            {boxes}
        </div>
    );
}

export default DeepfakeDetector;
