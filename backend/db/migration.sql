-- Run this in the Supabase SQL Editor to create the required tables.

-- Wallet analysis results
CREATE TABLE IF NOT EXISTS wallet_scans (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    address     TEXT NOT NULL,
    blockchain  TEXT NOT NULL DEFAULT 'ethereum',
    suspicious  BOOLEAN NOT NULL,
    score       REAL NOT NULL,
    scanned_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Coin / token analysis results
CREATE TABLE IF NOT EXISTS coin_scans (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    coin        TEXT NOT NULL,
    blockchain  TEXT NOT NULL DEFAULT 'ethereum',
    is_scam     BOOLEAN NOT NULL,
    label       TEXT NOT NULL,
    score       REAL NOT NULL,
    scanned_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Community-submitted scam reports
CREATE TABLE IF NOT EXISTS scam_reports (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    address       TEXT NOT NULL,
    blockchain    TEXT NOT NULL DEFAULT 'ethereum',
    description   TEXT NOT NULL DEFAULT '',
    amount_lost   NUMERIC NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'under_review', 'verified', 'rejected')),
    reporter      TEXT,
    reported_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Community hub posts
CREATE TABLE IF NOT EXISTS community_posts (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL DEFAULT 'discussions'
                 CHECK (category IN ('scam_alerts', 'tips', 'discussions', 'news')),
    author       TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for fast recent-first queries
CREATE INDEX IF NOT EXISTS idx_wallet_scans_scanned_at  ON wallet_scans     (scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_coin_scans_scanned_at    ON coin_scans       (scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_scam_reports_reported_at ON scam_reports     (reported_at DESC);
CREATE INDEX IF NOT EXISTS idx_community_posts_created  ON community_posts  (created_at DESC);
