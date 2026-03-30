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

-- Indexes for fast recent-first queries
CREATE INDEX IF NOT EXISTS idx_wallet_scans_scanned_at ON wallet_scans (scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_coin_scans_scanned_at   ON coin_scans   (scanned_at DESC);
