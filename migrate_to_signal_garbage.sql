-- One-time migration: rename bot→garbage framing across all tables
-- Run this against your Postgres database before deploying the new code.
--
-- Usage: psql -U jafar -d jafar -f migrate_to_signal_garbage.sql

BEGIN;

-- 1. Rename bot_judgments → signal_judgments
ALTER TABLE IF EXISTS bot_judgments RENAME TO signal_judgments;

-- 2. Rename columns in signal_judgments
ALTER TABLE signal_judgments RENAME COLUMN bot_probability TO garbage_probability;
ALTER TABLE signal_judgments RENAME COLUMN ml_bot_score TO ml_garbage_score;

-- 3. Drop the pipeline_run column (no longer used, judgments come from dashboard)
ALTER TABLE signal_judgments DROP COLUMN IF EXISTS pipeline_run;

-- 4. Update classification values: bot→garbage, likely_bot→likely_garbage, human→signal, likely_human→likely_signal
UPDATE signal_judgments SET classification = 'garbage' WHERE classification = 'bot';
UPDATE signal_judgments SET classification = 'likely_garbage' WHERE classification = 'likely_bot';
UPDATE signal_judgments SET classification = 'signal' WHERE classification = 'human';
UPDATE signal_judgments SET classification = 'likely_signal' WHERE classification = 'likely_human';

-- 5. Rename columns in account_scores
ALTER TABLE account_scores RENAME COLUMN bot_score TO garbage_score;

-- 6. Add is_anomaly column if it doesn't exist (may already be there)
ALTER TABLE account_scores ADD COLUMN IF NOT EXISTS is_anomaly BOOLEAN DEFAULT FALSE;

-- 7. Update human_labels: bot→garbage, human→signal
UPDATE human_labels SET label = 'garbage' WHERE label = 'bot';
UPDATE human_labels SET label = 'signal' WHERE label = 'human';

-- 8. Update blocked_accounts reasons that reference old "bot" framing
UPDATE blocked_accounts SET reason = REPLACE(reason, 'LLM judge: bot', 'LLM judge: garbage') WHERE reason LIKE '%LLM judge: bot%';
UPDATE blocked_accounts SET reason = REPLACE(reason, 'LLM judge: likely_bot', 'LLM judge: likely_garbage') WHERE reason LIKE '%LLM judge: likely_bot%';

-- 10. Drop the app_state table (no longer used)
DROP TABLE IF EXISTS app_state;

-- 11. Create new tables if they don't exist (dashboard create_all handles this,
--    but including here for completeness)
CREATE TABLE IF NOT EXISTS watched_accounts (
    username VARCHAR(255) PRIMARY KEY,
    reason TEXT,
    added_at TIMESTAMP DEFAULT NOW(),
    last_scraped_at TIMESTAMP,
    last_scraped_tweet_id BIGINT
);

CREATE TABLE IF NOT EXISTS blocked_accounts (
    username VARCHAR(255) PRIMARY KEY,
    reason TEXT,
    added_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR(20) PRIMARY KEY,
    started_at TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    step1_complete BOOLEAN DEFAULT FALSE,
    step2_complete BOOLEAN DEFAULT FALSE,
    step3_complete BOOLEAN DEFAULT FALSE,
    step4_complete BOOLEAN DEFAULT FALSE,
    step5_complete BOOLEAN DEFAULT FALSE,
    step6_complete BOOLEAN DEFAULT FALSE,
    topics_completed JSON DEFAULT '[]',
    trends JSON DEFAULT '[]',
    trends_completed JSON DEFAULT '[]',
    analysis TEXT DEFAULT '',
    signal_strength VARCHAR(20) DEFAULT '',
    is_notable BOOLEAN DEFAULT FALSE,
    top_engagement FLOAT DEFAULT 0.0,
    error TEXT DEFAULT '',
    last_updated TIMESTAMP DEFAULT NOW()
);

COMMIT;
