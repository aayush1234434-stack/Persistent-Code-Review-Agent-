ALTER TABLE pr_reviews
    ADD COLUMN IF NOT EXISTS source_sha TEXT,
    ADD COLUMN IF NOT EXISTS review_version INTEGER,
    ADD COLUMN IF NOT EXISTS webhook_delivery_id TEXT,
    ADD COLUMN IF NOT EXISTS lock_version BIGINT NOT NULL DEFAULT 0;

UPDATE pr_reviews
SET source_sha = NULLIF(pr_context->>'source_sha', '')
WHERE source_sha IS NULL;

WITH numbered AS (
    SELECT id,
           ROW_NUMBER() OVER (PARTITION BY repo, pr_number ORDER BY id) AS version
    FROM pr_reviews
    WHERE review_version IS NULL
)
UPDATE pr_reviews AS review
SET review_version = numbered.version
FROM numbered
WHERE review.id = numbered.id;

ALTER TABLE pr_reviews
    ALTER COLUMN review_version SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_pr_reviews_repo_pr_source_sha
    ON pr_reviews (repo, pr_number, source_sha)
    WHERE source_sha IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_pr_reviews_repo_pr_version
    ON pr_reviews (repo, pr_number, review_version);

CREATE INDEX IF NOT EXISTS idx_pr_reviews_source_sha
    ON pr_reviews (source_sha);

CREATE TABLE IF NOT EXISTS review_jobs (
    id BIGSERIAL PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    source_sha TEXT,
    event_action TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'retry', 'completed', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5 CHECK (max_attempts > 0),
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    locked_at TIMESTAMPTZ,
    locked_by TEXT,
    last_error TEXT,
    review_id BIGINT REFERENCES pr_reviews(id) ON DELETE SET NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_review_jobs_claim
    ON review_jobs (status, available_at, id);

CREATE INDEX IF NOT EXISTS idx_review_jobs_expired_lease
    ON review_jobs (locked_at)
    WHERE status = 'running';

CREATE INDEX IF NOT EXISTS idx_review_jobs_review_id
    ON review_jobs (review_id);

CREATE TABLE IF NOT EXISTS review_finding_feedback (
    review_id BIGINT NOT NULL REFERENCES pr_reviews(id) ON DELETE CASCADE,
    finding_index INTEGER NOT NULL CHECK (finding_index >= 0),
    verdict TEXT NOT NULL CHECK (verdict IN ('valid', 'invalid')),
    note TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (review_id, finding_index)
);
