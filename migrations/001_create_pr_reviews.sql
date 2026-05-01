CREATE TABLE IF NOT EXISTS pr_reviews (
    id BIGSERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    pr_context JSONB,
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pr_reviews_repo_pr
    ON pr_reviews (repo, pr_number);

CREATE INDEX IF NOT EXISTS idx_pr_reviews_created_at
    ON pr_reviews (created_at DESC);
