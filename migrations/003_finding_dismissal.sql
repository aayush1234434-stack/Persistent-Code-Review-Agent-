ALTER TABLE review_finding_feedback
    DROP CONSTRAINT IF EXISTS review_finding_feedback_verdict_check;

ALTER TABLE review_finding_feedback
    ADD CONSTRAINT review_finding_feedback_verdict_check
    CHECK (verdict IN ('valid', 'invalid', 'dismissed'));
