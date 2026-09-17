# Real-world PR corpus v1

`real_world_prs.v1.jsonl` contains 120 unique public GitHub pull requests with
human inline-review labels and changed diff context. It is a deterministic,
category-stratified sample of positive review examples from the public
[`ronantakizawa/github-codereview`](https://huggingface.co/datasets/ronantakizawa/github-codereview)
dataset at revision `c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca`.

Each case retains its original repository, PR number, direct GitHub URL, source
row, language, file, diff context, and human review. One case is selected per PR
to prevent a heavily discussed PR from dominating the score. Category labels
are deterministic keyword groupings of the human comments; `human_verified`
describes the review comment, not the derived category.

The upstream `comment_line` is retained as `metadata.source_comment_line`, but
is not used as a ground-truth diff line because it is relative to the dataset's
code window rather than a stable GitHub diff position. Default matching is by
category and file; judge-assisted runs can provide explicit `matched_label_ids`.

This is a sparse-label benchmark: a PR may contain valid issues beyond the one
retained human comment. An unmatched finding is counted as a benchmark false
positive, not asserted to be objectively wrong. Interpret that rate together
with production human acceptance and inspect unmatched findings before accepting
or rejecting a regression baseline.

The corpus is intentionally checked in so CI and historical runs are stable.
Regenerate it only as an explicit corpus-version change:

```bash
python benchmarks/import_real_world_corpus.py --limit 120
```

Source repository licenses still apply to the included snippets. The upstream
dataset declares its own dataset license as `other`; consult the upstream card
and repository licenses before redistributing the corpus outside this project.
