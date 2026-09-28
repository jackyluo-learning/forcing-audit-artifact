"""Reproducible privacy-auditing pipeline.

Design invariant: a single per-attempt log (ResultsStore) is the only source of
truth. Record-level and per-field metrics are both derived from it, so they can
never disagree."""
