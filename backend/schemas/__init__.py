"""Pydantic data-transfer objects.

These are the contracts crossing layer boundaries: raw Racing API payloads are
parsed into schemas *before* they are allowed near the database (Phase 2/4), and
ORM rows are serialised through schemas on the way out to HTTP (Phase 6+).
"""
