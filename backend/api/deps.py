"""Reusable FastAPI dependencies.

Kept deliberately thin in Phase 1: the database session and the settings
singleton. Phase 2 adds the Racing API client, Phase 10 adds the authenticated
user dependency.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.utils.config import Settings, get_settings

SessionDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

__all__ = ["SessionDep", "SettingsDep"]
