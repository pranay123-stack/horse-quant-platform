"""Declarative base and shared column mixins.

A strict naming convention is applied to every constraint and index. Without it
Alembic autogenerate produces unnamed constraints that cannot be dropped in a
downgrade, which makes migrations one-way -- unacceptable for a system that must
be reproducible from scratch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_N_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Base class for every ORM model in the platform."""

    metadata = metadata

    def to_dict(self, *, exclude: set[str] | None = None) -> dict[str, Any]:
        """Shallow dict of column values -- handy for logging and DataFrames."""
        skip = exclude or set()
        return {
            column.key: getattr(self, column.key)
            for column in self.__table__.columns
            if column.key not in skip
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pk = ", ".join(f"{col.name}={getattr(self, col.name)!r}" for col in self.__table__.primary_key)
        return f"<{type(self).__name__} {pk}>"


class TimestampMixin:
    """``created_at`` / ``updated_at`` maintained by the database itself."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Row insertion time (UTC, set by PostgreSQL).",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Last modification time (UTC).",
    )


__all__ = ["NAMING_CONVENTION", "Base", "TimestampMixin", "metadata"]
