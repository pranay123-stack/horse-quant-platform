"""Database layer tests (SQLite-backed -- no server required)."""

from __future__ import annotations

import pytest
from sqlalchemy import Integer, String, select
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import NAMING_CONVENTION, Base, TimestampMixin
from backend.database.session import (
    check_database_connection,
    dispose_engine,
    get_engine,
    get_session_factory,
    session_scope,
)

pytestmark = pytest.mark.unit


class _Widget(Base):
    """Throwaway model used only by these tests."""

    __tablename__ = "_test_widgets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)


@pytest.fixture
def prepared_engine(settings):
    engine = get_engine(settings)
    _Widget.__table__.create(bind=engine, checkfirst=True)
    yield engine
    _Widget.__table__.drop(bind=engine, checkfirst=True)


def test_naming_convention_is_applied():
    assert "pk" in NAMING_CONVENTION
    assert _Widget.__table__.primary_key.name == "pk__test_widgets"


def test_timestamp_mixin_declares_both_columns():
    class _Stamped(TimestampMixin):
        pass

    assert hasattr(_Stamped, "created_at")
    assert hasattr(_Stamped, "updated_at")


def test_engine_is_a_singleton(settings):
    assert get_engine(settings) is get_engine(settings)


def test_dispose_engine_resets_the_singleton(settings):
    first = get_engine(settings)
    dispose_engine()
    assert get_engine(settings) is not first


def test_connection_check_succeeds(settings):
    assert check_database_connection(settings) is True


def test_connection_check_fails_gracefully(monkeypatch, settings):
    from backend.utils.config import Settings

    broken = Settings(DATABASE_URL="postgresql+psycopg2://nobody:nobody@127.0.0.1:1/none")
    dispose_engine()
    assert check_database_connection(broken) is False
    dispose_engine()


def test_session_scope_commits(prepared_engine):
    with session_scope() as session:
        session.add(_Widget(name="committed"))

    with session_scope() as session:
        found = session.execute(select(_Widget).where(_Widget.name == "committed")).scalar_one()
        assert found.name == "committed"


def test_session_scope_rolls_back_on_error(prepared_engine):
    with pytest.raises(RuntimeError), session_scope() as session:
        session.add(_Widget(name="rolled-back"))
        raise RuntimeError("boom")

    with session_scope() as session:
        assert session.execute(select(_Widget).where(_Widget.name == "rolled-back")).first() is None


def test_to_dict_returns_column_values(prepared_engine):
    widget = _Widget(id=1, name="serialisable")
    assert widget.to_dict() == {"id": 1, "name": "serialisable"}
    assert widget.to_dict(exclude={"id"}) == {"name": "serialisable"}


def test_session_factory_is_reused(settings):
    assert get_session_factory(settings) is get_session_factory(settings)
