"""Horse Quant Platform -- quantitative betting system for UK horse racing.

Layer map (dependencies point downward only)::

    api            HTTP interface (FastAPI routers, middleware, error mapping)
    services       External integrations -- The Racing API client, notifiers
    data_pipeline  ETL: collect -> validate -> transform -> load
    ml             Feature engineering, training, calibration, inference
    strategies     Expected-value engine and staking rules
    backtesting    Historical simulation and performance reporting
    models         SQLAlchemy ORM entities
    schemas        Pydantic DTOs crossing layer boundaries
    database       Engine, sessions, declarative base
    utils          Config, logging, exceptions, time helpers
"""

__version__ = "0.1.0"
