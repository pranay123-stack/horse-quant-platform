"""HTTP layer tests: health endpoints, middleware, error mapping."""

from __future__ import annotations

import pytest

from backend.api.middleware import REQUEST_ID_HEADER, RESPONSE_TIME_HEADER

pytestmark = pytest.mark.unit


def test_root_banner(client, settings):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == settings.app_name
    assert body["api"] == settings.api_v1_prefix


def test_health_is_ok(client, settings):
    response = client.get(f"{settings.api_v1_prefix}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["uptime_seconds"] >= 0


def test_health_is_also_served_unprefixed(client):
    """Orchestrator probes use the stable, unversioned path."""
    assert client.get("/health").status_code == 200
    assert client.get("/health/live").status_code == 200


def test_version_endpoint(client, settings):
    response = client.get(f"{settings.api_v1_prefix}/health/version")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == settings.app_version
    assert body["python_version"].startswith("3.12")


def test_database_probe_reports_up(client, settings):
    response = client.get(f"{settings.api_v1_prefix}/health/db")
    assert response.status_code == 200
    assert response.json()["database"] == "up"


def test_database_probe_never_leaks_the_password(client, settings):
    response = client.get(f"{settings.api_v1_prefix}/health/db")
    assert settings.postgres_password.get_secret_value() not in response.text


def test_readiness_lists_components(client, settings):
    response = client.get(f"{settings.api_v1_prefix}/health/ready")
    assert response.status_code == 200
    names = {component["name"] for component in response.json()["components"]}
    assert names == {"database", "racing_api_credentials"}


def test_config_endpoint_masks_secrets(client, settings):
    response = client.get(f"{settings.api_v1_prefix}/meta/config")
    assert response.status_code == 200
    body = response.json()

    assert body["environment"] == "test"
    assert body["racing_api_configured"] is True
    assert body["betting_defaults"]["min_expected_value"] == settings.min_expected_value

    for secret in (
        settings.postgres_password.get_secret_value(),
        settings.racing_api_key.get_secret_value(),
        settings.secret_key.get_secret_value(),
    ):
        assert secret not in response.text


def test_request_id_header_is_generated(client):
    response = client.get("/health")
    assert REQUEST_ID_HEADER in response.headers
    assert len(response.headers[REQUEST_ID_HEADER]) == 12


def test_client_supplied_request_id_is_echoed(client):
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-me-1234"})
    assert response.headers[REQUEST_ID_HEADER] == "trace-me-1234"


def test_process_time_header_is_present(client):
    response = client.get("/health")
    assert float(response.headers[RESPONSE_TIME_HEADER]) >= 0


def test_access_log_line_carries_the_request_id(client, settings):
    """The completion log must be written *before* the correlation id is reset."""
    request_id = "corr-id-9911"
    client.get(f"{settings.api_v1_prefix}/health/version", headers={REQUEST_ID_HEADER: request_id})

    api_log = (settings.log_dir / "api.log").read_text(encoding="utf-8")
    completion_lines = [
        line for line in api_log.splitlines() if "request completed" in line and "health/version" in line
    ]
    assert completion_lines, "no access-log line was written"
    assert request_id in completion_lines[-1], "access log lost the correlation id"


def test_unknown_route_returns_the_standard_error_envelope(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"error", "message", "details", "request_id"}


def test_platform_errors_map_to_their_status_code(app, client, settings):
    from backend.utils.exceptions import RateLimitError

    @app.get("/_test/rate-limited")
    def _boom() -> None:
        raise RateLimitError("upstream throttled us", retry_after=30, endpoint="/v1/results")

    response = client.get("/_test/rate-limited")
    assert response.status_code == 429
    body = response.json()
    assert body["error"] == "rate_limit_error"
    assert body["details"]["retry_after"] == 30


def test_unhandled_errors_do_not_leak_internals(app, client):
    @app.get("/_test/explode")
    def _explode() -> None:
        raise RuntimeError("secret internal detail")

    response = client.get("/_test/explode")
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert response.json()["error"] == "internal_error"


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["version"]
    assert "/api/v1/health" in schema["paths"]
