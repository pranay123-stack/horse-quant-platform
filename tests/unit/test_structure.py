"""Architecture guardrails.

These tests fail loudly if the agreed layout drifts -- a missing package, a
deleted config template, or (most importantly) a module reaching for
``os.environ`` instead of the settings object.
"""

from __future__ import annotations

import ast
import importlib
import re

import pytest

from backend.utils.config import PROJECT_ROOT

pytestmark = pytest.mark.unit

REQUIRED_PACKAGES = [
    "backend",
    "backend.api",
    "backend.api.v1",
    "backend.backtesting",
    "backend.data_quality",
    "backend.database",
    "backend.data_pipeline",
    "backend.features",
    "backend.ml",
    "backend.ml.calibration",
    "backend.ml.metrics",
    "backend.ml.models",
    "backend.ml.preprocessing",
    "backend.models",
    "backend.prediction_service",
    "backend.research",
    "backend.schemas",
    "backend.services",
    "backend.services.racing_api",
    "backend.strategies",
    "backend.strategy",
    "backend.utils",
]

REQUIRED_FILES = [
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "Makefile",
    "README.md",
    "alembic.ini",
    "docker-compose.yml",
    "migrations/env.py",
    "migrations/script.py.mako",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements.txt",
    "frontend/dashboard/index.html",
]

REQUIRED_DIRECTORIES = [
    "backend",
    "backend/data_quality",
    "backend/features",
    "backend/ml/calibration",
    "backend/ml/metrics",
    "backend/ml/models",
    "backend/ml/preprocessing",
    "backend/prediction_service",
    "backend/research",
    "backend/services/racing_api",
    "backend/strategy",
    "frontend/dashboard",
    "logs",
    "models",
    "migrations/versions",
    "scripts",
    "tests/fixtures",
]


@pytest.mark.parametrize("module_name", REQUIRED_PACKAGES)
def test_package_imports(module_name):
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_file_exists(relative_path):
    assert (PROJECT_ROOT / relative_path).is_file(), f"missing {relative_path}"


@pytest.mark.parametrize("relative_path", REQUIRED_DIRECTORIES)
def test_required_directory_exists(relative_path):
    assert (PROJECT_ROOT / relative_path).is_dir(), f"missing {relative_path}/"


def test_env_example_documents_every_credential():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in (
        "RACING_API_USERNAME",
        "RACING_API_PASSWORD",
        "RACING_API_KEY",
        "POSTGRES_PASSWORD",
        "SECRET_KEY",
        "MIN_EXPECTED_VALUE",
    ):
        assert key in text, f"{key} is not documented in .env.example"


def test_env_example_ships_no_real_secrets():
    """Every credential line in the template must be blank or a placeholder."""
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    placeholders = {"", "change_me_locally", "change-me-in-production"}
    for line in text.splitlines():
        if re.match(r"^(RACING_API_(USERNAME|PASSWORD|KEY)|POSTGRES_PASSWORD|SECRET_KEY)=", line):
            value = line.split("=", 1)[1].strip().strip('"')
            assert value in placeholders, f"real-looking secret in .env.example: {line}"


def test_dotenv_is_git_ignored():
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignore
    assert "!.env.example" in ignore


def _reads_environment(tree: ast.AST) -> bool:
    """True if the module actually *executes* an environment read.

    Parsed rather than grepped: docstrings legitimately mention ``os.environ``
    when explaining why the module does not use it, and a regex cannot tell
    documentation from code.
    """
    for node in ast.walk(tree):
        is_os_attribute = (
            isinstance(node, ast.Attribute)
            and node.attr in {"environ", "getenv"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        )
        is_os_import = (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and any(alias.name in {"environ", "getenv"} for alias in node.names)
        )
        if is_os_attribute or is_os_import:
            return True
    return False


def test_no_module_reads_os_environ_directly():
    """Configuration must flow through backend.utils.config, with one exception."""
    allowed = {PROJECT_ROOT / "backend" / "utils" / "config.py"}
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "backend").rglob("*.py")
        if path not in allowed and _reads_environment(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, f"these modules bypass Settings: {offenders}"


def test_the_environment_guardrail_actually_detects_violations():
    """A guardrail that cannot fail is not a guardrail."""
    assert _reads_environment(ast.parse("import os\nx = os.environ['SECRET']"))
    assert _reads_environment(ast.parse("import os\nx = os.getenv('SECRET')"))
    assert _reads_environment(ast.parse("from os import environ\nx = environ['SECRET']"))
    assert not _reads_environment(ast.parse('"""Never read os.environ here."""\nx = 1'))


def test_no_hardcoded_credentials_in_backend():
    """Catch an obvious `password = "literal"` slipping into the codebase."""
    pattern = re.compile(
        r"""(password|passwd|api_key|secret|token)\s*=\s*["'](?!\s*$)(?!change-me)[^"'\n]{8,}["']""",
        re.IGNORECASE,
    )
    offenders = []
    for path in (PROJECT_ROOT / "backend").rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "SecretStr(" in line or "get_secret_value" in line:
                continue
            if pattern.search(line):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")
    assert not offenders, f"possible hardcoded credentials: {offenders}"


def test_cli_tracebacks_never_dump_local_variables():
    """Typer prints frame locals on an unhandled exception by default.

    That bypasses every credential protection in the project: `SecretStr` masks
    `repr(settings)`, but the database password reaches SQLAlchemy as a plain
    string and sits in a frame local, so a dropped connection prints it to the
    console in full. Found exactly that way — a failed `phase6_cli audit` dumped
    the DSN kwargs, password included.
    """
    offenders = []
    for path in sorted((PROJECT_ROOT / "scripts").glob("*_cli.py")):
        source = path.read_text(encoding="utf-8")
        if "typer.Typer(" not in source:
            continue
        if "pretty_exceptions_show_locals=False" not in source:
            offenders.append(path.name)
    assert not offenders, (
        f"these CLIs would print credentials in a traceback: {offenders}. "
        "Pass pretty_exceptions_show_locals=False to typer.Typer()."
    )


def test_every_cli_is_covered_by_that_guardrail():
    """The guardrail is worthless if it silently matches nothing."""
    clis = list((PROJECT_ROOT / "scripts").glob("*_cli.py"))
    assert len(clis) >= 5, f"expected the CLI suite, found {[p.name for p in clis]}"
