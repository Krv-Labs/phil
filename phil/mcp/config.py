"""
Canonical YAML schema, validation, and override application for Phil sweeps.

The MCP layer accepts a small, declarative YAML config and materializes it
into the keyword arguments that :class:`phil.Phil` already understands.
Only the surface that the agent needs to control is exposed; advanced
sklearn objects (estimators, ParameterGrids) are constructed for the agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import yaml
from sklearn.model_selection import ParameterGrid

from phil.gallery import GridGallery
from phil.imputation import ImputationConfig
from phil.magic import ECTConfig


GRID_INTENTS: dict[str, str] = {
    "default": "Mixed regression imputers across BayesianRidge, Decision Tree, Random Forest, and Gradient Boosting.",
    "sampling": "Distribution sampling imputation across 100 seeds (good for preserving marginals).",
    "finance": "Iterative + KNN + Simple imputers tuned for tabular financial data.",
    "healthcare": "KNN, Simple, and Iterative imputers tuned for clinical tables.",
    "marketing": "Categorical-friendly Simple, KNN, and Iterative imputers.",
    "engineering": "Robust mean/median + KNN + Decision-Tree iterative imputation for sensor data.",
}

_ALLOWED_TOP_LEVEL = {"run", "imputation", "magic", "output"}
_ALLOWED_RUN = {"name", "data"}
_ALLOWED_IMPUTATION = {
    "grid",
    "custom",
    "samples",
    "random_state",
    "max_iter",
    "drop_cols",
    "missingness_thresh",
    "encode_categoricals",
}
_ALLOWED_CUSTOM = {"methods", "modules", "grids"}
_ALLOWED_MAGIC = {
    "method",
    "num_thetas",
    "radius",
    "resolution",
    "scale",
    "normalize",
    "seed",
}
_ALLOWED_OUTPUT = {"write_csv", "write_parquet"}

_SUPPORTED_MAGIC_METHODS = {"ECT"}
_BUILTIN_GRIDS = set(GRID_INTENTS.keys())


@dataclass
class ValidationIssue:
    path: str
    message: str
    expected: str | None = None
    received: Any = None
    suggestion: str | None = None


@dataclass
class ValidationReport:
    ok: bool
    normalized_yaml: str | None
    resolved_dataset_path: str | None
    issues: list[ValidationIssue] = field(default_factory=list)
    error_code: str | None = None
    agent_action: str | None = None


@dataclass
class ConfigOverrideResult:
    config_yaml: str
    applied_overrides: list[str]
    diff: list[dict[str, Any]]


def parse_yaml_mapping(config_yaml: str) -> dict[str, Any]:
    """Parse YAML, rejecting fenced Markdown code blocks."""
    stripped = config_yaml.strip()
    if stripped.startswith("```"):
        raise ValueError(
            "config_yaml must be raw YAML, not a fenced Markdown code block"
        )
    parsed = yaml.safe_load(config_yaml)
    if not isinstance(parsed, dict):
        raise ValueError("config_yaml must be a valid YAML mapping")
    return parsed


def default_config_dict(
    *, run_name: str = "phil_sweep", data: str = ""
) -> dict[str, Any]:
    """Return the canonical default config dict for Phil sweeps."""
    return {
        "run": {"name": run_name, "data": data},
        "imputation": {
            "grid": "default",
            "samples": 30,
            "random_state": 42,
            "max_iter": 5,
            "drop_cols": [],
            "missingness_thresh": None,
            "encode_categoricals": True,
        },
        "magic": {
            "method": "ECT",
            "num_thetas": 64,
            "radius": 1.0,
            "resolution": 100,
            "scale": 500,
            "normalize": True,
            "seed": 42,
        },
        "output": {"write_csv": ""},
    }


def render_default_config_yaml(*, run_name: str = "phil_sweep", data: str = "") -> str:
    return yaml.safe_dump(
        default_config_dict(run_name=run_name, data=data),
        sort_keys=False,
    )


def validate_config_yaml(
    config_yaml: str,
    *,
    dataset_path: str | None = None,
) -> ValidationReport:
    """Validate a Phil MCP config YAML string."""
    issues: list[ValidationIssue] = []

    try:
        raw = parse_yaml_mapping(config_yaml)
    except ValueError as exc:
        error_code = "YAML_NOT_RAW" if "raw YAML" in str(exc) else "CONFIG_YAML_INVALID"
        return ValidationReport(
            ok=False,
            normalized_yaml=None,
            resolved_dataset_path=dataset_path,
            issues=[ValidationIssue(path="$", message=str(exc))],
            error_code=error_code,
            agent_action=(
                "Pass raw YAML only. Do not wrap config_yaml in Markdown fences."
            ),
        )

    _validate_known_sections(raw, issues)
    _validate_run(raw.get("run"), issues)
    _validate_imputation(raw.get("imputation"), issues)
    _validate_magic(raw.get("magic"), issues)
    _validate_output(raw.get("output"), issues)

    if issues:
        return ValidationReport(
            ok=False,
            normalized_yaml=None,
            resolved_dataset_path=dataset_path,
            issues=issues,
            error_code="CONFIG_VALIDATION_FAILED",
            agent_action="Resolve the issues listed above, then re-validate.",
        )

    normalized = _normalize_config(raw)
    normalized_yaml = yaml.safe_dump(normalized, sort_keys=False)
    resolved_path = dataset_path or normalized.get("run", {}).get("data") or None
    return ValidationReport(
        ok=True,
        normalized_yaml=normalized_yaml,
        resolved_dataset_path=resolved_path,
        issues=[],
    )


def _validate_known_sections(
    raw: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    unknown = sorted(set(raw.keys()) - _ALLOWED_TOP_LEVEL)
    for key in unknown:
        issues.append(
            ValidationIssue(
                path=f"$.{key}",
                message=f"Unknown top-level section '{key}'.",
                expected=f"One of {sorted(_ALLOWED_TOP_LEVEL)}",
                received=key,
                suggestion="Remove the section or move its keys under a supported one.",
            )
        )


def _validate_run(section: Any, issues: list[ValidationIssue]) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        issues.append(ValidationIssue(path="$.run", message="'run' must be a mapping."))
        return
    unknown = sorted(set(section.keys()) - _ALLOWED_RUN)
    for key in unknown:
        issues.append(
            ValidationIssue(
                path=f"$.run.{key}",
                message=f"Unknown key '{key}' under 'run'.",
                expected=f"One of {sorted(_ALLOWED_RUN)}",
            )
        )


def _validate_imputation(section: Any, issues: list[ValidationIssue]) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        issues.append(
            ValidationIssue(
                path="$.imputation", message="'imputation' must be a mapping."
            )
        )
        return
    unknown = sorted(set(section.keys()) - _ALLOWED_IMPUTATION)
    for key in unknown:
        issues.append(
            ValidationIssue(
                path=f"$.imputation.{key}",
                message=f"Unknown key '{key}' under 'imputation'.",
                expected=f"One of {sorted(_ALLOWED_IMPUTATION)}",
            )
        )

    grid = section.get("grid", "default")
    if grid not in _BUILTIN_GRIDS and grid != "custom":
        issues.append(
            ValidationIssue(
                path="$.imputation.grid",
                message=f"Unknown imputation grid '{grid}'.",
                expected=f"One of {sorted(_BUILTIN_GRIDS)} or 'custom'.",
                received=grid,
                suggestion="Call list_grids to see all built-in grids.",
            )
        )

    if grid == "custom":
        custom = section.get("custom")
        if not isinstance(custom, dict):
            issues.append(
                ValidationIssue(
                    path="$.imputation.custom",
                    message="grid='custom' requires a 'custom' mapping with methods/modules/grids.",
                    expected="{methods: [...], modules: [...], grids: [...]}",
                )
            )
            return
        unknown_custom = sorted(set(custom.keys()) - _ALLOWED_CUSTOM)
        for key in unknown_custom:
            issues.append(
                ValidationIssue(
                    path=f"$.imputation.custom.{key}",
                    message=f"Unknown key '{key}' under 'imputation.custom'.",
                    expected=f"One of {sorted(_ALLOWED_CUSTOM)}",
                )
            )
        for required in ("methods", "modules", "grids"):
            if required not in custom:
                issues.append(
                    ValidationIssue(
                        path=f"$.imputation.custom.{required}",
                        message=f"Missing required key '{required}' for custom grid.",
                    )
                )
                continue
            if not isinstance(custom[required], list):
                issues.append(
                    ValidationIssue(
                        path=f"$.imputation.custom.{required}",
                        message=f"'{required}' must be a list.",
                    )
                )
        if (
            isinstance(custom.get("methods"), list)
            and isinstance(custom.get("modules"), list)
            and isinstance(custom.get("grids"), list)
            and not (
                len(custom["methods"]) == len(custom["modules"]) == len(custom["grids"])
            )
        ):
            issues.append(
                ValidationIssue(
                    path="$.imputation.custom",
                    message="methods, modules, and grids must have the same length.",
                )
            )

    for int_key in ("samples", "random_state", "max_iter"):
        if int_key in section and section[int_key] is not None:
            if not isinstance(section[int_key], int) or isinstance(
                section[int_key], bool
            ):
                issues.append(
                    ValidationIssue(
                        path=f"$.imputation.{int_key}",
                        message=f"'{int_key}' must be an integer.",
                        received=section[int_key],
                    )
                )

    if "samples" in section and isinstance(section["samples"], int):
        if section["samples"] <= 0:
            issues.append(
                ValidationIssue(
                    path="$.imputation.samples",
                    message="'samples' must be > 0.",
                    received=section["samples"],
                )
            )

    if "drop_cols" in section and section["drop_cols"] is not None:
        drop_cols = section["drop_cols"]
        if not isinstance(drop_cols, list):
            issues.append(
                ValidationIssue(
                    path="$.imputation.drop_cols",
                    message="'drop_cols' must be a list of column names.",
                    received=drop_cols,
                )
            )
        elif any(not isinstance(col, str) for col in drop_cols):
            issues.append(
                ValidationIssue(
                    path="$.imputation.drop_cols",
                    message="'drop_cols' entries must be strings.",
                    received=drop_cols,
                )
            )

    if (
        "missingness_thresh" in section
        and section["missingness_thresh"] is not None
        and (
            not isinstance(section["missingness_thresh"], (int, float))
            or isinstance(section["missingness_thresh"], bool)
        )
    ):
        issues.append(
            ValidationIssue(
                path="$.imputation.missingness_thresh",
                message="'missingness_thresh' must be a number between 0 and 1.",
                received=section["missingness_thresh"],
            )
        )
    elif (
        "missingness_thresh" in section
        and section["missingness_thresh"] is not None
        and not 0 <= float(section["missingness_thresh"]) <= 1
    ):
        issues.append(
            ValidationIssue(
                path="$.imputation.missingness_thresh",
                message="'missingness_thresh' must be between 0 and 1.",
                received=section["missingness_thresh"],
            )
        )

    if (
        "encode_categoricals" in section
        and section["encode_categoricals"] is not None
        and not isinstance(section["encode_categoricals"], bool)
    ):
        issues.append(
            ValidationIssue(
                path="$.imputation.encode_categoricals",
                message="'encode_categoricals' must be a boolean.",
                received=section["encode_categoricals"],
            )
        )


def _validate_magic(section: Any, issues: list[ValidationIssue]) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        issues.append(
            ValidationIssue(path="$.magic", message="'magic' must be a mapping.")
        )
        return
    unknown = sorted(set(section.keys()) - _ALLOWED_MAGIC)
    for key in unknown:
        issues.append(
            ValidationIssue(
                path=f"$.magic.{key}",
                message=f"Unknown key '{key}' under 'magic'.",
                expected=f"One of {sorted(_ALLOWED_MAGIC)}",
            )
        )
    method = section.get("method", "ECT")
    if method not in _SUPPORTED_MAGIC_METHODS:
        issues.append(
            ValidationIssue(
                path="$.magic.method",
                message=f"Unsupported magic method '{method}'.",
                expected=f"One of {sorted(_SUPPORTED_MAGIC_METHODS)}",
            )
        )


def _validate_output(section: Any, issues: list[ValidationIssue]) -> None:
    if section is None:
        return
    if not isinstance(section, dict):
        issues.append(
            ValidationIssue(path="$.output", message="'output' must be a mapping.")
        )
        return
    unknown = sorted(set(section.keys()) - _ALLOWED_OUTPUT)
    for key in unknown:
        issues.append(
            ValidationIssue(
                path=f"$.output.{key}",
                message=f"Unknown key '{key}' under 'output'.",
                expected=f"One of {sorted(_ALLOWED_OUTPUT)}",
            )
        )


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill in defaults so downstream callers can assume a complete shape."""
    base = default_config_dict()
    for section_key in _ALLOWED_TOP_LEVEL:
        if section_key in raw and isinstance(raw[section_key], dict):
            base[section_key].update(
                {k: v for k, v in raw[section_key].items() if v is not None}
            )
        elif section_key in raw:
            base[section_key] = raw[section_key]
    if base["imputation"].get("grid") != "custom":
        base["imputation"].pop("custom", None)
    return base


def render_validation_report(report: ValidationReport) -> str:
    """Render a validation report as a JSON-compatible dict-string."""
    import json

    payload: dict[str, Any] = {
        "status": "ok" if report.ok else "error",
        "resolved_dataset_path": report.resolved_dataset_path,
    }
    if report.ok:
        payload["config_yaml"] = report.normalized_yaml
    else:
        payload["error_code"] = report.error_code
        payload["agent_action"] = report.agent_action
        payload["issues"] = [asdict(i) for i in report.issues]
    return json.dumps(payload, indent=2)


def apply_overrides(
    config_yaml: str, overrides: dict[str, Any]
) -> ConfigOverrideResult:
    """Apply dotted-path overrides to a config YAML.

    Override keys use ``a.b.c`` form, e.g. ``imputation.samples`` or
    ``magic.num_thetas``. Unknown keys raise ``ValueError`` listing the
    valid paths.
    """
    raw = parse_yaml_mapping(config_yaml)
    valid_paths = _allowed_override_paths()
    applied: list[str] = []
    diff: list[dict[str, Any]] = []
    for dotted, new_value in overrides.items():
        if dotted not in valid_paths:
            raise ValueError(
                f"Unknown override key '{dotted}'. Valid keys: {sorted(valid_paths)}"
            )
        old_value = _get_dotted(raw, dotted)
        _set_dotted(raw, dotted, new_value)
        applied.append(dotted)
        diff.append({"path": dotted, "old": old_value, "new": new_value})
    normalized = _normalize_config(raw)
    return ConfigOverrideResult(
        config_yaml=yaml.safe_dump(normalized, sort_keys=False),
        applied_overrides=applied,
        diff=diff,
    )


def _allowed_override_paths() -> set[str]:
    paths: set[str] = set()
    for key in _ALLOWED_RUN:
        paths.add(f"run.{key}")
    for key in _ALLOWED_IMPUTATION:
        if key == "custom":
            continue
        paths.add(f"imputation.{key}")
    for key in _ALLOWED_CUSTOM:
        paths.add(f"imputation.custom.{key}")
    for key in _ALLOWED_MAGIC:
        paths.add(f"magic.{key}")
    for key in _ALLOWED_OUTPUT:
        paths.add(f"output.{key}")
    return paths


def _get_dotted(obj: dict[str, Any], dotted: str) -> Any:
    cursor: Any = obj
    for part in dotted.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
        if cursor is None:
            return None
    return cursor


def _set_dotted(obj: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = obj
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def config_to_phil_kwargs(config_yaml: str) -> dict[str, Any]:
    """Materialize a validated config YAML into ``Phil`` constructor kwargs.

    Returns a dict with keys: ``samples``, ``param_grid`` (str name OR
    ``ImputationConfig``), ``magic`` (str name), ``config`` (``ECTConfig``),
    ``random_state``, ``max_iter``, plus convenience entries ``run_name``
    and ``data_path`` for orchestration.
    """
    raw = parse_yaml_mapping(config_yaml)
    normalized = _normalize_config(raw)

    imputation = normalized["imputation"]
    grid_name = imputation.get("grid", "default")
    if grid_name == "custom":
        custom = imputation.get("custom") or {}
        param_grid: Any = ImputationConfig(
            methods=list(custom["methods"]),
            modules=list(custom["modules"]),
            grids=[
                ParameterGrid(g) if isinstance(g, dict) else g for g in custom["grids"]
            ],
        )
    else:
        param_grid = grid_name

    magic_section = normalized.get("magic", {})
    ect_config = ECTConfig(
        num_thetas=int(magic_section.get("num_thetas", 64)),
        radius=float(magic_section.get("radius", 1.0)),
        resolution=int(magic_section.get("resolution", 100)),
        scale=int(magic_section.get("scale", 500)),
        normalize=bool(magic_section.get("normalize", True)),
        seed=int(magic_section.get("seed", 42)),
    )

    return {
        "samples": int(imputation.get("samples", 30)),
        "param_grid": param_grid,
        "magic": str(magic_section.get("method", "ECT")),
        "config": ect_config,
        "random_state": imputation.get("random_state", 42),
        "max_iter": int(imputation.get("max_iter", 5)),
        "drop_cols": list(imputation.get("drop_cols", [])),
        "missingness_thresh": imputation.get("missingness_thresh"),
        "encode_categoricals": bool(imputation.get("encode_categoricals", True)),
        "run_name": normalized.get("run", {}).get("name") or "phil_sweep",
        "data_path": normalized.get("run", {}).get("data") or "",
    }


def list_builtin_grids() -> list[dict[str, Any]]:
    """Return a list of built-in grid descriptions."""
    payload: list[dict[str, Any]] = []
    for name in sorted(_BUILTIN_GRIDS):
        grid: ImputationConfig = GridGallery.get(name)
        payload.append(
            {
                "name": name,
                "intent": GRID_INTENTS.get(name, ""),
                "methods": list(grid.methods),
                "modules": list(grid.modules),
            }
        )
    return payload
