"""Shared grounding rules for pair-local clinical evidence references."""

from __future__ import annotations

from collections.abc import Mapping

PAIR_EVIDENCE_ROOTS = (
    "/patient_version/topic",
    "/patient_version/derived_evidence",
    "/trial_version/trial",
    "/trial_version/derived_evidence",
    "/visible_pair_evidence",
)


def pair_evidence_reference_paths(payload: Mapping[str, object]) -> frozenset[str]:
    """Return JSON Pointer nodes that resolve inside supplied pair evidence."""

    allowed: set[str] = set()

    def add_paths(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                nested_path = f"{path}/{escaped}"
                allowed.add(nested_path)
                add_paths(nested, nested_path)
        elif isinstance(value, list | tuple):
            for index, nested in enumerate(value):
                nested_path = f"{path}/{index}"
                allowed.add(nested_path)
                add_paths(nested, nested_path)

    for root in PAIR_EVIDENCE_ROOTS:
        value: object = payload
        for part in root.removeprefix("/").split("/"):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            add_paths(value, root)
    return frozenset(allowed)


__all__ = ["PAIR_EVIDENCE_ROOTS", "pair_evidence_reference_paths"]
