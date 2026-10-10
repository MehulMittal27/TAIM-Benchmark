"""Exact Unicode source citations with content identity and span validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from taim.contracts import require_non_empty, require_sha256
from taim.schemas import SchemaValidationError


@dataclass(frozen=True, slots=True)
class SourceCitation:
    source_id: str
    source_sha256: str
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        require_non_empty(self.source_id, "source_id")
        require_sha256(self.source_sha256, "source_sha256")
        require_non_empty(self.quote, "quote")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 0 <= self.start < self.end
        ):
            raise SchemaValidationError("invalid citation offsets")

    @classmethod
    def from_source(
        cls, source_id: str, text: str, start: int, end: int, quote: str
    ) -> SourceCitation:
        citation = cls(
            source_id,
            "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            start,
            end,
            quote,
        )
        citation.validate_source(text)
        return citation

    def validate_source(self, text: str) -> None:
        if "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest() != self.source_sha256:
            raise SchemaValidationError("citation source hash does not match source")
        if self.end > len(text) or text[self.start : self.end] != self.quote:
            raise SchemaValidationError("citation quote or offsets do not match source")

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}
