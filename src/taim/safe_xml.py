"""Parse XML from frozen benchmark inputs without accepting a DTD.

CPython's :mod:`xml.etree.ElementTree` never resolves external entities and
never retrieves an external DTD, so classic XXE does not apply here.  It does
expand *internally declared* entities, which leaves the "billion laughs"
entity-expansion denial of service open::

    <!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>
    <lolz>&lol2;</lolz>

Entities can only be declared inside a DTD, so refusing a DTD outright closes
that gap without adding a runtime dependency.  The C accelerator exposes no
expat handler to hook, so the declaration is detected by walking the prolog,
which per the XML grammar may contain only whitespace, processing
instructions, and comments before an optional ``<!DOCTYPE`` and the root
element.  Walking it explicitly, rather than searching the whole document,
keeps a literal ``<!DOCTYPE`` inside a comment or CDATA section from being
mistaken for a real declaration.

None of TAIM's frozen inputs (TREC topics, ClinicalTrials.gov trial records,
OOXML workbook members) legitimately carries a DTD, so canonical outputs are
unaffected.
"""

from __future__ import annotations

from xml.etree import ElementTree

__all__ = ["UnsafeXmlError", "fromstring"]

_BOM = b"\xef\xbb\xbf"
_DOCTYPE = b"<!DOCTYPE"
_WHITESPACE = b" \t\r\n"


class UnsafeXmlError(ValueError):
    """Raised when XML input declares a DTD."""


def _reject_doctype(serialized: bytes) -> None:
    """Raise if a ``<!DOCTYPE`` declaration appears in the document prolog."""

    index = len(_BOM) if serialized.startswith(_BOM) else 0
    total = len(serialized)
    while index < total:
        character = serialized[index : index + 1]
        if character in _WHITESPACE:
            index += 1
        elif serialized.startswith(b"<?", index):
            end = serialized.find(b"?>", index + 2)
            if end == -1:
                return  # Malformed; let ElementTree report it.
            index = end + 2
        elif serialized.startswith(b"<!--", index):
            end = serialized.find(b"-->", index + 4)
            if end == -1:
                return  # Malformed; let ElementTree report it.
            index = end + 3
        elif serialized.startswith(_DOCTYPE, index):
            raise UnsafeXmlError(
                "XML document type declarations are not accepted; "
                "entity expansion is disabled for frozen benchmark inputs"
            )
        else:
            return  # Root element (or malformed input): the prolog is over.


def fromstring(serialized: bytes) -> ElementTree.Element:
    """Parse ``serialized`` like ``ElementTree.fromstring``, but reject DTDs."""

    _reject_doctype(serialized)
    return ElementTree.fromstring(serialized)  # noqa: S314 - DTD rejected above
