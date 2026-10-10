# ruff: noqa: S101
from __future__ import annotations

from xml.etree import ElementTree

import pytest

from taim import safe_xml

BILLION_LAUGHS = (
    b'<?xml version="1.0"?>\n'
    b'<!DOCTYPE lolz [<!ENTITY lol "lol">'
    b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>\n'
    b"<lolz>&lol2;</lolz>"
)
EXTERNAL_DTD = b'<?xml version="1.0"?><!DOCTYPE r SYSTEM "http://example.invalid/x.dtd"><r>ok</r>'
XXE = b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><r>&x;</r>'


def test_billion_laughs_expands_under_plain_elementtree() -> None:
    """Pin the vulnerability this module exists to close.

    The unhardened call is the subject under test, so the S314 report it
    triggers is expected here and nowhere else.
    """

    assert ElementTree.fromstring(BILLION_LAUGHS).text == "lol" * 10  # noqa: S314


@pytest.mark.parametrize(
    "payload",
    [BILLION_LAUGHS, EXTERNAL_DTD, XXE],
    ids=["billion-laughs", "external-dtd", "xxe"],
)
def test_documents_declaring_a_dtd_are_rejected(payload: bytes) -> None:
    with pytest.raises(safe_xml.UnsafeXmlError, match="document type declarations are not"):
        safe_xml.fromstring(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"<r>ok</r>",
        b'<?xml version="1.0"?><r>ok</r>',
        b'<?xml version="1.0"?>\n<!-- a comment -->\n<r>ok</r>',
        b"\xef\xbb\xbf<r>ok</r>",
        b"  \n\t<r>ok</r>",
        b'<?xml version="1.0"?><?processing instruction?><r>ok</r>',
    ],
    ids=["bare", "declared", "commented", "bom", "leading-space", "processing-instruction"],
)
def test_ordinary_documents_still_parse(payload: bytes) -> None:
    assert safe_xml.fromstring(payload).text == "ok"


@pytest.mark.parametrize(
    "payload",
    [
        b"<r><!-- <!DOCTYPE fake [<!ENTITY x 'y'>]> --></r>",
        b"<r><![CDATA[<!DOCTYPE fake>]]></r>",
        b"<r>&lt;!DOCTYPE fake&gt;</r>",
    ],
    ids=["inside-comment", "inside-cdata", "escaped-text"],
)
def test_a_doctype_string_in_content_is_not_mistaken_for_a_declaration(payload: bytes) -> None:
    """Only a real prolog declaration counts, so content is never misread."""

    assert safe_xml.fromstring(payload).tag == "r"


def test_malformed_input_still_raises_the_parser_error() -> None:
    with pytest.raises(ElementTree.ParseError):
        safe_xml.fromstring(b"<r>unclosed")


def test_comment_before_a_doctype_does_not_hide_it() -> None:
    payload = b"<!-- leading --><!DOCTYPE r><r>ok</r>"

    with pytest.raises(safe_xml.UnsafeXmlError):
        safe_xml.fromstring(payload)
