"""Loaders: HTML→text, PDF error handling, and the SSRF URL guard."""

from __future__ import annotations

import pytest

from app.rag.loaders import IngestError, html_to_text, parse_pdf, validate_public_url


def test_html_to_text_strips_tags_and_scripts() -> None:
    html = "<html><head><style>.x{}</style></head><body><h1>Returns</h1>"\
           "<script>evil()</script><p>30 day window</p></body></html>"
    text = html_to_text(html)
    assert "Returns" in text
    assert "30 day window" in text
    assert "evil" not in text
    assert ".x{}" not in text


def test_parse_pdf_rejects_non_pdf_bytes() -> None:
    with pytest.raises(IngestError):
        parse_pdf(b"this is definitely not a pdf")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",          # bad scheme
        "file:///etc/passwd",           # bad scheme
        "http://127.0.0.1/admin",       # loopback
        "http://10.0.0.5/internal",     # private
        "http://192.168.1.1/",          # private
        "http://169.254.169.254/latest/meta-data",  # link-local (cloud metadata)
    ],
)
def test_validate_public_url_rejects_unsafe(url: str) -> None:
    with pytest.raises(IngestError):
        validate_public_url(url)


def test_validate_public_url_rejects_ipv4_mapped_ipv6() -> None:
    # ::ffff:127.0.0.1 must be unwrapped to loopback and rejected.
    with pytest.raises(IngestError):
        validate_public_url("http://[::ffff:127.0.0.1]/")


def test_validate_public_url_allows_public_ip_literal() -> None:
    # IP literal → no DNS needed; 93.184.216.34 (example.com) is public.
    validate_public_url("https://93.184.216.34/policy")
