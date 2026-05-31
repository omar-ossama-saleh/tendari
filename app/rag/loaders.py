"""Source loaders: extract plain text from PDF bytes, URLs, and HTML.

URL fetching is operator-driven (the documents endpoint), but we still guard
against SSRF: only http/https, and the host must resolve to a public address —
no loopback/private/link-local/metadata endpoints.
"""

from __future__ import annotations

import http.client
import io
import ipaddress
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

_FETCH_TIMEOUT_S = 20
_MAX_FETCH_BYTES = 10 * 1024 * 1024  # 10 MB cap
_MAX_REDIRECTS = 5
_USER_AGENT = "tendari-ingest/0.1"
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class IngestError(Exception):
    """Raised when a source cannot be loaded/extracted."""


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pypdf raises a variety of errors on bad input
        raise IngestError(f"Could not parse PDF: {exc}") from exc
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise IngestError("PDF contained no extractable text (is it a scan?).")
    return text


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return "\n".join(parser.parts)


# --------------------------------------------------------------------------- #
# URL — SSRF-hardened fetch
#
# Two classic SSRF bypasses are explicitly defended:
#   * redirect bypass — every hop's URL is re-validated, not just the first;
#   * DNS rebinding / TOCTOU — we resolve the host, validate the IP, then
#     connect the socket to THAT validated IP (sending the original Host header
#     / SNI), so the address we checked is the address we talk to.
# --------------------------------------------------------------------------- #
def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    # Unwrap IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) before classifying.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_safe_ip(host: str) -> str:
    """Resolve ``host`` and require EVERY returned address to be public.

    Returns one validated IP to connect to. Rejecting on any non-public address
    blocks split-horizon / multi-A-record tricks.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise IngestError(f"Could not resolve host: {host}") from exc
    ips = [info[4][0] for info in infos]
    if not ips:
        raise IngestError(f"Could not resolve host: {host}")
    for ip in ips:
        if not _is_public_ip(ip):
            raise IngestError("URL resolves to a non-public address.")
    return ips[0]


def validate_public_url(url: str) -> None:
    """Reject non-http(s) schemes and hosts that resolve to non-public IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IngestError("Only http/https URLs are allowed.")
    if not parsed.hostname:
        raise IngestError("URL has no host.")
    _resolve_safe_ip(parsed.hostname)


def _http_get_pinned(scheme: str, host: str, ip: str, port: int, path: str):
    """GET ``path`` from ``host`` but with the socket pinned to validated ``ip``."""
    if scheme == "https":
        context = ssl.create_default_context()
        conn = http.client.HTTPSConnection(host, port, timeout=_FETCH_TIMEOUT_S, context=context)
        raw = socket.create_connection((ip, port), timeout=_FETCH_TIMEOUT_S)
        # SNI + cert verification against the real hostname; bytes go to `ip`.
        conn.sock = context.wrap_socket(raw, server_hostname=host)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=_FETCH_TIMEOUT_S)
        conn.sock = socket.create_connection((ip, port), timeout=_FETCH_TIMEOUT_S)
    try:
        conn.request("GET", path, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
        resp = conn.getresponse()
        status = resp.status
        headers = resp.msg
        body = resp.read(_MAX_FETCH_BYTES + 1)
    finally:
        conn.close()
    return status, headers, body


def _safe_fetch(url: str) -> tuple[bytes, str]:
    """Fetch ``url`` with per-hop SSRF re-validation and IP pinning."""
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            raise IngestError("Only http/https URLs are allowed.")
        host = parsed.hostname
        if not host:
            raise IngestError("URL has no host.")
        ip = _resolve_safe_ip(host)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        try:
            status, headers, body = _http_get_pinned(parsed.scheme, host, ip, port, path)
        except IngestError:
            raise
        except Exception as exc:
            raise IngestError(f"Failed to fetch URL: {exc}") from exc

        if status in _REDIRECT_STATUSES:
            location = headers.get("Location")
            if not location:
                raise IngestError("Redirect response without a Location header.")
            current = urljoin(current, location)  # re-validated on next loop
            continue
        if status >= 400:
            raise IngestError(f"Fetch failed with HTTP status {status}.")
        if len(body) > _MAX_FETCH_BYTES:
            raise IngestError("Remote document exceeds the 10 MB fetch limit.")
        return body, headers.get_content_type()

    raise IngestError("Too many redirects.")


def fetch_url(url: str) -> str:
    """Fetch a URL (SSRF-guarded, redirect-safe) and return extracted plain text."""
    body, content_type = _safe_fetch(url)
    if content_type == "application/pdf" or url.lower().endswith(".pdf"):
        return parse_pdf(body)
    text = html_to_text(body.decode("utf-8", errors="replace"))
    if not text.strip():
        raise IngestError("Fetched URL contained no extractable text.")
    return text
