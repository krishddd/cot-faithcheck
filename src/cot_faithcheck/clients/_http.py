"""A tiny JSON-over-HTTP helper built on the standard library.

Keeping this dependency-free means ``pip install cot-faithcheck`` pulls in nothing
but Python itself; the provider SDKs are optional. The helper does POST-JSON with
a couple of retries on transient failures.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ..errors import ClientError


def post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
    retries: int = 2,
    backoff: float = 1.5,
) -> Dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response.

    Retries on network errors and 5xx / 429 responses with exponential backoff.
    Raises :class:`ClientError` once retries are exhausted.
    """
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            status = exc.code
            detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
            last_err = ClientError(f"HTTP {status} from {url}: {detail[:500]}")
            # 4xx other than 429 are not worth retrying.
            if status not in (429,) and status < 500:
                raise last_err from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = ClientError(f"request to {url} failed: {exc}")
        except json.JSONDecodeError as exc:
            raise ClientError(f"invalid JSON from {url}: {exc}") from exc

        if attempt < retries:
            time.sleep(backoff**attempt)

    assert last_err is not None
    raise last_err
