"""Call every API endpoint once against a running server and show the ``api_call_logs`` rows the
calls produced.

    python run.py                          # in another terminal, with IRENT_DB_PATH set
    python scripts/smoke_test_api.py [--image car.jpg] [--plate RAC-4582]

Exits non-zero if any call returns an unexpected status code.
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from PIL import Image

CLIENT_ID = "smoke-test"


@dataclass(frozen=True)
class Case:
    name: str
    method: str
    path: str
    expected: int
    upload: tuple[str, bytes, str] | None = None
    form: dict[str, str] | None = None


def _sample_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(200, 120, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def build_cases(image: tuple[str, bytes, str], plate: str | None) -> list[Case]:
    plate_form = {"plate_number": plate} if plate else None
    return [
        Case("health", "GET", "/api/v1/health", 200),
        Case("damage evaluate", "POST", "/api/v1/damage/evaluate", 200, image, plate_form),
        Case("plate recognize", "POST", "/api/v1/plate/recognize", 200, image),
        Case("tire detect", "POST", "/api/v1/tire/detect", 200, image, plate_form),
        Case(
            "damage evaluate (GIF -> 415)",
            "POST",
            "/api/v1/damage/evaluate",
            415,
            ("bad.gif", b"GIF89a", "image/gif"),
        ),
    ]


def run_case(client: httpx.Client, case: Case) -> bool:
    files = {"file": case.upload} if case.upload else None
    try:
        resp = client.request(case.method, case.path, files=files, data=case.form)
    except httpx.HTTPError as exc:
        print(f"FAIL  {case.name:<30} {case.method} {case.path} -> {exc}")
        return False
    ok = resp.status_code == case.expected
    detail = "" if ok else f" (expected {case.expected}) {resp.text[:200]}"
    print(f"{'PASS' if ok else 'FAIL'}  {case.name:<30} {case.method} {case.path} -> "
          f"{resp.status_code}{detail}")
    return ok


def print_log_rows(db: Path, since: str) -> None:
    if not db.exists():
        print(f"\n(no database at {db} - skipping api_call_logs check)")
        return
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT id, api_name, http_method, return_code, return_msg, client, record_time "
            "FROM api_call_logs WHERE client = ? AND record_time >= ? ORDER BY id",
            (CLIENT_ID, since),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"\n(could not read api_call_logs: {exc})")
        return
    finally:
        conn.close()
    print(f"\napi_call_logs rows from this run ({len(rows)}):")
    for row in rows:
        print("  " + " | ".join(str(v) for v in row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, help="Vehicle photo to upload (default: synthetic)")
    parser.add_argument("--plate", help="plate_number form field for damage / tire calls")
    parser.add_argument("--db", type=Path, default=Path("data/irent_op_backend.sqlite"))
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request seconds")
    args = parser.parse_args()

    if args.image:
        suffix = args.image.suffix.lower().lstrip(".")
        content_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, f"image/{suffix}")
        image = (args.image.name, args.image.read_bytes(), content_type)
    else:
        image = ("sample.jpg", _sample_jpeg(), "image/jpeg")

    # record_time is SQLite CURRENT_TIMESTAMP: UTC, 'YYYY-MM-DD HH:MM:SS'.
    since = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    with httpx.Client(
        base_url=args.base_url, headers={"X-Client-Id": CLIENT_ID}, timeout=args.timeout
    ) as client:
        results = [run_case(client, case) for case in build_cases(image, args.plate)]

    print_log_rows(args.db, since)
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
