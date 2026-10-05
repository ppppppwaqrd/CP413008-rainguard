"""Download weatherAUS from the first mirror that responds."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rainguard import config


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_CSV.exists() and config.RAW_CSV.stat().st_size > 1_000_000:
        print(f"already present: {config.RAW_CSV}")
        return
    errors: list[str] = []
    for url in config.DOWNLOAD_URLS:
        try:
            print(f"trying {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "rainguard-course-project"})
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if len(payload) < 1_000_000 or b"RainTomorrow" not in payload[:5000]:
                errors.append(f"{url} returned {len(payload)} bytes without the expected header")
                continue
            config.RAW_CSV.write_bytes(payload)
            print(f"wrote {config.RAW_CSV} ({len(payload)} bytes)")
            return
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
    raise SystemExit("download failed\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
