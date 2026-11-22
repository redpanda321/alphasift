"""Daily read-only market research pipeline; never places trades."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from alphasift.lifecycle_contract import strategy_contract


def validate_scan(payload):
    rows = payload["rows"]
    if not rows or len(rows) != payload["attempted"]:
        raise ValueError("Empty or inconsistent scan")
    failures = sum(r["status"] == "FAILED" for r in rows)
    if failures / len(rows) > 0.20:
        raise ValueError(f"Provider failure rate {failures}/{len(rows)} exceeds 20%")
    if not any(r["status"] == "AGREEMENT" for r in rows):
        raise ValueError("No agreement candidates; preserve previous successful report")
    return {"attempted": len(rows), "failed": failures, "as_of": payload["as_of"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration without network or scanning",
    )
    parser.add_argument(
        "--data-root", type=Path, help="Persistent root for immutable daily runs"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    source = args.data_root.resolve() if args.data_root else root / "data"
    source.mkdir(parents=True, exist_ok=True)
    from alphasift.freshness import latest_completed_session

    expected = {m: latest_completed_session(m) for m in ("cn", "us")}
    if args.check:
        print(
            json.dumps(
                {
                    "status": "configuration_ok",
                    "sessions": expected,
                    "python": sys.executable,
                    "strategy": strategy_contract(),
                }
            )
        )
        return
    runs = source / "daily-runs"
    runs.mkdir(exist_ok=True)
    # OS-released lock survives crashes without leaving an orphan lock file.
    with (runs / "pipeline.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another daily pipeline is already running") from None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work = runs / stamp
        data = work / "data"
        data.mkdir(parents=True)
        manifest = {
            "started_at": stamp,
            "status": "running",
            "sessions": expected,
            "strategy": strategy_contract(),
            "metadata": "saved sector classifications; may be stale",
        }
        manifest_path = work / "manifest.json"

        def save():
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        save()
        env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(root))

        def run(module, *arguments):
            with (work / "run.log").open("a", encoding="utf-8") as log:
                log.write(
                    f"\n{datetime.now(timezone.utc).isoformat()} {module} {arguments}\n"
                )
                log.flush()
                subprocess.run(
                    [sys.executable, "-u", "-m", module, *arguments],
                    cwd=work,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=21600,
                )

        try:
            for market in ("cn", "us"):
                name = f"lifecycle-crosscheck-{market}.json"
                run(
                    "alphasift.lifecycle_crosscheck",
                    "--market",
                    market,
                    "--output",
                    str(data / name),
                )
                payload = json.loads((data / name).read_text(encoding="utf-8"))
                if payload.get("strategy") != strategy_contract():
                    raise ValueError(
                        "Scan strategy differs from the scheduled strategy contract"
                    )
                manifest[market] = validate_scan(payload)
                save()
            run("alphasift.lifecycle_daily_report", "--data-dir", str(data))
            result = json.loads(
                (data / "lifecycle-industry-top5.json").read_text(encoding="utf-8")
            )
            if not result["selected"]:
                raise ValueError("No financial selections; previous report retained")
            # Publish only the pointer to a fully completed, immutable run.
            manifest.update(
                status="success",
                completed_at=datetime.now(timezone.utc).isoformat(),
                candidates=result["scope"],
                eligible=result["eligible"],
                selected=result["selected"],
                stage_counts=result["stage_counts"],
                exclusions=result["exclusions"],
            )
            save()
            pointer = runs / "latest-success.tmp"
            pointer.write_text(
                json.dumps(
                    {
                        "run": str(work),
                        "report": str(data / "lifecycle-industry-top5.html"),
                    }
                ),
                encoding="utf-8",
            )
            pointer.replace(runs / "latest-success.json")
            import html

            link = (data / "lifecycle-industry-top5.html").as_uri()
            landing = source / "lifecycle-daily-latest.tmp"
            landing.write_text(
                '<!doctype html><meta charset="utf-8"><title>AlphaSift每日结果</title>'
                "<h1>AlphaSift最近成功报告</h1><p>"
                + html.escape(str(manifest))
                + '</p><a href="'
                + html.escape(link)
                + '">打开分类结果</a>',
                encoding="utf-8",
            )
            landing.replace(source / "lifecycle-daily-latest.html")
        except Exception as exc:
            manifest.update(
                status="failed",
                error=str(exc),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            save()
            raise


if __name__ == "__main__":
    main()
