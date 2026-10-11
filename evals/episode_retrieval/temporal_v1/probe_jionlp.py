"""Run frozen parser cases in a temporary environment; never edit expectations."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time


def effective_range(parsed: dict) -> list[str] | None:
    """Only finite, accurate calendar bounds may become retrieval filters."""
    bounds = parsed.get("time")
    if (
        parsed.get("definition") != "accurate"
        or parsed.get("type") not in {"time_point", "time_span"}
        or not isinstance(bounds, list)
        or len(bounds) != 2
    ):
        return None
    try:
        start, end = (datetime.strptime(value, "%Y-%m-%d %H:%M:%S") for value in bounds)
    except (TypeError, ValueError):
        return None
    return bounds if start <= end else None


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--output", type=Path, required=True)
    options = args.parse_args()
    import jionlp as jio

    source = Path(__file__).with_name("cases.json").read_bytes()
    fixture = json.loads(source)
    base = datetime.fromisoformat(fixture["clock"])
    results = []
    for case in fixture["cases"]:
        started = time.perf_counter()
        row = {"id": case["id"], "query": case["query"], "expected_range": case["expected_range"]}
        try:
            entities = jio.ner.extract_time(case["query"], time_base=base, with_parsing=False)
            parsed = [
                {"entity": entity, "parsed": jio.parse_time(entity["text"], time_base=base, strict=True)}
                for entity in entities
            ]
            ranges = [effective_range(item["parsed"]) for item in parsed]
            # No entities, a blurry entity or conflicting entities: no filter.
            unique = {tuple(value) for value in ranges if value is not None}
            actual = list(next(iter(unique))) if ranges and None not in ranges and len(unique) == 1 else None
            extracted = any(item["text"] == case["expression"] for item in entities)
            row.update(entities=parsed, extracted_target=extracted, actual_range=actual,
                       passed=extracted and actual == case["expected_range"])
        except Exception as error:
            row.update(error=f"{type(error).__name__}: {error}", passed=False)
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        results.append(row)
    report = {
        "jionlp_version": version("jionlp"), "python": platform.python_version(),
        "clock": fixture["clock"], "fixture_sha256": hashlib.sha256(source).hexdigest(),
        "passed": sum(row["passed"] for row in results), "total": len(results),
        "dependency_gate_passed": all(row["passed"] for row in results), "results": results,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))
    for row in results:
        if not row["passed"]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
