#!/usr/bin/env python3
"""Sequential OpenAI API acceptance test that detects the known token-0 bug."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


CASES = (
    ("中国的首都是哪里？只回答城市名。", ("北京",)),
    ("17乘以23等于多少？只回答数字。", ("391",)),
    ("法国的首都是哪里？只回答城市名。", ("巴黎", "Paris", "paris")),
)


def post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"body": body}
        return error.code, parsed


def collect_completion_token_ids(response: dict[str, Any]) -> list[int]:
    result: list[int] = []
    choices = response.get("choices")
    if not isinstance(choices, list):
        return result
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        token_ids = choice.get("token_ids")
        if isinstance(token_ids, list):
            result.extend(item for item in token_ids if isinstance(item, int))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--check-logprobs", action="store_true")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the three-case sequence this many times.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    failures: list[str] = []
    request_number = 0
    for round_number in range(1, args.repeat + 1):
        for case_number, (prompt, expected) in enumerate(CASES, 1):
            request_number += 1
            payload: dict[str, Any] = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 32,
                "return_token_ids": True,
            }
            if args.check_logprobs:
                payload.update(logprobs=True, top_logprobs=1)
            status, response = post(f"{args.base_url}/chat/completions", payload)
            content = ""
            try:
                content = response["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                pass
            token_ids = collect_completion_token_ids(response)
            print(
                json.dumps(
                    {
                        "round": round_number,
                        "case": case_number,
                        "request": request_number,
                        "status": status,
                        "content": content,
                        "token_ids": token_ids,
                    },
                    ensure_ascii=False,
                )
            )

            label = f"round {round_number} case {case_number}"
            if status != 200:
                failures.append(f"{label}: HTTP {status}: {response}")
            if token_ids and all(token_id == 0 for token_id in token_ids):
                failures.append(f"{label}: every completion token id is 0")
            if content.count("<｜begin▁of▁sentence｜>") >= 2:
                failures.append(f"{label}: repeated BOS token")
            if not any(term in content for term in expected):
                failures.append(
                    f"{label}: expected one of {expected!r}, got {content!r}"
                )

    if failures:
        print("Sequential smoke test FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("Sequential smoke test passed.")


if __name__ == "__main__":
    main()
