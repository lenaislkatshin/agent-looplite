"""比对两边的结果：事件序列、工具执行、消息和请求体。有约定之外的差异时退出码为 1。

    python compare.py <结果目录>

请求体里约定的差异见 `docs/design.md`“与 Pi 的差异”中的“Chat Completions 请求体”：Pi 多发
`max_completion_tokens` 和 `store`，用户消息发内容块数组。比对前先去掉这两个字段，把用户消息的
内容块拼成字符串。
"""

import difflib
import json
import pathlib
import sys
from typing import Any

PI_ONLY_FIELDS = {"max_completion_tokens", "store"}


def normalize_pi_request(body: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in body.items() if k not in PI_ONLY_FIELDS}
    messages = []
    for message in body["messages"]:
        if message["role"] == "user" and isinstance(message["content"], list):
            text = "".join(part["text"] for part in message["content"] if part["type"] == "text")
            message = {**message, "content": text}
        messages.append(message)
    return {**body, "messages": messages}


def main() -> int:
    out = pathlib.Path(sys.argv[1])
    pi = json.loads((out / "pi-result.json").read_text())
    lo = json.loads((out / "looplite-result.json").read_text())
    problems = 0

    diff = list(difflib.unified_diff(pi["events"], lo["events"], "Pi TS", "looplite", lineterm=""))
    print(f"事件：Pi {len(pi['events'])} 个，本库 {len(lo['events'])} 个，", end="")
    print("逐条一致" if not diff else "有差异")
    if diff:
        print("\n".join(diff))
        problems += 1

    if pi["executed"] == lo["executed"]:
        print("工具执行：一致")
    else:
        print(f"工具执行：不一致 {pi['executed']} {lo['executed']}")
        problems += 1

    pi_messages = [m for m in pi["messages"] if m["role"] != "system"]
    if len(pi_messages) != len(lo["messages"]):
        print(f"消息条数不一致：Pi {len(pi_messages)}，本库 {len(lo['messages'])}")
        problems += 1
    for a, b in zip(pi_messages, lo["messages"], strict=False):
        different = sorted(k for k in set(a) | set(b) if k != "error" and a.get(k) != b.get(k))
        print(f"消息 {a['role']}：" + ("一致" if not different else f"不一致 {different}"))
        problems += bool(different)

    number = 1
    while (out / f"request-pi-{number}.json").exists():
        a = normalize_pi_request(json.loads((out / f"request-pi-{number}.json").read_text()))
        b = json.loads((out / f"request-looplite-{number}.json").read_text())
        fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        print(
            f"第 {number} 次请求体：" + ("约定差异之外一致" if not fields else f"不一致 {fields}")
        )
        problems += bool(fields)
        number += 1

    return 1 if problems else 0


sys.exit(main())
