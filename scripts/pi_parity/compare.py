"""比对两边的结果：事件序列、工具执行、消息和请求体。

python compare.py <结果目录>
"""

import difflib
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
pi = json.loads((out / "pi-result.json").read_text())
lo = json.loads((out / "looplite-result.json").read_text())

diff = list(difflib.unified_diff(pi["events"], lo["events"], "Pi TS", "looplite", lineterm=""))
print(
    f"事件：Pi {len(pi['events'])} 个，本库 {len(lo['events'])} 个，"
    + ("逐条一致" if not diff else "有差异")
)
print("\n".join(diff))
print(
    "工具执行："
    + ("一致" if pi["executed"] == lo["executed"] else f"不一致 {pi['executed']} {lo['executed']}")
)

pi_messages = [m for m in pi["messages"] if m["role"] != "system"]
for a, b in zip(pi_messages, lo["messages"], strict=True):
    different = {k for k in set(a) | set(b) if k != "error" and a.get(k) != b.get(k)}
    print(f"消息 {a['role']}：" + ("一致" if not different else f"不一致 {sorted(different)}"))

for number in (1, 2):
    a = json.loads((out / f"request-pi-{number}.json").read_text())
    b = json.loads((out / f"request-looplite-{number}.json").read_text())
    fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k != "messages")
    messages = [
        i for i, (x, y) in enumerate(zip(a["messages"], b["messages"], strict=True)) if x != y
    ]
    print(f"第 {number} 次请求体：不同的字段 {fields}，不同的消息序号 {messages}")
