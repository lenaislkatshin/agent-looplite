"""本地 SSE 服务：按路径的第一段区分调用方，第 1 次请求回放录制的分片，之后回一段固定的回答。

    python server.py <分片文件> <端口> <请求体输出目录>

分片文件是 JSON 数组，每个元素是一个 `data:` 后面的分片对象。收到的请求体写到输出目录，
文件名是 `request-<调用方>-<序号>.json`。
"""

import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RECORDED = json.loads(pathlib.Path(sys.argv[1]).read_text())
OUTPUT = pathlib.Path(sys.argv[3])


def chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


FOLLOW_UP = [
    chunk({"role": "assistant", "content": ""}),
    chunk({"content": "两者限速相差"}),
    chunk({"content": " 10 公里/小时。"}),
    chunk({}, "stop"),
    {
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"prompt_tokens": 120, "completion_tokens": 10, "total_tokens": 130},
    },
]
counts: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        caller = self.path.strip("/").split("/")[0]
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        counts[caller] = counts.get(caller, 0) + 1
        number = counts[caller]
        (OUTPUT / f"request-{caller}-{number}.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=1)
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for item in RECORDED if number == 1 else FOLLOW_UP:
            self.wfile.write(f"data: {json.dumps(item, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True

    def log_message(self, *args: object) -> None:
        pass


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()
