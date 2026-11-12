import subprocess
import sys
from pathlib import Path

import pytest

COMPARE = Path(__file__).resolve().parents[1] / "scripts/pi_parity/compare.py"


@pytest.mark.parametrize("extra", ["looplite", "pi"])
def test_extra_request_fails_comparison(tmp_path: Path, extra: str) -> None:
    for side in ("pi", "looplite"):
        (tmp_path / f"{side}-result.json").write_text('{"events":[],"executed":[],"messages":[]}')
        (tmp_path / f"request-{side}-1.json").write_text('{"messages":[]}')
    (tmp_path / f"request-{extra}-2.json").write_text('{"messages":[]}')
    result = subprocess.run([sys.executable, COMPARE, tmp_path], capture_output=True, text=True)
    assert result.returncode == 1
    assert "请求编号" in result.stdout and "2" in result.stdout
