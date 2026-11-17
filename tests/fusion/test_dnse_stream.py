import asyncio
import json

import pytest

pytest.importorskip("vnstock_oapi.stream")  # only in ~/.venv_6x

from vnstock_oapi.stream import Frame  # noqa: E402

from ai_core.fusion import dnse_stream  # noqa: E402


class FakeClient:
    def __init__(self):
        self.subs = []

    def subscribe(self, channel, symbols):
        self.subs.append((channel, symbols))
        return self

    def stop(self):
        pass

    async def frames(self):
        yield Frame("dnse", "", 1, 1, 0, b"", kind="connected")
        yield Frame("dnse", "tick", 1, 2, 7, b'{"T":"t","s":"FPT","p":100}')


def test_run_writes_data_frames_only(tmp_path):
    c = FakeClient()
    assert asyncio.run(dnse_stream.run("tick", ["FPT"], tmp_path, c)) == 1
    assert c.subs == [("tick.G1", ["FPT"])]
    (f,) = tmp_path.glob("*.jsonl")
    assert json.loads(f.read_text()) == {"recv_at_ns": 7, "channel": "tick", "msg": {"T": "t", "s": "FPT", "p": 100}}
