"""DNSE LightSpeed market-data stream (wss://ws-openapi.dnse.com.vn/v1/stream) -> data/dnse/<day>.jsonl, one raw frame per line
(binary/msgpack frames, if the server sends them, are kept as {"b64": ...}).

Keys: DNSE_API_KEY and DNSE_API_SECRET in the git-ignored .env at the repo root (or the environment). They never leave
vnstock_oapi, which signs the handshake and refuses order/account channels. Runs on ~/.venv_6x (vnstock 6 + `vnstock-oapi[stream]`).

Use (HOSE session 09:00-15:00, stop with Ctrl-C or SIGTERM):
  ~/.venv_6x/bin/python -m ai_core.fusion.dnse_stream                       # tick.G1 for every symbol of the universe
  ~/.venv_6x/bin/python -m ai_core.fusion.dnse_stream ohlc_closed.1D FPT HPG
Channel names: tick, tick_extra, top_price, security_definition, foreign, ohlc.<res>, ohlc_closed.<res> (see streams.toml).
"""
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime

from vnstock_oapi.stream import StreamClient

from ai_core import paths
from ai_core.prices import config as PC

log = logging.getLogger(__name__)


def subscribe(client, spec, symbols):
    """'tick' -> tick.G1 (HOSE board); a name with a dot is used as given (e.g. 'ohlc_closed.1D')."""
    return client.subscribe(spec if "." in spec else f"{spec}.G1", symbols)


async def run(spec, symbols, out_dir=paths.DNSE, client=None):
    client = subscribe(client or StreamClient("dnse"), spec, symbols)
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, client.stop)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, last = 0, time.monotonic()
    async for f in client.frames():
        if f.kind != "data":
            log.info("%s %s", f.kind, f.detail)
            continue
        day = datetime.now().strftime("%Y-%m-%d")
        with open(out_dir / f"{day}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"recv_at_ns": f.recv_at_ns, "channel": f.channel, "msg": f.json() if f.encoding == "text" else {"b64": f.to_dict()["payload"]}}, ensure_ascii=False) + "\n")
        n += 1
        if time.monotonic() - last >= 10:
            last = time.monotonic()
            log.info("%d frames so far, last %s", n, f.channel)
    return n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    spec = sys.argv[1] if len(sys.argv) > 1 else "tick"
    symbols = sys.argv[2:] or sorted(PC.TICKERS)
    try:
        log.info("frames written: %d", asyncio.run(run(spec, symbols)))
    except KeyboardInterrupt:
        pass
