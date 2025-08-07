import asyncio
import pathlib
import sys

from solders.errors import SerdeJSONError

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from main import RugRiskMonitor


class FailingClient:
    async def get_account_info(self, *args, **kwargs):
        raise SerdeJSONError("bad json")


def test_onchain_stops_after_consecutive_errors(monkeypatch):
    monitor = RugRiskMonitor(
        "solana",
        "pair",
        "So11111111111111111111111111111111111111112",
        "amm",
    )
    monkeypatch.setattr(monitor, "rpc_client", FailingClient())

    async def run():
        for _ in range(3):
            await monitor.fetch_onchain_data()

    asyncio.run(run())
    assert monitor._onchain_error is True
