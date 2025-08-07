"""Real-time rug-risk detection pipeline for Solana tokens.

This script polls Dexscreener for market data and Solana RPC for on-chain
information to build a structured JSON payload. The payload is periodically
sent to an LLM (OpenAI GPT-4) which returns trade recommendations.
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from dexscreener import DexscreenerClient
from solders.pubkey import Pubkey as PublicKey
try:
    # SerdeJSONError moved under solders.errors in newer versions
    from solders.errors import SerdeJSONError
except ImportError:  # pragma: no cover - fallback for older versions
    from solders import SerdeJSONError  # type: ignore[attr-defined]
from solana.rpc.async_api import AsyncClient
from openai import AsyncOpenAI


SYSTEM_PROMPT = (
    "You are an expert crypto analyst tasked to evaluate short-term trades of "
    "high-risk Solana tokens, explicitly focusing on detecting imminent rug "
    "pulls. Given structured data (price metrics, liquidity analysis, rug-risk "
    "indicators) every 30 seconds, provide a recommendation for immediate "
    "action:\n\n- HOLD: Momentum and conditions still safe for gains.\n- "
    "TAKE_PROFIT: Immediate gains capture required due to increasing risk. "
    "Provide a suggested profit-taking percentage.\n- EXIT_NOW: Clear rug "
    "indicators detected. Exit immediately."
)


class RugRiskMonitor:
    """Monitor a single SPL token for potential rug risk."""

    def __init__(self, chain: str, pair: str, token: str, amm_program: str) -> None:
        self.chain = chain
        self.pair_address = pair
        self.token_address = token
        self.amm_program = amm_program

        self.dex_client = DexscreenerClient()
        self.rpc_client = AsyncClient("https://api.mainnet-beta.solana.com")

        api_key = os.getenv("OPENAI_API_KEY")
        self.openai: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=api_key) if api_key else None
        )

        self.snapshots: Deque[Dict[str, Any]] = deque(maxlen=60)
        self.onchain_state: Dict[str, Any] = {}

        # Aggregation control
        self._last_aggregate = time.time()

        # Stop flag used when on-chain data cannot be fetched
        self._onchain_error: bool = False
        # Track consecutive on-chain fetch failures
        self._onchain_error_count: int = 0

    async def fetch_dex_data(self) -> None:
        """Pull latest market data from Dexscreener."""
        try:
            pair = await self.dex_client.get_token_pair_async(
                self.chain, self.pair_address
            )
            logging.debug("Dexscreener response for %s: %s", self.pair_address, pair)
            if not pair:
                logging.warning("Pair %s not found on chain %s", self.pair_address, self.chain)
                return

            now = int(time.time())
            snapshot = {
                "timestamp": now,
                "current_price_usd": pair.price_usd or 0.0,
                "liquidity_usd": pair.liquidity.usd if pair.liquidity and pair.liquidity.usd else 0.0,
                "buy_tx_count_5m": pair.transactions.m5.buys,
                "sell_tx_count_5m": pair.transactions.m5.sells,
                "volume_usd_5m": pair.volume.m5 or 0.0,
                "base_reserve": pair.liquidity.base if pair.liquidity else 0.0,
                "quote_reserve": pair.liquidity.quote if pair.liquidity else 0.0,
                "fdv_usd": pair.fdv or 0.0,
            }
            self.snapshots.append(snapshot)
            logging.info("Snapshot stored: %s", snapshot)
        except Exception as exc:  # pragma: no cover - network failure
            logging.exception("Dexscreener API error: %s", exc)
    async def fetch_onchain_data(self) -> None:
        """Fetch freeze/mint authorities from Solana RPC."""
        try:
            try:
                pubkey = PublicKey.from_string(self.token_address)
            except ValueError:
                logging.error("Invalid token address: %s", self.token_address)
                return

            resp = await self.rpc_client.get_account_info_json_parsed(pubkey)
            logging.debug(
                "RPC account info for %s: %s", self.token_address, resp
            )
            info = resp.value
            if info and getattr(info, "data", None) and hasattr(info.data, "parsed"):
                token_info = info.data.parsed.get("info", {})
                freeze = token_info.get("freezeAuthority")
                mint = token_info.get("mintAuthority")
                self.onchain_state = {
                    "freeze_authority": freeze,
                    "mint_authority": mint,
                }
                logging.info(
                    "Token authorities: freeze=%s mint=%s",
                    freeze,
                    mint,
                )
                # Reset error counter on success
                self._onchain_error_count = 0
        except SerdeJSONError as exc:  # pragma: no cover - malformed response
            logging.error(
                "Solana RPC returned malformed JSON for token %s: %s",
                self.token_address,
                exc,
            )
            self._onchain_error_count += 1
            if self._onchain_error_count >= 3:
                self._onchain_error = True
        except Exception as exc:  # pragma: no cover - network failure
            logging.exception("Solana RPC error: %s", exc)
            self._onchain_error_count += 1
            if self._onchain_error_count >= 3:
                self._onchain_error = True

    def _snapshot_n_seconds_ago(self, seconds: int) -> Optional[Dict[str, Any]]:
        """Return snapshot from `seconds` ago if available."""
        if not self.snapshots:
            return None
        target = self.snapshots[-1]["timestamp"] - seconds
        for snap in reversed(self.snapshots):
            if snap["timestamp"] <= target:
                return snap
        return None

    def _pct_change(self, field: str, seconds: int) -> float:
        past = self._snapshot_n_seconds_ago(seconds)
        if not past:
            return 0.0
        current = self.snapshots[-1][field]
        previous = past[field]
        return ((current - previous) / previous * 100) if previous else 0.0

    def aggregate_snapshots(self) -> Dict[str, Any]:
        """Aggregate metrics over stored snapshots."""
        current = self.snapshots[-1]

        # Price metrics
        price_change = {
            "5s": self._pct_change("current_price_usd", 5),
            "10s": self._pct_change("current_price_usd", 10),
            "30s": self._pct_change("current_price_usd", 30),
        }
        price_accel = {
            "5s": price_change["5s"] / 5,
            "10s": price_change["10s"] / 10,
        }
        prices_30 = [s["current_price_usd"] for s in self.snapshots if s["timestamp"] >= current["timestamp"] - 30]
        price_vol = (
            statistics.pstdev(prices_30) / statistics.fmean(prices_30) * 100
            if len(prices_30) > 1 and statistics.fmean(prices_30)
            else 0.0
        )

        # Liquidity metrics
        liquidity_change = {
            "5s": self._pct_change("liquidity_usd", 5),
            "10s": self._pct_change("liquidity_usd", 10),
            "30s": self._pct_change("liquidity_usd", 30),
        }
        liq10 = self._snapshot_n_seconds_ago(10)
        if liq10 and liq10["liquidity_usd"]:
            drain_rate = (
                (liq10["liquidity_usd"] - current["liquidity_usd"]) / liq10["liquidity_usd"] / 10 * 100
            )
        else:
            drain_rate = 0.0
        liqs_30 = [s["liquidity_usd"] for s in self.snapshots if s["timestamp"] >= current["timestamp"] - 30]
        liq_vol = (
            statistics.pstdev(liqs_30) / statistics.fmean(liqs_30) * 100
            if len(liqs_30) > 1 and statistics.fmean(liqs_30)
            else 0.0
        )

        # Transaction metrics
        def tx_ratio(seconds: int) -> float:
            past = self._snapshot_n_seconds_ago(seconds)
            if not past:
                return 0.0
            buy_delta = max(0, current["buy_tx_count_5m"] - past["buy_tx_count_5m"])
            sell_delta = max(0, current["sell_tx_count_5m"] - past["sell_tx_count_5m"])
            total = buy_delta + sell_delta
            return (buy_delta / total * 100) if total else 0.0

        buy_sell_ratio = {
            "5s": tx_ratio(5),
            "10s": tx_ratio(10),
            "30s": tx_ratio(30),
        }
        volume_change = {
            "5s": self._pct_change("volume_usd_5m", 5),
            "10s": self._pct_change("volume_usd_5m", 10),
            "30s": self._pct_change("volume_usd_5m", 30),
        }

        return {
            "price_analysis": {
                "current_price_usd": current["current_price_usd"],
                "price_change_pct": price_change,
                "price_acceleration_pct_per_sec": price_accel,
                "price_volatility_pct_30s": price_vol,
            },
            "liquidity_analysis": {
                "current_liquidity_usd": current["liquidity_usd"],
                "liquidity_change_pct": liquidity_change,
                "liquidity_drain_rate_pct_per_sec_10s": drain_rate,
                "liquidity_volatility_pct_30s": liq_vol,
            },
            "transaction_analysis": {
                "buy_sell_tx_ratio_pct": buy_sell_ratio,
                "buy_sell_volume_change_pct": volume_change,
            },
            "market_evaluation": {
                "fully_diluted_valuation_usd": current["fdv_usd"],
            },
        }

    def build_payload(self) -> Dict[str, Any]:
        aggregates = self.aggregate_snapshots()
        payload = {
            "token_metadata": {
                "token_address": self.token_address,
                "pair_address": self.pair_address,
                "amm_program": self.amm_program,
            },
            **aggregates,
        }
        return payload

    async def send_to_llm(self, payload: Dict[str, Any]) -> None:
        """Send metrics to GPT-4 and log the recommendation."""
        if not self.openai:
            logging.info("Aggregated payload: %s", json.dumps(payload))
            print(json.dumps(payload, indent=2))
            return

        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content
            logging.info("LLM response: %s", content)
        except Exception as exc:  # pragma: no cover - network failure
            logging.exception("OpenAI API error: %s", exc)

    async def _dex_loop(self) -> None:
        while True:
            await self.fetch_dex_data()
            await asyncio.sleep(1)

    async def _onchain_loop(self) -> None:
        while not self._onchain_error:
            await self.fetch_onchain_data()
            await asyncio.sleep(5)

    async def _aggregate_loop(self) -> None:
        """Aggregate snapshots every 30 seconds and output analysis."""
        while True:
            if self.snapshots and time.time() - self._last_aggregate >= 30:
                payload = self.build_payload()
                await self.send_to_llm(payload)
                self._last_aggregate = time.time()
            await asyncio.sleep(1)

    async def run(self) -> None:
        await asyncio.gather(
            self._dex_loop(),
            self._onchain_loop(),
            self._aggregate_loop(),
        )


def build_amm_program() -> str:
    name = os.getenv("AMM_NAME", "Unknown AMM")
    program_id = os.getenv("PROGRAM_ID", "Unknown")
    return f"{name} (Program ID: {program_id})"


async def main(token_address: str) -> None:
    try:
        PublicKey.from_string(token_address)
    except ValueError:
        logging.error("Invalid token address: %s", token_address)
        return

    client = DexscreenerClient()
    pairs = await client.get_token_pairs_async(token_address)
    if not pairs:
        logging.error("No pairs found for token %s", token_address)
        return

    # Select the pair with the highest USD liquidity
    pair = max(
        pairs,
        key=lambda p: (p.liquidity.usd if p.liquidity and p.liquidity.usd else 0),
    )

    amm_program = build_amm_program()

    monitor = RugRiskMonitor(
        pair.chain_id, pair.pair_address, token_address, amm_program
    )
    await monitor.run()


if __name__ == "__main__":  # pragma: no cover - script entry point
    parser = argparse.ArgumentParser(description="Solana rug risk monitor")
    parser.add_argument("mint_address", help="SPL token mint address to monitor")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for troubleshooting",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        asyncio.run(main(args.mint_address))
    except KeyboardInterrupt:
        logging.info("Shutting down...")
