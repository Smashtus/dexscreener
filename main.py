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
from typing import Any, Dict, Optional

from dexscreener import DexscreenerClient
from solders.pubkey import Pubkey as PublicKey
try:
    # SerdeJSONError moved under solders.errors in newer versions
    from solders.errors import SerdeJSONError
except ImportError:  # pragma: no cover - fallback for older versions
    from solders import SerdeJSONError  # type: ignore[attr-defined]
from solana.rpc.async_api import AsyncClient
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

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

        self.start_time = time.time()
        self.entry_price: Optional[float] = None
        self.price_history: list[tuple[float, float]] = []
        self.liquidity_history: list[tuple[float, float]] = []

        self.dex_state: Dict[str, Any] = {}
        self.onchain_state: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        # Stop flag used when on-chain data cannot be fetched
        self._onchain_error: bool = False

    async def fetch_dex_data(self) -> None:
        """Pull latest market data from Dexscreener."""
        try:
            pair = await self.dex_client.get_token_pair_async(self.chain, self.pair_address)
            if not pair:
                logging.warning("Pair %s not found on chain %s", self.pair_address, self.chain)
                return

            now = time.time()
            price = pair.price_usd or 0.0
            if self.entry_price is None:
                self.entry_price = price
            self.price_history.append((now, price))

            if pair.liquidity and pair.liquidity.usd is not None:
                self.liquidity_history.append((now, pair.liquidity.usd))

            tx = pair.transactions.m5
            total_tx = tx.buys + tx.sells
            buy_pct = tx.buys / total_tx * 100 if total_tx else 0.0
            sell_pct = tx.sells / total_tx * 100 if total_tx else 0.0

            self.dex_state = {
                "price_usd": price,
                "buy_volume_pct": buy_pct,
                "sell_volume_pct": sell_pct,
                "whale_sell_pct": 0.0,
                "liquidity_usd": pair.liquidity.usd if pair.liquidity else 0.0,
                "sol_reserve": pair.liquidity.base if pair.liquidity else 0.0,
                "token_reserve": pair.liquidity.quote if pair.liquidity else 0.0,
                "fdv": pair.fdv,
                "market_cap": pair.fdv,
            }
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

            resp = await self.rpc_client.get_account_info(
                pubkey, encoding="jsonParsed"
            )
            info = resp.get("result", {}).get("value")
            if info and info.get("data"):
                parsed = info["data"]["parsed"]["info"]
                freeze = parsed.get("freezeAuthority")
                mint = parsed.get("mintAuthority")
                self.onchain_state = {
                    "freeze_authority": freeze,
                    "mint_authority": mint,
                }
        except SerdeJSONError as exc:  # pragma: no cover - malformed response
            logging.error(
                "Solana RPC returned malformed JSON for token %s: %s",
                self.token_address,
                exc,
            )
            self._onchain_error = True
        except Exception as exc:  # pragma: no cover - network failure
            logging.exception("Solana RPC error: %s", exc)
            self._onchain_error = True

    def compute_metrics(self) -> None:
        """Compute derived metrics from historical data."""
        if len(self.price_history) >= 2:
            current_price = self.price_history[-1][1]
            price_change = (
                (current_price - self.entry_price) / self.entry_price * 100
                if self.entry_price
                else 0.0
            )
            prices = [p for _, p in self.price_history[-30:]]
            mean_price = statistics.fmean(prices) if prices else 0.0
            volatility = (
                (max(prices) - min(prices)) / mean_price * 100 if mean_price else 0.0
            )
            if len(self.price_history) >= 3:
                t0, p0 = self.price_history[-3]
                t1, p1 = self.price_history[-2]
                t2, p2 = self.price_history[-1]
                slope1 = (p1 - p0) / (t1 - t0) if t1 != t0 else 0.0
                slope2 = (p2 - p1) / (t2 - t1) if t2 != t1 else 0.0
                acceleration = (
                    (slope2 - slope1) / (t2 - t1) if t2 != t1 else 0.0
                )
            else:
                acceleration = 0.0

            self.metrics.update(
                {
                    "price_change_pct": price_change,
                    "price_acceleration": acceleration,
                    "price_volatility": volatility,
                }
            )

        if len(self.liquidity_history) >= 2:
            t0, l0 = self.liquidity_history[-2]
            t1, l1 = self.liquidity_history[-1]
            drain_rate = (
                ((l1 - l0) / l0) / (t1 - t0) * 100 if l0 and t1 != t0 else 0.0
            )
            redeemed_pct = max(0.0, (l0 - l1) / l0 * 100) if l0 else 0.0
            flash_crash = redeemed_pct > 50
            self.metrics.update(
                {
                    "liquidity_drain_rate": drain_rate,
                    "lp_tokens_redeemed_pct": redeemed_pct,
                    "flash_crash_detected": flash_crash,
                    "flash_crash_magnitude_pct": redeemed_pct if flash_crash else 0.0,
                }
            )

    def build_payload(self) -> Dict[str, Any]:
        """Construct JSON payload sent to the LLM."""
        now = time.time()
        current_price = self.price_history[-1][1] if self.price_history else 0.0
        payload = {
            "token_metadata": {
                "token_address": self.token_address,
                "pair_address": self.pair_address,
                "amm_program": self.amm_program,
                "freeze_authority": self.onchain_state.get("freeze_authority"),
                "mint_authority": self.onchain_state.get("mint_authority"),
            },
            "price_analysis": {
                "time_since_entry_sec": int(now - self.start_time),
                "entry_price_usd": self.entry_price or 0.0,
                "current_price_usd": current_price,
                "price_change_pct_since_entry": self.metrics.get(
                    "price_change_pct", 0.0
                ),
                "price_acceleration_pct_per_sec": self.metrics.get(
                    "price_acceleration", 0.0
                ),
                "price_volatility_pct": self.metrics.get("price_volatility", 0.0),
            },
            "volume_analysis": {
                "instant_volume_window_sec": 60,
                "buy_volume_pct": self.dex_state.get("buy_volume_pct", 0.0),
                "sell_volume_pct": self.dex_state.get("sell_volume_pct", 0.0),
                "recent_whale_sell_pct": self.dex_state.get("whale_sell_pct", 0.0),
            },
            "liquidity_pool_analysis": {
                "liquidity_formula": "CPMM: x * y = k",
                "sol_reserve": self.dex_state.get("sol_reserve", 0.0),
                "token_reserve": self.dex_state.get("token_reserve", 0.0),
                "current_liquidity_usd": self.dex_state.get("liquidity_usd", 0.0),
                "lp_tokens_redeemed_pct": self.metrics.get(
                    "lp_tokens_redeemed_pct", 0.0
                ),
                "liquidity_drain_rate_pct_per_sec": self.metrics.get(
                    "liquidity_drain_rate", 0.0
                ),
            },
            "market_evaluation": {
                "market_cap_usd": self.dex_state.get("market_cap"),
                "fully_diluted_valuation_usd": self.dex_state.get("fdv"),
                "holders_count": 0,
                "top_holder_pct": 0.0,
            },
            "rug_risk_indicators": {
                "freeze_authority_status": "None (Safe)"
                if not self.onchain_state.get("freeze_authority")
                else "Set (Risk)",
                "mint_authority_status": "None (Safe)"
                if not self.onchain_state.get("mint_authority")
                else "Set (Risk)",
                "deployer_wallet_lp_redemption": {
                    "recent_lp_redemptions_pct": self.metrics.get(
                        "lp_tokens_redeemed_pct", 0.0
                    ),
                    "status": "HIGH Risk"
                    if self.metrics.get("lp_tokens_redeemed_pct", 0.0) > 50
                    else "LOW Risk",
                },
                "flash_crash_detected": self.metrics.get("flash_crash_detected", False),
                "flash_crash_magnitude_pct": self.metrics.get(
                    "flash_crash_magnitude_pct", 0.0
                ),
            },
        }
        return payload

    async def send_to_llm(self) -> None:
        """Send metrics to GPT-4 and log the recommendation."""
        if not self.openai:
            logging.warning("OpenAI API key not set; skipping LLM call")
            return

        payload = self.build_payload()
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

    async def _llm_loop(self) -> None:
        while True:
            self.compute_metrics()
            await self.send_to_llm()
            await asyncio.sleep(30)

    async def run(self) -> None:
        await asyncio.gather(
            self._dex_loop(),
            self._onchain_loop(),
            self._llm_loop(),
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
    args = parser.parse_args()

    try:
        asyncio.run(main(args.mint_address))
    except KeyboardInterrupt:
        logging.info("Shutting down...")
