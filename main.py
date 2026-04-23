from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ============================================================================
# Endpoints
# ============================================================================

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class Filters:
    days: int = 90

    # Market discovery
    events_limit: int = 100
    max_markets_to_analyze: int = 40

    # Public trades endpoint supports large limits
    trades_page_size: int = 500
    max_trade_pages_per_batch: int = 20

    # Wallet selection
    volume_min_usd: float = 2_000
    profit_min_usd: float = 300
    win_rate_min: float = 0.55
    nb_trades_min: int = 15
    nb_markets_min: int = 3

    # Anti-bot / anti-low-signal
    trades_per_week_max: float = 20
    price_min: float = 0.05
    price_max: float = 0.95
    avg_entry_price_max: float = 0.90
    edge_proxy_min: float = 0.03
    bet_ratio_min: float = 1.5


@dataclass(frozen=True)
class Weights:
    profit: float = 0.35
    win_rate: float = 0.20
    edge_proxy: float = 0.20
    timing: float = 0.15
    consistency: float = 0.10


FILTERS = Filters()
WEIGHTS = Weights()

CONSISTENCY_BONUS = 8
HIGH_PRICE_MALUS = 15

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 0.20


# ============================================================================
# Helpers
# ============================================================================

def utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    # Data API timestamps are often unix seconds / ms
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None

    if s.isdigit():
        try:
            ts = float(s)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.utcfromtimestamp(ts)
        except Exception:
            return None

    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "null"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default



def unique_trade_key(trade: dict[str, Any]) -> tuple:
    return (
        trade.get("transactionHash"),
        trade.get("conditionId"),
        trade.get("proxyWallet"),
        trade.get("side"),
        trade.get("price"),
        trade.get("size"),
        trade.get("timestamp"),
        trade.get("outcome"),
    )


# ============================================================================
# HTTP client
# ============================================================================

class APIClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/json",
            }
        )

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                if attempt == MAX_RETRIES:
                    print(f"[!] Timeout on {url}")
            except requests.exceptions.ConnectionError as exc:
                if attempt == MAX_RETRIES:
                    print(f"[!] Connection error on {url}: {exc}")
            except requests.exceptions.HTTPError as exc:
                body = ""
                try:
                    body = response.text[:300]
                except Exception:
                    pass
                print(f"[!] HTTP error on {url}: {exc} {body}")
                return None
            except Exception as exc:
                print(f"[!] Unexpected error on {url}: {exc}")
                return None

            time.sleep(0.8 * attempt)
        return None

    def get_active_events(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        data = self.get_json(
            f"{GAMMA_BASE}/events",
            params={
                "active": "true",
                "closed": "false",
                "order": "volume_24hr",
                "ascending": "false",
                "limit": limit,
                "offset": offset,
            },
        )
        return data if isinstance(data, list) else []

    def get_public_trades_for_market(
        self,
        market_id: str,
        limit: int,
        offset: int = 0,
        taker_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Une requête par marché — l'API Data n'accepte qu'un seul conditionId."""
        data = self.get_json(
            f"{DATA_BASE}/trades",
            params={
                "market": market_id,
                "limit": limit,
                "offset": offset,
                "takerOnly": str(taker_only).lower(),
            },
        )
        return data if isinstance(data, list) else []


# ============================================================================
# Domain models
# ============================================================================

@dataclass
class MarketMeta:
    condition_id: str
    question: str
    event_id: int | None
    created_at: datetime | None
    volume: float


@dataclass
class WalletStats:
    volume: float = 0.0
    amount_invested: float = 0.0
    gross_gains: float = 0.0
    wins: int = 0
    losses: int = 0
    trades: int = 0

    markets: set[str] = field(default_factory=set)
    timestamps: list[datetime] = field(default_factory=list)
    bet_sizes: list[float] = field(default_factory=list)
    entry_prices: list[float] = field(default_factory=list)
    edge_proxies: list[float] = field(default_factory=list)
    timing_scores: list[float] = field(default_factory=list)
    monthly_profit: dict[str, float] = field(default_factory=lambda: defaultdict(float))


# ============================================================================
# Market discovery
# ============================================================================

def flatten_top_markets_from_events(events: list[dict[str, Any]], max_markets: int) -> dict[str, MarketMeta]:
    markets: dict[str, MarketMeta] = {}

    for event in events:
        event_id = event.get("id")
        raw_markets = event.get("markets") or []

        for market in raw_markets:
            condition_id = str(market.get("conditionId") or market.get("id") or "").strip()
            if not condition_id or condition_id in markets:
                continue

            question = str(
                market.get("question")
                or event.get("title")
                or event.get("slug")
                or condition_id
            ).strip()

            created_at = parse_dt(
                market.get("createdAt")
                or market.get("created_at")
                or event.get("createdAt")
                or event.get("created_at")
            )

            volume = safe_float(
                market.get("volume")
                or market.get("volumeNum")
                or event.get("volume")
                or 0
            )

            markets[condition_id] = MarketMeta(
                condition_id=condition_id,
                question=question,
                event_id=event_id if isinstance(event_id, int) else None,
                created_at=created_at,
                volume=volume,
            )

            if len(markets) >= max_markets:
                return markets

    return markets


# ============================================================================
# Trade collection
# ============================================================================

def fetch_public_trades(
    client: APIClient,
    market_meta: dict[str, MarketMeta],
) -> list[dict[str, Any]]:
    market_ids = list(market_meta.keys())
    all_trades: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    if not market_ids:
        return all_trades

    for market_index, market_id in enumerate(market_ids, start=1):
        question = market_meta[market_id].question[:55]
        print(f"\n[{market_index:02d}/{len(market_ids)}] {question}")

        for page in range(FILTERS.max_trade_pages_per_batch):
            offset = page * FILTERS.trades_page_size
            trades = client.get_public_trades_for_market(
                market_id=market_id,
                limit=FILTERS.trades_page_size,
                offset=offset,
                taker_only=True,
            )

            if not trades:
                break

            fresh_count = 0
            for trade in trades:
                key = unique_trade_key(trade)
                if key not in seen:
                    seen.add(key)
                    all_trades.append(trade)
                    fresh_count += 1

            print(
                f"  page={page + 1:02d} offset={offset:<5} "
                f"received={len(trades):<4} new={fresh_count:<4} total={len(all_trades)}"
            )

            if len(trades) < FILTERS.trades_page_size:
                break

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_trades


# ============================================================================
# Scoring helpers
# ============================================================================

def edge_proxy(price: float, side: str) -> float:
    # Heuristic only, not true EV
    return 0.5 - price if side == "buy" else price - 0.5


def trade_frequency_per_week(timestamps: list[datetime]) -> float:
    if len(timestamps) < 2:
        return 0.0
    days = max((max(timestamps) - min(timestamps)).days, 1)
    return len(timestamps) / (days / 7)


def consistency_ratio(monthly_profit: dict[str, float]) -> float:
    if not monthly_profit:
        return 0.0
    positive_months = sum(1 for v in monthly_profit.values() if v > 0)
    return positive_months / len(monthly_profit)


def normalize_profit(profit: float) -> float:
    return math.log1p(max(profit, 0.0)) / math.log1p(100_000)


def timing_score_for_trade(ts: datetime, market_created_at: datetime | None) -> float:
    if market_created_at is None or ts < market_created_at:
        return 0.5

    total_age_h = max((utcnow_naive() - market_created_at).total_seconds() / 3600, 1.0)
    trade_delay_h = max((ts - market_created_at).total_seconds() / 3600, 0.0)
    return max(0.0, 1.0 - min(trade_delay_h / total_age_h, 1.0))


def composite_score(stats: dict[str, float]) -> float:
    score = (
        WEIGHTS.profit * normalize_profit(stats["profit_net"])
        + WEIGHTS.win_rate * stats["win_rate"]
        + WEIGHTS.edge_proxy * min(max(stats["edge_proxy_avg"], 0.0) / 0.3, 1.0)
        + WEIGHTS.timing * stats["timing_avg"]
        + WEIGHTS.consistency * stats["consistency"]
    ) * 100

    if stats["consistency"] >= 0.9:
        score += CONSISTENCY_BONUS
    if stats["avg_entry_price"] > FILTERS.avg_entry_price_max:
        score -= HIGH_PRICE_MALUS

    return round(min(max(score, 0.0), 100.0), 2)


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_wallets(
    trades: list[dict[str, Any]],
    market_meta: dict[str, MarketMeta],
) -> dict[str, WalletStats]:
    cutoff = utcnow_naive() - timedelta(days=FILTERS.days)
    wallets: dict[str, WalletStats] = {}

    for trade in trades:
        ts = parse_dt(trade.get("timestamp"))
        if ts is None or ts < cutoff:
            continue

        address = str(trade.get("proxyWallet") or "").strip()
        if not address:
            continue

        side = str(trade.get("side") or "").strip().lower()
        if side not in {"buy", "sell"}:
            continue

        condition_id = str(trade.get("conditionId") or "").strip()
        size = safe_float(trade.get("size"))
        price = safe_float(trade.get("price"))

        if size <= 0 or not (FILTERS.price_min <= price <= FILTERS.price_max):
            continue

        amount = size * price
        wallet = wallets.setdefault(address, WalletStats())

        wallet.volume += amount
        wallet.trades += 1
        wallet.markets.add(condition_id)
        wallet.timestamps.append(ts)
        wallet.bet_sizes.append(amount)
        wallet.entry_prices.append(price)
        wallet.edge_proxies.append(edge_proxy(price, side))

        meta = market_meta.get(condition_id)
        wallet.timing_scores.append(timing_score_for_trade(ts, meta.created_at if meta else None))

        month_key = ts.strftime("%Y-%m")

        # Heuristic P&L proxy only
        if side == "sell":
            pnl = size * (price - 0.5)
            wallet.gross_gains += pnl
            wallet.monthly_profit[month_key] += pnl
            if price > 0.5:
                wallet.wins += 1
            else:
                wallet.losses += 1
        else:
            wallet.amount_invested += amount

    return wallets


def rank_wallets(wallets: dict[str, WalletStats]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for address, wallet in wallets.items():
        decided = wallet.wins + wallet.losses
        if decided == 0 or wallet.trades < FILTERS.nb_trades_min:
            continue

        win_rate = wallet.wins / decided
        profit_net = wallet.gross_gains - wallet.amount_invested * 0.05
        edge_proxy_avg = sum(wallet.edge_proxies) / len(wallet.edge_proxies) if wallet.edge_proxies else 0.0
        avg_entry_price = sum(wallet.entry_prices) / len(wallet.entry_prices) if wallet.entry_prices else 0.0
        timing_avg = sum(wallet.timing_scores) / len(wallet.timing_scores) if wallet.timing_scores else 0.0
        consistency = consistency_ratio(wallet.monthly_profit)
        freq_week = trade_frequency_per_week(wallet.timestamps)
        nb_markets = len(wallet.markets)

        if len(wallet.bet_sizes) >= 2:
            min_bet = max(min(wallet.bet_sizes), 0.01)
            bet_ratio = max(wallet.bet_sizes) / min_bet
        else:
            bet_ratio = 1.0

        if wallet.volume < FILTERS.volume_min_usd:
            continue
        if profit_net < FILTERS.profit_min_usd:
            continue
        if win_rate < FILTERS.win_rate_min:
            continue
        if nb_markets < FILTERS.nb_markets_min:
            continue
        if freq_week > FILTERS.trades_per_week_max:
            continue
        if avg_entry_price > FILTERS.avg_entry_price_max:
            continue
        if edge_proxy_avg < FILTERS.edge_proxy_min:
            continue
        if bet_ratio < FILTERS.bet_ratio_min:
            continue

        stats = {
            "profit_net": profit_net,
            "win_rate": win_rate,
            "edge_proxy_avg": edge_proxy_avg,
            "timing_avg": timing_avg,
            "consistency": consistency,
            "avg_entry_price": avg_entry_price,
        }
        score = composite_score(stats)

        rows.append(
            {
                "rank": 0,
                "address": address,
                "score": score,
                "profit_net_usd": round(profit_net, 2),
                "win_rate_pct": round(win_rate * 100, 1),
                "edge_proxy_pct": round(edge_proxy_avg * 100, 2),
                "timing_avg": round(timing_avg, 2),
                "consistency_months_pct": round(consistency * 100, 0),
                "nb_trades": wallet.trades,
                "trades_per_week": round(freq_week, 1),
                "nb_markets": nb_markets,
                "volume_usd": round(wallet.volume, 2),
                "avg_bet_usd": round(wallet.volume / wallet.trades, 2),
                "bet_ratio": round(bet_ratio, 1),
                "avg_entry_price": round(avg_entry_price, 3),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ============================================================================
# Output
# ============================================================================

def print_top(df: pd.DataFrame, top_n: int = 20) -> None:
    if df.empty:
        print("No qualifying whale found with the current filters.")
        return

    cols = [
        "rank",
        "address",
        "score",
        "profit_net_usd",
        "win_rate_pct",
        "edge_proxy_pct",
        "timing_avg",
        "consistency_months_pct",
        "nb_trades",
        "trades_per_week",
        "nb_markets",
    ]
    display_df = df[cols].head(top_n).copy()
    display_df["address"] = display_df["address"].str[:14] + "..."

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.float_format", "{:.2f}".format)

    print()
    print(display_df.to_string(index=False))
    print()
    print("Legend:")
    print("  score                  : composite score /100")
    print("  profit_net_usd         : estimated net profit proxy")
    print("  win_rate_pct           : heuristic win-rate proxy")
    print("  edge_proxy_pct         : price-vs-center proxy, not true EV")
    print("  timing_avg             : 1.0 = very early entries")
    print("  consistency_months_pct : % of profitable months")
    print("  trades_per_week        : average weekly activity")


def export_results(df: pd.DataFrame) -> None:
    if df.empty:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = Path(f"whales_{stamp}.csv")
    json_path = Path(f"whales_{stamp}.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\nExports: {csv_path.name} | {json_path.name}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 64)
    print("  Polymarket Whale Tracker v4 (public Data API)")
    print(f"  Window: {FILTERS.days} days")
    print("=" * 64)
    print()

    client = APIClient()

    print("Fetching active events from Gamma API...")
    events = client.get_active_events(limit=FILTERS.events_limit, offset=0)
    if not events:
        print("No active events fetched. Check your connection.")
        return

    market_meta = flatten_top_markets_from_events(events, FILTERS.max_markets_to_analyze)
    if not market_meta:
        print("No markets extracted from active events.")
        return

    print(f"{len(market_meta)} markets selected for analysis.\n")
    for idx, meta in enumerate(market_meta.values(), start=1):
        print(f"[{idx:02d}/{len(market_meta)}] {meta.question[:65]:<65} vol={meta.volume:.0f}")

    print("\nFetching public trades from Data API...")
    trades = fetch_public_trades(client, market_meta)
    print(f"\n{len(trades)} unique public trades collected.")

    if not trades:
        print("No trades were returned by the public Data API for the selected markets.")
        print("Try increasing max_trade_pages_per_batch, max_markets_to_analyze, or disabling takerOnly.")
        return

    print("Aggregating wallets...")
    wallets = aggregate_wallets(trades, market_meta)
    print(f"{len(wallets)} unique wallets found.")

    print("\nRanking whales...\n")
    df = rank_wallets(wallets)

    if df.empty:
        print("No wallet matched the current filters.")
        print("Try relaxing: edge_proxy_min, bet_ratio_min, nb_markets_min, or volume_min_usd.")
        return

    print(f"{len(df)} qualifying wallet(s).")
    print_top(df, top_n=20)
    export_results(df)


if __name__ == "__main__":
    main()


