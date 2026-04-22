import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# ─── Configuration ────────────────────────────────────────────────────────────
CLOB_BASE    = "https://clob.polymarket.com"
GAMMA_BASE   = "https://gamma-api.polymarket.com"

FILTRES = {
    "volume_min_usd":    1_000,   # volume total minimum
    "profit_min_usd":    500,     # profit net minimum
    "win_rate_min":      0.50,    # 50% de trades gagnants
    "nb_trades_min":     10,      # au moins 10 trades
    "jours":             30,      # fenêtre de 30 jours
}

SCORE_POIDS = {
    "profit":    0.50,
    "win_rate":  0.30,
    "volume":    0.20,
}
# ──────────────────────────────────────────────────────────────────────────────


def get_market_trades(market_id: str, limit: int = 500) -> list[dict]:
    """Récupère les trades d'un marché via l'API CLOB."""
    url = f"{CLOB_BASE}/trades"
    params = {"market": market_id, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"  [!] Erreur trades {market_id}: {e}")
        return []


def get_active_markets(limit: int = 50) -> list[dict]:
    """Récupère les marchés actifs avec le plus de liquidité."""
    url = f"{GAMMA_BASE}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume",
        "ascending": "false",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[!] Erreur marchés: {e}")
        return []


def analyser_trades(trades: list[dict], jours: int) -> dict[str, dict]:
    """
    Agrège les trades par adresse wallet.
    Retourne un dict {adresse: {profit, volume, wins, total_trades, ...}}
    """
    cutoff = datetime.utcnow() - timedelta(days=jours)
    wallets: dict[str, dict] = {}

    for t in trades:
        # Filtre temporel
        ts_str = t.get("timestamp") or t.get("created_at", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        if ts < cutoff:
            continue

        addr   = t.get("maker_address") or t.get("taker_address", "")
        if not addr:
            continue

        side   = t.get("side", "").lower()   # "buy" ou "sell"
        size   = float(t.get("size",  0))    # nb de shares
        price  = float(t.get("price", 0))    # prix par share (0-1)
        amount = size * price                 # montant USD investi

        if addr not in wallets:
            wallets[addr] = {
                "volume": 0.0,
                "montant_investi": 0.0,
                "gains":  0.0,
                "wins":   0,
                "losses": 0,
                "trades": 0,
                "marches": set(),
        }

        w = wallets[addr]
        w["volume"]  += amount
        w["trades"]  += 1
        w["marches"].add(t.get("market", ""))

        # Estimation P&L simplifiée :
        # - "sell" à prix > 0.5 → gain potentiel (position favorable)
        # - "buy"  à prix < 0.5 → position à faible coût
        if side == "sell":
            pnl = size * (price - 0.5)   # vs valeur médiane
            w["gains"] += pnl
            if price > 0.5:
                w["wins"] += 1
            else:
                w["losses"] += 1
        else:
            w["montant_investi"] += amount

    return wallets


def calculer_score(w: dict) -> float:
    """Score composite normalisé sur 100."""
    profit    = max(w["profit_net"], 0)
    win_rate  = w["win_rate"]
    volume    = w["volume"]

    # Normalisation logarithmique pour profit et volume
    import math
    norm_profit   = math.log1p(profit)   / math.log1p(50_000)
    norm_volume   = math.log1p(volume)   / math.log1p(500_000)
    norm_win_rate = win_rate

    score = (
        SCORE_POIDS["profit"]   * norm_profit   +
        SCORE_POIDS["win_rate"] * norm_win_rate +
        SCORE_POIDS["volume"]   * norm_volume
    ) * 100

    return round(min(score, 100), 2)


def filtrer_et_classer(wallets: dict) -> pd.DataFrame:
    """Applique les filtres et produit le classement final."""
    rows = []
    for addr, w in wallets.items():
        total = w["wins"] + w["losses"]
        if total == 0:
            continue

        win_rate  = w["wins"] / total
        profit_net = w["gains"] - w["montant_investi"] * 0.05  # frais ~5%

        # Filtres
        if w["volume"]  < FILTRES["volume_min_usd"]:  continue
        if profit_net   < FILTRES["profit_min_usd"]:  continue
        if win_rate     < FILTRES["win_rate_min"]:     continue
        if w["trades"]  < FILTRES["nb_trades_min"]:   continue

        w["profit_net"] = profit_net
        w["win_rate"]   = win_rate
        score = calculer_score(w)

        rows.append({
            "adresse":        addr,
            "score":          score,
            "profit_net_usd": round(profit_net, 2),
            "win_rate_pct":   round(win_rate * 100, 1),
            "volume_usd":     round(w["volume"], 2),
            "nb_trades":      w["trades"],
            "nb_marches":     len(w["marches"]),
            "avg_mise_usd":   round(w["volume"] / w["trades"], 2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def main():
    print("=== Polymarket Whale Tracker ===\n")

    print("📡 Récupération des marchés actifs...")
    marches = get_active_markets(limit=30)
    print(f"   → {len(marches)} marchés trouvés\n")

    tous_les_trades = []
    for i, m in enumerate(marches):
        mid  = m.get("conditionId") or m.get("id", "")
        name = m.get("question", mid)[:60]
        print(f"[{i+1}/{len(marches)}] {name}")
        trades = get_market_trades(mid, limit=500)
        tous_les_trades.extend(trades)
        time.sleep(0.3)  # respecter le rate limit

    print(f"\n📊 {len(tous_les_trades)} trades collectés")

    print("🔍 Analyse par wallet...")
    wallets = analyser_trades(tous_les_trades, jours=FILTRES["jours"])
    print(f"   → {len(wallets)} wallets uniques\n")

    print("🏆 Classement des baleines...")
    df = filtrer_et_classer(wallets)

    if df.empty:
        print("Aucune baleine trouvée avec ces filtres. Essayez de les assouplir.")
        return

    # Affichage
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.head(20).to_string(index=False))

    # Export
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    df.to_csv(f"whales_{ts}.csv", index=False)
    df.to_json(f"whales_{ts}.json", orient="records", indent=2)
    print(f"\n✅ Fichiers exportés : whales_{ts}.csv / .json")


if __name__ == "__main__":
    main()