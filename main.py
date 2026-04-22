"""
Polymarket Whale Tracker v2
============================
Critères de sélection :
  - Fréquence faible (pas de bot)
  - Cotes non triviales (exclusion des quasi-certains)
  - Edge de valeur attendue (EV)
  - Timing précoce sur les marchés
  - Diversification thématique
  - Constance sur la durée
  - Taille de mise variable selon conviction
"""

import requests
import pandas as pd
import math
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ─── Session HTTP ─────────────────────────────────────────────────────────────
SESSION = requests.Session()
# Décommentez si vous utilisez un proxy SOCKS (VPN) :
# SESSION.proxies = {
#     "http":  "socks5://127.0.0.1:1080",
#     "https": "socks5://127.0.0.1:1080",
# }
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
})

# ─── Endpoints ────────────────────────────────────────────────────────────────
CLOB_BASE  = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# ─── Configuration des filtres ────────────────────────────────────────────────
FILTRES = {
    # --- Filtres de base ---
    "jours":                 90,     # fenêtre d'analyse (3 mois pour la constance)
    "volume_min_usd":      2_000,    # volume total minimum sur la période
    "profit_min_usd":        300,    # profit net minimum estimé
    "win_rate_min":          0.55,   # au moins 55% de trades gagnants
    "nb_trades_min":          15,    # au moins 15 trades (significativité stat.)
    "nb_marches_min":          3,    # au moins 3 marchés différents (diversification)

    # --- Filtre fréquence (anti-bot) ---
    "trades_par_semaine_max": 20,    # max 20 trades/semaine en moyenne

    # --- Filtre cotes (anti quasi-certain) ---
    "prix_min":              0.05,   # ne pas parier en dessous de 5%
    "prix_max":              0.95,   # ne pas parier au dessus de 95%
    "prix_moyen_max":        0.9,   # prix moyen d'entrée max (exclut les arb.)

    # --- Filtre EV ---
    "ev_moyenne_min":        0.03,   # EV moyenne > 3% (edge positif)

    # --- Filtre mise variable (anti flat-betting) ---
    "ratio_mise_max_min":    1.5,    # ratio entre mise max et mise min > 1.5x
                                     # (preuve de conviction variable)
}

# ─── Pondération du score composite ──────────────────────────────────────────
POIDS = {
    "profit":     0.35,
    "win_rate":   0.20,
    "ev_moyenne": 0.20,
    "timing":     0.15,
    "constance":  0.10,
}

BONUS_CONSTANCE_MOIS = 8    # bonus si profitable sur 3 mois glissants
MALUS_PRIX_ELEVE     = 15   # malus si prix moyen > 85%


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(
            ts_str.replace("Z", "+00:00")
        ).replace(tzinfo=None)
    except Exception:
        return None


def get_json(url: str, params: dict = None, timeout: int = 20) -> dict | list | None:
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectTimeout:
        print(f"  [!] Timeout sur {url} — vérifiez votre VPN.")
    except requests.exceptions.ConnectionError as e:
        print(f"  [!] Connexion refusée : {e}")
    except Exception as e:
        print(f"  [!] Erreur {url}: {e}")
    return None


# ─── Collecte des données ─────────────────────────────────────────────────────

def get_active_markets(limit: int = 50) -> list[dict]:
    """Récupère les marchés actifs triés par volume décroissant."""
    data = get_json(f"{GAMMA_BASE}/markets", params={
        "active":     "true",
        "closed":     "false",
        "limit":      limit,
        "order":      "volume",
        "ascending":  "false",
    })
    if data is None:
        return []
    return data if isinstance(data, list) else data.get("markets", [])


def get_market_info(market_id: str) -> dict:
    """Récupère les métadonnées d'un marché (date création, volume, etc.)."""
    data = get_json(f"{GAMMA_BASE}/markets/{market_id}")
    return data if isinstance(data, dict) else {}


def get_market_trades(market_id: str, limit: int = 500) -> list[dict]:
    """Récupère les trades d'un marché via l'API CLOB."""
    data = get_json(f"{CLOB_BASE}/trades", params={
        "market": market_id,
        "limit":  limit,
    })
    if data is None:
        return []
    return data.get("data", []) if isinstance(data, dict) else []


# ─── Analyse par wallet ───────────────────────────────────────────────────────

def wallet_vide() -> dict:
    return {
        "volume":           0.0,
        "montant_investi":  0.0,
        "gains_bruts":      0.0,
        "wins":             0,
        "losses":           0,
        "trades":           0,
        "marches":          set(),
        "timestamps":       [],      # pour fréquence et constance
        "mises":            [],      # pour ratio max/min
        "prix_entrees":     [],      # pour prix moyen et EV
        "ev_list":          [],      # EV de chaque trade
        "timing_scores":    [],      # rang d'entrée par marché (0=1er, 1=dernier)
        "profits_par_mois": defaultdict(float),  # pour constance temporelle
    }


def calculer_ev(prix: float, side: str) -> float:
    """
    EV simplifiée par rapport à la valeur médiane (0.5).
    - Acheter 'Yes' à 0.30 : on estime que le marché sous-évalue → EV positive
    - Acheter 'Yes' à 0.85 : quasi-certain, EV proche de 0 voire négative
    EV = |prix - 0.5| comme proxy de l'écart au consensus.
    Négatif si on bet dans le sens attendu (prix > 0.5 pour buy).
    """
    if side == "buy":
        return 0.5 - prix    # positif si on achète sous-évalué (prix < 0.5)
    else:
        return prix - 0.5    # positif si on vend sur-évalué (prix > 0.5)


def analyser_trades(
    tous_les_trades: list[dict],
    market_meta: dict[str, dict],
) -> dict[str, dict]:
    """
    Agrège tous les trades par adresse wallet.
    market_meta : {market_id: {"created_at": ..., "total_trades_count": ...}}
    """
    cutoff = utcnow() - timedelta(days=FILTRES["jours"])
    wallets: dict[str, dict] = {}

    # Index des trades par marché pour calculer le timing
    trades_par_marche: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for t in tous_les_trades:
        mid  = t.get("market", "")
        ts   = parse_ts(t.get("timestamp") or t.get("created_at", ""))
        addr = t.get("maker_address") or t.get("taker_address", "")
        if ts and addr:
            trades_par_marche[mid].append((ts, addr))

    # Trier chaque marché chronologiquement
    for mid in trades_par_marche:
        trades_par_marche[mid].sort(key=lambda x: x[0])

    # Calcul du rang de chaque (marché, wallet) → timing_score entre 0 et 1
    timing_par_marche_wallet: dict[tuple, float] = {}
    for mid, entries in trades_par_marche.items():
        n = len(entries)
        for idx, (_, addr) in enumerate(entries):
            key = (mid, addr)
            if key not in timing_par_marche_wallet:
                timing_par_marche_wallet[key] = idx / max(n - 1, 1)

    # Agrégation principale
    for t in tous_les_trades:
        ts_str = t.get("timestamp") or t.get("created_at", "")
        ts     = parse_ts(ts_str)
        if ts is None or ts < cutoff:
            continue

        addr  = t.get("maker_address") or t.get("taker_address", "")
        if not addr:
            continue

        side  = t.get("side", "").lower()
        size  = float(t.get("size",  0))
        price = float(t.get("price", 0))
        mid   = t.get("market", "")

        # Filtre cotes non triviales
        if not (FILTRES["prix_min"] <= price <= FILTRES["prix_max"]):
            continue

        amount = size * price

        if addr not in wallets:
            wallets[addr] = wallet_vide()

        w = wallets[addr]
        w["volume"]      += amount
        w["trades"]      += 1
        w["marches"].add(mid)
        w["timestamps"].append(ts)
        w["mises"].append(amount)
        w["prix_entrees"].append(price)

        # EV du trade
        ev = calculer_ev(price, side)
        w["ev_list"].append(ev)

        # P&L estimé
        if side == "sell":
            pnl = size * (price - 0.5)
            w["gains_bruts"] += pnl
            mois_key = ts.strftime("%Y-%m")
            w["profits_par_mois"][mois_key] += pnl
            if price > 0.5:
                w["wins"]  += 1
            else:
                w["losses"] += 1
        else:
            w["montant_investi"] += amount

        # Timing score (0 = entré en 1er, 1 = entré en dernier)
        timing = timing_par_marche_wallet.get((mid, addr))
        if timing is not None:
            w["timing_scores"].append(1.0 - timing)  # on inverse : 1 = très tôt

    return wallets


# ─── Scoring ──────────────────────────────────────────────────────────────────

def calculer_constance(profits_par_mois: dict) -> float:
    """
    Retourne la fraction de mois où le wallet était profitable.
    Ex: 2 mois profitables sur 3 → 0.67
    """
    if not profits_par_mois:
        return 0.0
    mois_positifs = sum(1 for v in profits_par_mois.values() if v > 0)
    return mois_positifs / len(profits_par_mois)


def calculer_frequence_hebdo(timestamps: list[datetime]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duree_jours = (max(timestamps) - min(timestamps)).days or 1
    semaines = duree_jours / 7
    return len(timestamps) / semaines


def calculer_score(stats: dict) -> float:
    """Score composite sur 100 avec bonus/malus."""
    profit    = max(stats["profit_net"], 0)
    win_rate  = stats["win_rate"]
    ev_moy    = max(stats["ev_moyenne"], 0)
    timing    = stats["timing_moyen"]
    constance = stats["constance"]

    norm_profit   = math.log1p(profit)   / math.log1p(100_000)
    norm_ev       = min(ev_moy / 0.3, 1.0)
    norm_timing   = timing
    norm_constance = constance

    score = (
        POIDS["profit"]     * norm_profit    +
        POIDS["win_rate"]   * win_rate       +
        POIDS["ev_moyenne"] * norm_ev        +
        POIDS["timing"]     * norm_timing    +
        POIDS["constance"]  * norm_constance
    ) * 100

    # Bonus constance 3 mois
    if constance >= 0.9:
        score += BONUS_CONSTANCE_MOIS

    # Malus prix moyen élevé (arb. quasi-certain)
    if stats["prix_moyen"] > FILTRES["prix_moyen_max"]:
        score -= MALUS_PRIX_ELEVE

    return round(min(max(score, 0), 100), 2)


# ─── Filtrage et classement ───────────────────────────────────────────────────

def filtrer_et_classer(wallets: dict) -> pd.DataFrame:
    rows = []

    for addr, w in wallets.items():
        total_decided = w["wins"] + w["losses"]
        if total_decided == 0 or w["trades"] < FILTRES["nb_trades_min"]:
            continue

        win_rate   = w["wins"] / total_decided
        profit_net = w["gains_bruts"] - w["montant_investi"] * 0.05
        ev_moyenne = sum(w["ev_list"]) / len(w["ev_list"]) if w["ev_list"] else 0
        prix_moyen = sum(w["prix_entrees"]) / len(w["prix_entrees"]) if w["prix_entrees"] else 0
        timing_moy = sum(w["timing_scores"]) / len(w["timing_scores"]) if w["timing_scores"] else 0
        constance  = calculer_constance(w["profits_par_mois"])
        freq_hebdo = calculer_frequence_hebdo(w["timestamps"])
        nb_marches = len(w["marches"])

        # Ratio mise max / min (conviction variable)
        mises = w["mises"]
        ratio_mises = max(mises) / max(min(mises), 0.01) if len(mises) >= 2 else 1.0

        # ─── Application des filtres ───
        if w["volume"]    < FILTRES["volume_min_usd"]:      continue
        if profit_net     < FILTRES["profit_min_usd"]:      continue
        if win_rate       < FILTRES["win_rate_min"]:        continue
        if nb_marches     < FILTRES["nb_marches_min"]:      continue
        if freq_hebdo     > FILTRES["trades_par_semaine_max"]: continue
        if prix_moyen     > FILTRES["prix_moyen_max"]:      continue
        if ev_moyenne     < FILTRES["ev_moyenne_min"]:      continue
        if ratio_mises    < FILTRES["ratio_mise_max_min"]:  continue

        stats = {
            "profit_net":  profit_net,
            "win_rate":    win_rate,
            "ev_moyenne":  ev_moyenne,
            "timing_moyen": timing_moy,
            "constance":   constance,
            "prix_moyen":  prix_moyen,
        }
        score = calculer_score(stats)

        rows.append({
            "rang":              0,
            "adresse":           addr,
            "score":             score,
            "profit_net_usd":    round(profit_net, 2),
            "win_rate_pct":      round(win_rate * 100, 1),
            "ev_moyenne_pct":    round(ev_moyenne * 100, 2),
            "timing_moyen":      round(timing_moy, 2),
            "constance_mois":    round(constance * 100, 0),
            "nb_trades":         w["trades"],
            "freq_trades_semaine": round(freq_hebdo, 1),
            "nb_marches":        nb_marches,
            "volume_usd":        round(w["volume"], 2),
            "avg_mise_usd":      round(w["volume"] / w["trades"], 2),
            "ratio_mises":       round(ratio_mises, 1),
            "prix_moyen_entree": round(prix_moyen, 3),
        })

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    df["rang"] = df.index + 1
    return df


# ─── Affichage ────────────────────────────────────────────────────────────────

def afficher_top(df: pd.DataFrame, n: int = 20) -> None:
    if df.empty:
        return
    cols_affichage = [
        "rang", "adresse", "score",
        "profit_net_usd", "win_rate_pct", "ev_moyenne_pct",
        "timing_moyen", "constance_mois",
        "nb_trades", "freq_trades_semaine", "nb_marches",
    ]
    df_display = df[cols_affichage].head(n).copy()
    df_display["adresse"] = df_display["adresse"].str[:14] + "..."
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df_display.to_string(index=False))
    print()
    print("Légende des colonnes :")
    print("  score            : score composite /100")
    print("  profit_net_usd   : profit net estimé en USD")
    print("  win_rate_pct     : % de trades gagnants")
    print("  ev_moyenne_pct   : edge moyen par trade (%)")
    print("  timing_moyen     : 1.0 = entre toujours en 1er, 0 = toujours en dernier")
    print("  constance_mois   : % de mois profitable sur la période")
    print("  freq_trades_sem  : nombre moyen de trades par semaine")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Polymarket Whale Tracker v2")
    print(f"  Fenêtre : {FILTRES['jours']} jours")
    print("=" * 55)
    print()

    print("Récupération des marchés actifs...")
    marches = get_active_markets(limit=40)
    if not marches:
        print("Aucun marché récupéré. Vérifiez votre VPN / connexion.")
        return
    print(f"  {len(marches)} marchés trouvés\n")

    tous_les_trades: list[dict] = []
    market_meta:     dict[str, dict] = {}

    for i, m in enumerate(marches):
        mid  = m.get("conditionId") or m.get("id", "")
        name = (m.get("question") or mid)[:55]
        vol  = m.get("volume", 0)
        print(f"[{i+1:02d}/{len(marches)}] {name:<55}  vol={vol}")

        trades = get_market_trades(mid, limit=500)
        tous_les_trades.extend(trades)

        market_meta[mid] = {
            "created_at":         m.get("createdAt") or m.get("created_at", ""),
            "total_trades_count": m.get("tradesCount", len(trades)),
        }
        time.sleep(0.35)

    print(f"\n{len(tous_les_trades)} trades collectés au total.")

    print("Analyse par wallet...")
    wallets = analyser_trades(tous_les_trades, market_meta)
    print(f"  {len(wallets)} wallets uniques trouvés.\n")

    print("Classement des baleines (application des filtres)...\n")
    df = filtrer_et_classer(wallets)

    if df.empty:
        print("Aucune baleine trouvée avec ces filtres.")
        print("Essayez d'assouplir : ev_moyenne_min, ratio_mise_max_min ou nb_marches_min.")
        return

    print(f"  {len(df)} baleine(s) qualifiée(s)\n")
    afficher_top(df, n=20)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    df.to_csv(f"whales_{ts}.csv",  index=False)
    df.to_json(f"whales_{ts}.json", orient="records", indent=2)
    print(f"\nExports : whales_{ts}.csv  |  whales_{ts}.json")


if __name__ == "__main__":
    main()