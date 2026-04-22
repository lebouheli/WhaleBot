# Polymarket Whale Tracker v2

## Overview

This project is a Python script designed to identify and rank "whales" on Polymarket based on their trading behavior.

Instead of simply tracking large wallets, it applies a set of behavioral filters to detect traders who may have a consistent edge:
- Non-trivial pricing (avoiding near-certain bets)
- Early market entry (timing advantage)
- Positive expected value proxy
- Consistent profitability
- Diversification across markets
- Variable bet sizing (conviction-based)

The output is a ranked list of wallets with a composite score.

---

## Features

- Fetches active Polymarket markets (via Gamma API)
- Collects trade data (via CLOB API)
- Aggregates trades by wallet
- Computes multiple behavioral metrics:
  - Estimated profit
  - Win rate (proxy)
  - Timing score
  - Consistency over time
  - Trade frequency
  - Bet size variability
- Applies filters to remove bots and low-signal wallets
- Ranks wallets using a composite score
- Exports results to CSV and JSON

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/polymarket-whale-tracker.git
cd polymarket-whale-tracker