# dacos

> A Python library for Medium-Frequency Trading alpha research.  
> Very early stage, experimental, and definitely not production-ready.  
> Use at your own risk, and please be kind – we're learning here.

dacos (pronounced "dalam-kost") is a personal project born out of curiosity about statistical arbitrage, mean-reversion, and momentum strategies in crypto markets. It aims to provide a simple, modular, and safe toolkit for building and testing alpha signals without blowing up your laptop.

## Why "dacos"?

It stands for **Da**ta **Co**nstruction **S**ystem, but honestly, and it’s available on PyPI—coming soon (as of this writing). 

## Features (so far)

- **ETL Pipeline** – Build "skinny" tables from raw Parquet files using Polars lazy evaluation. No RAM overload, I promise.
- **Asynchronous Time Alignment** – A pure-Polars metronome (`upsample`) that perfectly aligns different coin timestamps using forward-fill physics (No look-ahead bias).
- **StatArb Engine & Physics Lab** – Built-in OLS Rolling Z-Score calculation, Hurst Exponent, ADF Stationarity Test, and Ornstein-Uhlenbeck Half-life.
- **Result Monad** – Explicit error handling (Rust-style `Ok`/`Err`) so your code doesn't crash unexpectedly in the middle of a multi-million row calculation.
- **Type Hints Everywhere** – Because my ADHD brain forgets what a function returns after 5 minutes.

## Installation

```bash
# From PyPI 
pip install dacos
```

## Status
This is a learning project. I'm not a quant, not a data engineer, just someone who enjoys messing around with data and trading ideas. The code works (for me), but it's full of imperfections, rough edges, and probably a few bugs.

If you're an experienced developer and spot something horrible, feel free to open an issue or PR – but please be gentle.

Roadmap
    [x] ETL pipeline (extract, filter, log transform)
    [x] Result monad for error handling
    [x] Data alignment (as-of join for multiple symbols)
    [x] Statistical tests (Hurst, ADF, half-life)
    [x] Pairs trading engine (Rolling Z-Score)
    [ ] Momentum signals
    [ ] Execution Tactics (Position Sizing & Entry/Exit Logic)
    [ ] Backtesting framework (maybe)

## How to Use
dacos is designed to hide complex data engineering behind a simple API.
1. Extracting Data
```python
from dacos.builder import create_skinny_builder

# Point to your raw Parquet folder and where you want the silver table
builder = create_skinny_builder("data/raw", "data/silver")
result = builder.execute_pipeline()

if result.is_ok():
    print("Success! Skinny table created.")
else:
    print(f"Oh no: {result.error()}")
```

2. Pairs Research
```python
import dacos.api as dc

# 1. Run the entire pipeline (Ingestion -> Alignment -> Validation -> Math -> Z-Score)
res = dc.run_pairs_research(
    silver_path="data/silver",
    y_symbol="WIF_USDT",
    x_symbol="PEPE_USDT",
    frekuensi="1m",
    z_window=100
)

if res.is_ok():
    output = res.unwrap()
    df_zscore = output["data"]
    metrics = output["metrics"]
    
    print(f"Hurst Exponent: {metrics['hurst']}")
    print(f"Half-Life: {metrics['halflife']} minutes")
    print(df_zscore.tail(5)) # View the latest Z-Scores!
```

# Point to your raw Parquet folder and where you want the silver table
builder = create_skinny_builder("data/raw", "data/silver")
result = builder.execute_pipeline()

if result.is_ok():
    print("Success! Skinny table created.")
else:
    print(f"Oh no: {result.error()}")

## Contributing
Contributions are welcome! But please keep in mind:

    I'm still learning, so my code style might be weird.
    Write tests if you add Features.
    Be nice.

License
MIT – do whatever you want, but don't blame me if you lose money trading.
---

### PERUBAHAN UTAMA:
1.  **Daftar Fitur:** Saya menambahkan *Time Alignment* dan *StatArb Engine*. Ini adalah "nilai jual" terbesar *library* Anda saat ini.
2.  **Roadmap:** Saya mengubahnya menjadi *Checkbox* (`[x]`). Ini memberikan kepuasan psikologis bagi pembaca (dan bagi Anda!) bahwa proyek ini aktif dan setengah dari fiturnya sudah selesai.
3.  **How To Use:** Saya membaginya menjadi 2 bagian. Bagian kedua adalah pameran fungsi `run_pairs_research` dari `api.py` yang akan membuat *user* PyPI kagum betapa mudahnya meriset 2 koin dengan `dacos`.

### Next Steps for You:

Silakan timpa *file* `README.md` Anda dengan draf di atas.

Jika "Brosur Etalase" (Task 3.1) ini sudah rapi, apakah Anda siap mengeksekusi **Task 3.2 (`uv build`)** dan **Task 3.3 (`uv publish`)** untuk meresmikan `dacos` v0.1.1 ke seluruh dunia?
