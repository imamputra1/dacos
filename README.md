# dacos

> A Python library for Medium-Frequency Trading alpha research.  
> Very early stage, experimental, and definitely not production-ready.  
> Use at your own risk, and please be kind – we're learning here.

dacos (pronounced "dalam-kost") is a personal project born out of curiosity about statistical arbitrage, mean-reversion, and momentum strategies in crypto markets. It aims to provide a simple, modular, and safe toolkit for building and testing alpha signals without blowing up your laptop.

## Why "dacos"?

It stands for **Da**ta **Co**nstruction **S**ystem, but honestly, and it’s available on PyPI—coming soon (as of this writing). 

## Features (so far)

- **ETL Pipeline** – Build "skinny" tables from raw Parquet files using Polars lazy evaluation. No RAM overload, I promise.
- **Result Monad** – Explicit error handling (Rust-style `Ok`/`Err`) so your code doesn't crash unexpectedly.
- **Type Hints Everywhere** – Because my ADHD brain forgets what a function returns after 5 minutes.
- **Configuration Constants** – No magic numbers; all thresholds live in one file.

## Installation

```bash
# From PyPI (when published)
pip install dacos

# Or directly from GitHub
pip install git+https://github.com/yourusername/dacos.git
```

## Status
This is a learning project. I'm not a quant, not a data engineer, just someone who enjoys messing around with data and trading ideas. The code works (fo me), but it's full of imperfections, rough edges, and probably a few bugs.

If you're an experienced developer and spot something horrible, feel free to open an issue or PR – but please be gentle.

## Roadmap 
- ETL pipeline (extract, filter, log transform)
- Result monad for error handling
- Data alignment (as-of join for multiple symbols)
- Statistical tests (Hurst, ADF, half-life)
- Pairs trading engine
- Momentum signals
- Backtesting framework (maybe)

## Contributing
Contributions are welcome! But please keep in mind:

    I'm still learning, so my code style might be weird.
    Write tests if you add Features.
    Be nice.

License
MIT – do whatever you want, but don't blame me if you lose money trading.

## How to Use
```python
from dacos.builder import create_skinny_builder

# Point to your raw Parquet folder and where you want the skinny table
builder = create_skinny_builder("data/raw", "data/silver")

# Run the pipeline – it returns a Result object
result = builder.execute_pipeline()

if result.is_ok():
    print("Success! Skinny table created.")
else:
    print(f"Oh no: {result.error()}")
```

