# dacos

> A humble Python library for Medium-Frequency Trading alpha research.  
> Very early stage, experimental, and definitely not production-ready.  
> Use at your own risk, and please be kind – we're learning here.

dacos (pronounced "dah-kos") is a personal project born out of curiosity about statistical arbitrage, mean-reversion, and momentum strategies in crypto markets. It aims to provide a simple, modular, and safe toolkit for building and testing alpha signals without blowing up your laptop.

## Why "dacos"?

It stands for **Da**ta **Co**nstruction **S**ystem, but honestly, it sounded cool and was available on PyPI (at the time of writing). 🤷

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
