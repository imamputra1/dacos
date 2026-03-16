"""
This module provides a single function `run_pairs_research` that orchestrates
the entire pipeline: ingestion, alignment, validation, engine execution, and
statistical metrics calculation. It uses Railway Oriented Programming with
Result monad to handle errors gracefully.
"""

from typing import Any

from dacos.core import (
    create_universe_aligner,
    create_universe_validator,
)
from dacos.laws import (
    calculate_adf_pvalue,
    calculate_halflife,
    calculate_hurst,
)
from dacos.paradigms import create_stat_arb_engine
from dacos.protocols import PathLike
from dacos.utils import Err, Ok, Result


def run_pairs_research(
    silver_path: PathLike,
    y_symbol: str,
    x_symbol: str,
    frequency: str = "1m",
    min_rows: int = 100,
    z_window: int = 100,
) -> Result[dict[str, Any], Exception]:
    """
    Execute the complete pairs research pipeline.

    Steps:
    1. Ingest data for the two symbols from the silver lake.
    2. Align to a regular time grid (upsample + forward fill).
    3. Validate data quality (length, nulls, stagnant coins).
    4. Run the StatArb engine to compute spread and rolling z‑score.
    5. Calculate statistical metrics (Hurst, ADF p‑value, half‑life) on the spread.

    Args:
        silver_path: Path to the silver lake (skinny table).
        y_symbol: Dependent (Y) symbol.
        x_symbol: Independent (X) symbol.
        frequency: Target frequency for alignment (e.g., "1m", "5s").
        min_rows: Minimum number of rows required after validation.
        z_window: Rolling window size for z‑score calculation.

    Returns:
        Ok(dict) with keys:
            - "data": Polars DataFrame containing timestamp, prices, spread, z_score.
            - "metrics": dict with Hurst, ADF p‑value, half‑life (None if calculation failed).
        Err(exception) if any pipeline step fails.
    """

    ingestor = create_universe_ingestor(silver_path)
    res_data = ingestor.load_universe([y_symbol, x_symbol])
    if res_data.is_err():
        return Err(res_data.unwrap_err())
    raw_data = res_data.unwrap()

    aligner = create_universe_aligner(frequency)
    res_aligned = aligner.align(raw_data)
    if res_aligned.is_err():
        return Err(res_aligned.unwrap_err())
    aligned_data = res_aligned.unwrap()

    validator = create_universe_validator(min_rows)
    res_valid = validator.validate(aligned_data)
    if res_valid.is_err():
        return Err(res_valid.unwrap_err())
    clean_data = res_valid.unwrap()

    engine = create_stat_arb_engine(z_window)
    res_engine = engine.run_engine(clean_data, y_symbol, x_symbol)
    if res_engine.is_err():
        return Err(res_engine.unwrap_err())
    final_data = res_engine.unwrap()

    spread_series = final_data.get_column("spread").drop_nulls()
    spread_array = spread_series.to_numpy()

    hurst_res = calculate_hurst(spread_array)
    adf_res = calculate_adf_pvalue(spread_array)
    hl_res = calculate_halflife(spread_array)

    metrics = {
        "hurst": hurst_res.ok() if hurst_res.is_ok() else None,
        "adf_pvalue": adf_res.ok() if adf_res.is_ok() else None,
        "halflife": hl_res.ok() if hl_res.is_ok() else None,
    }

    return Ok({"data": final_data, "metrics": metrics})
