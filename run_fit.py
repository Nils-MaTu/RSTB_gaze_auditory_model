#!/usr/bin/env python3
"""
Fit and validate receptive field shift (model 1) and retinocentric expansion (model 2).
"""

from __future__ import annotations

import json
from pathlib import Path

from gaze_models import fit_and_validate


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    results = fit_and_validate(data_dir)

    out_path = base_dir / "model_parameters.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    model1 = results["model1"]
    model2 = results["model2"]
    lrt = results["lrt"]
    val = results["validation"]

    print("Fitted parameters:")
    print(f"  Model 1: alpha={model1['alpha']:.6f}, LL={model1['log_likelihood']:.4f}")
    print(
        f"  Model 2: alpha={model2['alpha']:.6f}, beta={model2['beta']:.6f}, we={model2['we']:.6f}, "
        f"LL={model2['log_likelihood']:.4f}"
    )
    print("LRT:")
    print(f"  chi2={lrt['chi2']:.4f}, df={int(lrt['df'])}, p={lrt['p_value']:.6f}")
    print("Validation (Lewald):")
    print(
        f"  Model 1: r={val['model1']['correlation']:.4f}, RMSE={val['model1']['rmse']:.4f}"
    )
    print(
        f"  Model 2: r={val['model2']['correlation']:.4f}, RMSE={val['model2']['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
