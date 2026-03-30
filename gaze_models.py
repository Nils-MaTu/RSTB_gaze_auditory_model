"""Gaze-auditory model: fitting + Lewald validation.

This file defines channel models (left/right/opponent), cue inversion for ILD/ITD,
and the maximum-likelihood objective used in `fit_and_validate()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import fsolve, minimize
from scipy.stats import chi2

# ============================================================================
# Fixed model constants
# ============================================================================

SPEED_OF_SOUND = 343.0
HEAD_RADIUS = 0.09
FREQUENCY = 4000.0

ILD_THRESHOLD = 5.9
ITD_THRESHOLD = 217.0

WS = 46.0


# ============================================================================
# Core channel / cue functions (sigmoid opponent channel)
# ============================================================================


def left_channel(
    azimuth: np.ndarray | float,
    gaze: float,
    alpha: float,
    beta: float | None = None,
    we: float | None = None,
) -> np.ndarray:
    """Left channel response (sigmoid), using gaze-dependent RF shift/expansion."""
    azimuth = np.asarray(azimuth)
    c = alpha * gaze

    if beta is not None and we is not None:
        E = beta * np.exp(-((azimuth - gaze) ** 2) / (2.0 * we**2))
        ws_eff = WS / (1.0 + E)
    else:
        ws_eff = WS

    # Sigmoid: 1 / (1 + exp((azimuth - c)/ws))
    return 1.0 / (1.0 + np.exp((azimuth - c) / ws_eff))


def right_channel(
    azimuth: np.ndarray | float,
    gaze: float,
    alpha: float,
    beta: float | None = None,
    we: float | None = None,
) -> np.ndarray:
    """Right channel response (sigmoid), using gaze-dependent RF shift/expansion."""
    azimuth = np.asarray(azimuth)
    c = alpha * gaze

    if beta is not None and we is not None:
        E = beta * np.exp(-((azimuth - gaze) ** 2) / (2.0 * we**2))
        ws_eff = WS / (1.0 + E)
    else:
        ws_eff = WS

    # Sigmoid: 1 / (1 + exp(-(azimuth - c)/ws))
    return 1.0 / (1.0 + np.exp(-(azimuth - c) / ws_eff))


def opponent_channel(
    azimuth: np.ndarray | float,
    gaze: float,
    alpha: float,
    beta: float | None = None,
    we: float | None = None,
) -> np.ndarray:
    """Opponent channel signal (R - L)."""
    R = right_channel(azimuth, gaze, alpha, beta, we)
    L = left_channel(azimuth, gaze, alpha, beta, we)
    return R - L


def opponent_scalar(
    azimuth: float,
    gaze: float,
    alpha: float,
    beta: float | None = None,
    we: float | None = None,
) -> float:
    """Opponent channel for a single azimuth (as a Python float)."""
    return float(np.asarray(opponent_channel(azimuth, gaze, alpha, beta, we)).flat[0])


def calculate_ild_sine_approx(
    azimuth_rad: float | np.ndarray, freq: float = FREQUENCY
) -> np.ndarray:
    """
    Sine-based ILD approximation.
    """
    wavelength = SPEED_OF_SOUND / freq
    k = 2.0 * np.pi / wavelength
    freq_weight = 1.0 if freq >= 1500.0 else 0.3
    alpha = 1.0 + k * HEAD_RADIUS
    ild_base = freq_weight * alpha * np.sin(azimuth_rad)
    ild_unscaled = ild_base * 30.0 / (alpha * freq_weight)
    return ild_unscaled


def calculate_ild(
    azimuth_rad: float | np.ndarray,
    freq: float = FREQUENCY,
) -> np.ndarray:
    """
    ILD calculation for the fixed release model.

    Azimuth inversion uses `azimuth_from_ild()`, and this ILD path is computed as
    a broadband log-sum over `calculate_ild_sine_approx`.
    """
    sqrt2 = math.sqrt(2.0)
    # 1-octave band: [freq/sqrt(2), freq*sqrt(2)]
    freqs = np.logspace(np.log10(freq / sqrt2), np.log10(freq * sqrt2), 50)

    ild_linear_sum = np.zeros_like(np.asarray(azimuth_rad), dtype=float)
    # Each sampled frequency gives the same ILD in this approximation, but we keep
    # broadband averaging for consistency with this model's pipeline.
    for f in freqs:
        ild_f = calculate_ild_sine_approx(azimuth_rad, f)
        ild_linear_sum += 10.0 ** (ild_f / 20.0)

    avg_ratio = ild_linear_sum / len(freqs)
    # Guard; avg_ratio should be > 0.
    return (
        20.0 * np.log10(avg_ratio)
        if np.all(avg_ratio > 0)
        else np.where(avg_ratio > 0, 20.0 * np.log10(avg_ratio), 0.0)
    )


def calculate_itd(azimuth_rad: float | np.ndarray) -> np.ndarray:
    """ITD using Woodworth formula."""
    return (HEAD_RADIUS / SPEED_OF_SOUND) * (azimuth_rad + np.sin(azimuth_rad)) * 1e6


def azimuth_from_ild(ild_target: float) -> float:
    """Invert broadband ILD -> azimuth using `fsolve`."""

    def equation(azimuth_rad: np.ndarray) -> np.ndarray:
        ild_calc = calculate_ild(azimuth_rad, FREQUENCY)
        return ild_calc - ild_target

    try:
        result = fsolve(equation, 0.0)[0]
        return float(np.degrees(np.clip(result, -np.pi / 2.0, np.pi / 2.0)))
    except (ValueError, RuntimeError):
        return float("nan")


def azimuth_from_itd(itd_target: float) -> float:
    """Invert ITD -> azimuth using `fsolve`."""

    def equation(azimuth_rad: np.ndarray) -> np.ndarray:
        itd_calc = calculate_itd(azimuth_rad)
        return itd_calc - itd_target

    try:
        result = fsolve(equation, 0.0)[0]
        return float(np.degrees(np.clip(result, -np.pi / 2.0, np.pi / 2.0)))
    except (ValueError, RuntimeError):
        return float("nan")


def predict_discrimination_sensitivity_maddox(
    gaze1: float,
    sound1_centre: float,
    gaze2: float,
    sound2_centre: float,
    alpha: float,
    beta: float,
    we: float,
    cue_type: str,
) -> float:
    """
    Predict % change in discrimination sensitivity for Maddox (opponent spread).
    """
    offset_at_25 = 12.0 if cue_type == "ild" else 231.0
    threshold = ILD_THRESHOLD if cue_type == "ild" else ITD_THRESHOLD

    def get_discrimination_positions(sound_dir: float) -> tuple[float, float]:
        center_cue = offset_at_25 if abs(sound_dir - 25.0) < 1.0 else 0.0
        cue_minus = center_cue - threshold / 2.0
        cue_plus = center_cue + threshold / 2.0

        if cue_type == "ild":
            az_minus = azimuth_from_ild(cue_minus)
            az_plus = azimuth_from_ild(cue_plus)
        else:
            az_minus = azimuth_from_itd(cue_minus)
            az_plus = azimuth_from_itd(cue_plus)

        return az_minus, az_plus

    az_1_minus, az_1_plus = get_discrimination_positions(sound1_centre)
    az_2_minus, az_2_plus = get_discrimination_positions(sound2_centre)

    opp_1_minus = opponent_scalar(az_1_minus, gaze1, alpha, beta, we)
    opp_1_plus = opponent_scalar(az_1_plus, gaze1, alpha, beta, we)
    opp_2_minus = opponent_scalar(az_2_minus, gaze2, alpha, beta, we)
    opp_2_plus = opponent_scalar(az_2_plus, gaze2, alpha, beta, we)

    change_1 = abs(opp_1_plus - opp_1_minus)
    change_2 = abs(opp_2_plus - opp_2_minus)

    return ((change_1 - change_2) / change_2) * 100.0 if change_2 != 0.0 else 0.0


def predict_discrimination_sensitivity_best(
    gaze: float,
    sound_target: float,
    alpha: float,
    beta: float,
    we: float,
) -> float:
    """
    Predict discrimination sensitivity for Best2023 (opponent spread version).
    """
    speakers = np.array([-30.0, -15.0, 0.0, 15.0, 30.0], dtype=float)

    O_all = opponent_channel(speakers, gaze, alpha, beta, we)
    O_target = opponent_scalar(sound_target, gaze, alpha, beta, we)

    non_target_mask = speakers != sound_target
    return float(np.mean(np.abs(O_target - O_all[non_target_mask])))


def predict_localization_error(
    veridical_azimuth: float,
    gaze: float,
    alpha: float,
    beta: float | None = None,
    we: float | None = None,
) -> float:
    """Predict perceived azimuth and localization error for Lewald data."""
    azimuths = np.linspace(-90.0, 90.0, 300)
    O_all = opponent_channel(azimuths, gaze, alpha, beta, we)
    O_veridical = opponent_scalar(veridical_azimuth, gaze, alpha, beta, we)

    max_abs_O = float(np.max(np.abs(O_all)))
    perceived_azimuth = 90.0 * O_veridical / max_abs_O if max_abs_O > 0.0 else 0.0

    return perceived_azimuth - float(veridical_azimuth)


# ============================================================================
# Training objective (log-likelihood) and fitting
# ============================================================================


def sensitivity_log_likelihood(
    params: np.ndarray,
    maddox_ild_data: pd.DataFrame,
    maddox_itd_data: pd.DataFrame,
    best_data: pd.DataFrame,
    model_type: str,
) -> float:
    """Compute the Gaussian log-likelihood used for fitting."""
    if model_type == "model1":
        alpha = float(params[0])
        beta = 0.0
        we = 1.0
    elif model_type == "model2":
        alpha = float(params[0])
        beta = float(params[1])
        we = float(params[2])
    else:
        raise ValueError("model_type must be 'model1' or 'model2'")

    ll = 0.0

    # Maddox ITD pairs
    if maddox_itd_data is not None and len(maddox_itd_data) > 0:
        unique_conds = (
            maddox_itd_data.groupby(["gaze_direction", "sound_azimuth_centre"])
            .first()
            .reset_index()
        )
        for i, row1 in unique_conds.iterrows():
            for j in range(i + 1, len(unique_conds)):
                row2 = unique_conds.iloc[j]
                delta_model = predict_discrimination_sensitivity_maddox(
                    float(row1["gaze_direction"]),
                    float(row1["sound_azimuth_centre"]),
                    float(row2["gaze_direction"]),
                    float(row2["sound_azimuth_centre"]),
                    alpha,
                    beta,
                    we,
                    "itd",
                )
                behav_imp = float(
                    row1["proportion_correct"] - row2["proportion_correct"]
                )
                sem = (
                    math.sqrt(float(row1["SEM"] ** 2 + row2["SEM"] ** 2))
                    if ("SEM" in row1 and "SEM" in row2)
                    else 0.1
                )
                residual = delta_model - behav_imp
                ll += -0.5 * np.log(2.0 * np.pi * sem**2) - (residual**2) / (
                    2.0 * sem**2
                )

    # Maddox ILD pairs
    if maddox_ild_data is not None and len(maddox_ild_data) > 0:
        unique_conds = (
            maddox_ild_data.groupby(["gaze_direction", "sound_azimuth_centre"])
            .first()
            .reset_index()
        )
        for i, row1 in unique_conds.iterrows():
            for j in range(i + 1, len(unique_conds)):
                row2 = unique_conds.iloc[j]
                delta_model = predict_discrimination_sensitivity_maddox(
                    float(row1["gaze_direction"]),
                    float(row1["sound_azimuth_centre"]),
                    float(row2["gaze_direction"]),
                    float(row2["sound_azimuth_centre"]),
                    alpha,
                    beta,
                    we,
                    "ild",
                )
                behav_imp = float(
                    row1["proportion_correct"] - row2["proportion_correct"]
                )
                sem = (
                    math.sqrt(float(row1["SEM"] ** 2 + row2["SEM"] ** 2))
                    if ("SEM" in row1 and "SEM" in row2)
                    else 0.1
                )
                residual = delta_model - behav_imp
                ll += -0.5 * np.log(2.0 * np.pi * sem**2) - (residual**2) / (
                    2.0 * sem**2
                )

    # Best 2023 pairs
    if best_data is not None and len(best_data) > 0:
        for sound_dir in best_data["veridical_sound_azimuth"].unique():
            sound_conds = best_data[best_data["veridical_sound_azimuth"] == sound_dir]
            unique_conds = (
                sound_conds.groupby(["gaze_direction", "veridical_sound_azimuth"])
                .first()
                .reset_index()
            )
            for i, row1 in unique_conds.iterrows():
                for j in range(i + 1, len(unique_conds)):
                    row2 = unique_conds.iloc[j]
                    sens1 = predict_discrimination_sensitivity_best(
                        float(row1["gaze_direction"]),
                        float(sound_dir),
                        alpha,
                        beta,
                        we,
                    )
                    sens2 = predict_discrimination_sensitivity_best(
                        float(row2["gaze_direction"]),
                        float(sound_dir),
                        alpha,
                        beta,
                        we,
                    )
                    delta_model = (
                        ((sens1 - sens2) / sens2) * 100.0 if sens2 > 0.0 else 0.0
                    )
                    behav_imp = float(
                        row1["proportion_correct"] - row2["proportion_correct"]
                    )
                    sem = (
                        math.sqrt(float(row1["SEM"] ** 2 + row2["SEM"] ** 2))
                        if ("SEM" in row1 and "SEM" in row2)
                        else 0.1
                    )
                    residual = delta_model - behav_imp
                    ll += -0.5 * np.log(2.0 * np.pi * sem**2) - (residual**2) / (
                        2.0 * sem**2
                    )

    return float(ll)


@dataclass(frozen=True)
class FitResult:
    alpha: float
    beta: float | None
    we: float | None
    log_likelihood: float
    success: bool


def fit_model(
    maddox_ild_data: pd.DataFrame,
    maddox_itd_data: pd.DataFrame,
    best_data: pd.DataFrame,
    model_type: str,
) -> FitResult:
    """Fit model1/model2 by minimizing negative log-likelihood."""
    if model_type == "model1":
        initial_params = np.array([0.5], dtype=float)
        bounds = [(0.0, 1.0)]
    elif model_type == "model2":
        initial_params = np.array([0.5, 0.5, 45.0], dtype=float)
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.1, 90.0)]
    else:
        raise ValueError("model_type must be 'model1' or 'model2'")

    result = minimize(
        lambda p: -sensitivity_log_likelihood(
            p,
            maddox_ild_data,
            maddox_itd_data,
            best_data,
            model_type=model_type,
        ),
        initial_params,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-6},
    )

    ll = sensitivity_log_likelihood(
        result.x,
        maddox_ild_data,
        maddox_itd_data,
        best_data,
        model_type=model_type,
    )

    if model_type == "model1":
        return FitResult(
            alpha=float(result.x[0]),
            beta=0.0,
            we=1.0,
            log_likelihood=ll,
            success=bool(result.success),
        )
    return FitResult(
        alpha=float(result.x[0]),
        beta=float(result.x[1]),
        we=float(result.x[2]),
        log_likelihood=ll,
        success=bool(result.success),
    )


def likelihood_ratio_test(ll1: float, ll2: float, k1: int, k2: int) -> dict[str, float]:
    """Compute LRT for nested models."""
    chi2_stat = 2.0 * (ll2 - ll1)
    df = float(k2 - k1)
    p_value = 1.0 - chi2.cdf(chi2_stat, df) if chi2_stat > 0.0 else 1.0
    return {"chi2": float(chi2_stat), "df": float(df), "p_value": float(p_value)}


# ============================================================================
# Data loading
# ============================================================================


def load_maddox_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = ["gaze_direction", "sound_azimuth_centre", "proportion_correct", "SEM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {path}: {missing}. Found: {list(df.columns)}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    return df


def load_best_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = [
        "gaze_direction",
        "veridical_sound_azimuth",
        "proportion_correct",
        "SEM",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in {path}: {missing}. Found: {list(df.columns)}"
        )

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    return df


def load_lewald_data(path_1998: Path, path_2006: Path) -> pd.DataFrame:
    def _load(path: Path, study: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        rename_map = {}
        if "veridical_azimuth" not in df.columns:
            if "veridical_sound_azimuth" in df.columns:
                rename_map["veridical_sound_azimuth"] = "veridical_azimuth"
        if "perceived_azimuth_deviation" not in df.columns:
            if "perceived_deviation" in df.columns:
                rename_map["perceived_deviation"] = "perceived_azimuth_deviation"
        if rename_map:
            df = df.rename(columns=rename_map)

        required = [
            "gaze_direction",
            "veridical_azimuth",
            "perceived_azimuth_deviation",
            "SEM",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing columns in {path}: {missing}. Found: {list(df.columns)}"
            )

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required)
        df["study"] = study
        return df

    return pd.concat(
        [_load(path_1998, "lewald1998"), _load(path_2006, "lewald2006")],
        ignore_index=True,
    )


def fit_and_validate(data_dir: Path) -> dict:
    """
    Fit both models and validate on Lewald.

    Returns a JSON-serializable dict.
    """
    maddox_ild = load_maddox_data(data_dir / "maddox_ild_2014.csv")
    maddox_itd = load_maddox_data(data_dir / "maddox_itd_2014.csv")
    best = load_best_data(data_dir / "best2023_corrected_eye.csv")
    lewald = load_lewald_data(data_dir / "lewald1998.csv", data_dir / "lewald2006.csv")

    model1 = fit_model(maddox_ild, maddox_itd, best, model_type="model1")
    model2 = fit_model(maddox_ild, maddox_itd, best, model_type="model2")

    k1, k2 = 1, 3
    lrt = likelihood_ratio_test(model1.log_likelihood, model2.log_likelihood, k1, k2)

    # Validation on Lewald et al (1998, 2006).
    observed_errors = lewald["perceived_azimuth_deviation"].values.astype(float)
    errors_model1 = [
        predict_localization_error(
            float(row["veridical_azimuth"]),
            float(row["gaze_direction"]),
            model1.alpha,
            beta=None,
            we=None,
        )
        for _, row in lewald.iterrows()
    ]
    errors_model2 = [
        predict_localization_error(
            float(row["veridical_azimuth"]),
            float(row["gaze_direction"]),
            model2.alpha,
            beta=model2.beta,
            we=model2.we,
        )
        for _, row in lewald.iterrows()
    ]

    errors_model1 = np.array(errors_model1, dtype=float)
    errors_model2 = np.array(errors_model2, dtype=float)

    corr1 = float(np.corrcoef(errors_model1, observed_errors)[0, 1])
    rmse1 = float(np.sqrt(np.mean((errors_model1 - observed_errors) ** 2)))
    corr2 = float(np.corrcoef(errors_model2, observed_errors)[0, 1])
    rmse2 = float(np.sqrt(np.mean((errors_model2 - observed_errors) ** 2)))

    return {
        "model1": {
            "alpha": model1.alpha,
            "log_likelihood": model1.log_likelihood,
            "success": model1.success,
        },
        "model2": {
            "alpha": model2.alpha,
            "beta": model2.beta,
            "we": model2.we,
            "log_likelihood": model2.log_likelihood,
            "success": model2.success,
        },
        "lrt": lrt,
        "validation": {
            "model1": {"correlation": corr1, "rmse": rmse1},
            "model2": {"correlation": corr2, "rmse": rmse2},
        },
    }
