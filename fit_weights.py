"""
Fit blend weights from backtest data instead of guessing them.
================================================================

The model currently blends component stats (BvP / Statcast-similar /
hand-split / recent-form) using hand-picked weights (e.g. 65% BvP + 35%
recent). This script replaces those guesses with weights fit directly from
backtest data -- and, critically, validates them on a HOLDOUT slice of
dates the fit never saw, so we're not just overfitting to one window.

APPROACH:
    The model is a tiered fallback (BvP-trusted, else hand-split, else
    recent-only), not one blended regression -- so weights are fit
    separately per tier, using only the rows that actually used that tier.
    This mirrors the real architecture instead of pretending it's one model.

    For each of (Hits, Total Bases, Runs, RBI) x (BvP tier, hand-split tier):
      1. Split rows by DATE (not row) into train/holdout, so the same
         batter appearing on multiple dates can't leak between the two.
      2. Fit an ordinary least squares linear model on the TRAIN rows:
         target ~ component_1 + component_2 [+ component_3] + intercept
      3. Normalize the slope coefficients into non-negative weights that
         sum to 1 (clipping negatives to 0), matching the blend-weight
         convention the model already uses.
      4. Evaluate BOTH the fitted model and the CURRENT hardcoded formula
         on the HOLDOUT rows (never seen during fitting) and report both,
         so you can see whether fitting actually beats guessing out-of-sample.

    Statcast-similar isn't fit here -- if your backtest ran with
    --no-statcast (as the wide one did), there's no data for it. Re-run
    with Statcast enabled on a slice of dates if you want that tier fit too.

USAGE:
    python fit_weights.py --csv backtest_2026-07-15_2026-08-21.csv
    python fit_weights.py --csv backtest_....csv --holdout-frac 0.25
    python fit_weights.py --csv backtest_....csv --split-date 2026-08-10

OUTPUT:
    Console report: fitted weights per tier/metric, train vs holdout
    correlation for the fitted model vs the current hardcoded formula, and
    a final recommendation of which fits are trustworthy enough to adopt
    (enough holdout rows + fitted model actually beats current on holdout).
"""

import argparse
import csv
import math

try:
    import numpy as np
except ImportError:
    raise SystemExit("This script needs numpy: pip install numpy")


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_f(v, default=None):
    if v in (None, "", "None"):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def split_train_holdout(rows, holdout_frac=0.25, split_date=None):
    dates = sorted(set(r["date"] for r in rows))
    if split_date:
        train_dates = {d for d in dates if d < split_date}
        holdout_dates = {d for d in dates if d >= split_date}
    else:
        cut = max(1, int(len(dates) * (1 - holdout_frac)))
        train_dates = set(dates[:cut])
        holdout_dates = set(dates[cut:])
    train = [r for r in rows if r["date"] in train_dates]
    holdout = [r for r in rows if r["date"] in holdout_dates]
    print(f"Train: {len(train_dates)} dates ({min(train_dates)} to {max(train_dates)}), {len(train)} rows")
    print(f"Holdout: {len(holdout_dates)} dates ({min(holdout_dates)} to {max(holdout_dates)}), {len(holdout)} rows\n")
    return train, holdout


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def mae(xs, ys):
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs) if xs else None


def fit_ols(X, y):
    """X: list of feature-vectors, y: list of targets. Returns (intercept, coefs)."""
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return coefs[0], coefs[1:]


def normalize_weights(coefs):
    """Clip negative coefficients to 0, renormalize the rest to sum to 1."""
    clipped = np.clip(coefs, 0, None)
    total = clipped.sum()
    if total == 0:
        return [1.0 / len(coefs)] * len(coefs)  # degenerate fallback: equal weights
    return list(clipped / total)


def fit_and_evaluate(train, holdout, feature_keys, target_key, tier_filter, label,
                      current_model_key, min_rows=30):
    """
    feature_keys: list of column names to use as predictors (missing -> 0).
    target_key: column name of the actual outcome to predict.
    tier_filter: function(row) -> bool, selects rows belonging to this tier.
    current_model_key: column with the CURRENT hardcoded-weight projection,
                        for a fair head-to-head comparison on holdout.
    """
    train_rows = [r for r in train if tier_filter(r)]
    holdout_rows = [r for r in holdout if tier_filter(r)]

    print(f"--- {label} ---")
    print(f"  Train rows: {len(train_rows)}   Holdout rows: {len(holdout_rows)}")

    if len(train_rows) < min_rows or len(holdout_rows) < min_rows:
        print(f"  [skip] fewer than {min_rows} rows in train or holdout -- not enough to fit/validate reliably.\n")
        return None

    X_train = [[to_f(r[k], 0.0) or 0.0 for k in feature_keys] for r in train_rows]
    y_train = [to_f(r[target_key], 0.0) or 0.0 for r in train_rows]
    intercept, coefs = fit_ols(X_train, y_train)
    weights = normalize_weights(coefs)

    print(f"  Fitted weights: " + ", ".join(f"{k}={w:.2f}" for k, w in zip(feature_keys, weights)))
    print(f"  (raw OLS intercept={intercept:.3f}, coefs=" +
          ", ".join(f"{c:.3f}" for c in coefs) + " -- weights above are the clipped/normalized version)")

    # Evaluate on holdout: fitted model (using normalized weights, no intercept,
    # matching how the current model composes its blend) vs current hardcoded formula.
    X_holdout = [[to_f(r[k], 0.0) or 0.0 for k in feature_keys] for r in holdout_rows]
    y_holdout = [to_f(r[target_key], 0.0) or 0.0 for r in holdout_rows]
    fitted_preds = [sum(w * x for w, x in zip(weights, row)) for row in X_holdout]
    current_preds = [to_f(r[current_model_key], 0.0) or 0.0 for r in holdout_rows]

    r_fitted = pearson(fitted_preds, y_holdout)
    r_current = pearson(current_preds, y_holdout)
    mae_fitted = mae(fitted_preds, y_holdout)
    mae_current = mae(current_preds, y_holdout)

    def fmt(v, spec=".3f"):
        return format(v, spec) if v is not None else "n/a"

    print(f"  HOLDOUT -- fitted:  r={fmt(r_fitted)}  MAE={fmt(mae_fitted)}")
    print(f"  HOLDOUT -- current: r={fmt(r_current)}  MAE={fmt(mae_current)}")
    better = r_fitted is not None and r_current is not None and r_fitted > r_current
    print(f"  {'-> Fitted weights beat current on holdout.' if better else '-> Current weights hold up fine; fit not worth adopting yet.'}\n")

    return {
        "label": label, "feature_keys": feature_keys, "weights": weights,
        "r_fitted": r_fitted, "r_current": r_current, "adopt": better,
        "n_train": len(train_rows), "n_holdout": len(holdout_rows),
    }


def main(csv_path, holdout_frac, split_date, min_bvp_ab):
    rows = load_rows(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}\n")
    train, holdout = split_train_holdout(rows, holdout_frac, split_date)

    def bvp_tier(r):
        return (to_f(r.get("bvp_ab"), 0) or 0) >= min_bvp_ab

    def handsplit_tier(r):
        bvp_ab = to_f(r.get("bvp_ab"), 0) or 0
        hs_ab = to_f(r.get("hand_split_ab"), 0) or 0
        return bvp_ab < min_bvp_ab and hs_ab >= min_bvp_ab

    results = []

    # --- Hits (target: did they get a hit at all, i.e. actual_h > 0) ---
    for r in rows:
        r["_got_hit"] = 1.0 if (to_f(r.get("actual_h"), 0) or 0) > 0 else 0.0

    results.append(fit_and_evaluate(
        train, holdout, ["bvp_avg", "recent_avg"], "_got_hit", bvp_tier,
        "HITS -- BvP-trusted tier", "expected_hits"))
    results.append(fit_and_evaluate(
        train, holdout, ["hand_split_avg", "recent_avg", "bvp_avg"], "_got_hit", handsplit_tier,
        "HITS -- hand-split-fallback tier", "expected_hits"))

    # --- Total Bases (continuous target: actual_total_bases) ---
    results.append(fit_and_evaluate(
        train, holdout, ["bvp_slg", "recent_slg"], "actual_total_bases", bvp_tier,
        "TOTAL BASES -- BvP-trusted tier", "expected_total_bases"))
    results.append(fit_and_evaluate(
        train, holdout, ["hand_split_slg", "recent_slg", "bvp_slg"], "actual_total_bases", handsplit_tier,
        "TOTAL BASES -- hand-split-fallback tier", "expected_total_bases"))

    # --- Runs ---
    results.append(fit_and_evaluate(
        train, holdout, ["bvp_runs_rate", "recent_runs_rate"], "actual_r", bvp_tier,
        "RUNS -- BvP-trusted tier", "expected_runs"))
    results.append(fit_and_evaluate(
        train, holdout, ["hand_split_runs_rate", "recent_runs_rate", "bvp_runs_rate"], "actual_r", handsplit_tier,
        "RUNS -- hand-split-fallback tier", "expected_runs"))

    # --- RBI ---
    results.append(fit_and_evaluate(
        train, holdout, ["bvp_rbi_rate", "recent_rbi_rate"], "actual_rbi", bvp_tier,
        "RBI -- BvP-trusted tier", "expected_rbi"))
    results.append(fit_and_evaluate(
        train, holdout, ["hand_split_rbi_rate", "recent_rbi_rate", "bvp_rbi_rate"], "actual_rbi", handsplit_tier,
        "RBI -- hand-split-fallback tier", "expected_rbi"))

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for res in results:
        if res is None:
            continue
        tag = "ADOPT" if res["adopt"] else "keep current"
        weight_str = ", ".join(f"{k}={w:.2f}" for k, w in zip(res["feature_keys"], res["weights"]))
        rf = format(res["r_fitted"], ".3f") if res["r_fitted"] is not None else "n/a"
        rc = format(res["r_current"], ".3f") if res["r_current"] is not None else "n/a"
        print(f"[{tag:12}] {res['label']:<40} fitted r={rf} vs current r={rc}  ({weight_str})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit blend weights from backtest data with train/holdout validation")
    parser.add_argument("--csv", required=True, help="Path to a backtest_*.csv file")
    parser.add_argument("--holdout-frac", type=float, default=0.25,
                         help="Fraction of dates (most recent) held out for validation (default: 0.25)")
    parser.add_argument("--split-date", type=str, default=None,
                         help="Explicit YYYY-MM-DD split instead of --holdout-frac (train = before this date)")
    parser.add_argument("--min-bvp-ab", type=int, default=8,
                         help="Must match the --min-bvp-ab used when the backtest was run")
    args = parser.parse_args()
    main(args.csv, args.holdout_frac, args.split_date, args.min_bvp_ab)
