# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import matplotlib
matplotlib.use("Agg")  # Export figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.classes import Batch, PolynomialRegressor
from src.constants import *


# ----------------------------------------------------------------------
# Data loading and experimental design
# ----------------------------------------------------------------------

def load_initial_conditions():
    """Read initial_conditions.csv (all 1029 possible batches)."""
    return pd.read_csv(INITIAL_CONDITIONS_FILE)


def select_batch_ids(initial_conditions, n_batches=N_BATCHES, max_batch_id=MAX_BATCH_ID):
    """Choose which batches to download using a space-filling (maximin) design.

    1. Keep one replicate of each unique combination of initial conditions
       (only among batches whose ID is <= max_batch_id, if given).
    2. Start with the 8 corners of the design space (to avoid extrapolation).
    3. Add the point closest to the center (to detect curvature).
    4. Repeatedly add the point farthest from all points already chosen.
    """
    if max_batch_id is not None:
        initial_conditions = initial_conditions[initial_conditions[COL_BATCH_ID] <= max_batch_id]
    design = initial_conditions.drop_duplicates(subset=FACTORS).reset_index(drop=True)
    values = design[FACTORS].to_numpy(dtype=float)
    scaled = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))

    is_corner = np.all((scaled == 0.0) | (scaled == 1.0), axis=1)
    selected = list(np.flatnonzero(is_corner))
    selected.append(int(np.argmin(np.linalg.norm(scaled - 0.5, axis=1))))

    while len(selected) < n_batches:
        distances = np.linalg.norm(scaled[:, None, :] - scaled[None, selected, :], axis=2)
        nearest = distances.min(axis=1)
        nearest[selected] = -1.0  # Never pick the same point twice
        selected.append(int(np.argmax(nearest)))

    return design.loc[selected, COL_BATCH_ID].astype(int).sort_values().tolist()


def load_batches(batch_ids, initial_conditions):
    """Read the selected batch files and return a list of Batch objects."""
    missing = [b for b in batch_ids
               if not (DATASETS_DIR / BATCH_FILE_PATTERN.format(b)).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing batch files in datasets/: "
            + ", ".join(BATCH_FILE_PATTERN.format(b) for b in missing))

    table = initial_conditions.set_index(COL_BATCH_ID)
    batches = []
    for batch_id in batch_ids:
        row = table.loc[batch_id]
        batches.append(Batch.from_csv(
            batch_id, row[COL_GLUCOSE], row[COL_GLUTAMINE], row[COL_VCD],
            DATASETS_DIR / BATCH_FILE_PATTERN.format(batch_id)))
    return batches


def build_training_frame(batches):
    """Stack all batches into one table (one row per measurement)."""
    return pd.concat([b.training_frame() for b in batches], ignore_index=True)


# ----------------------------------------------------------------------
# Economics
# ----------------------------------------------------------------------

def cell_culture_cost(glucose, glutamine, vcd):
    """Cost of the initial medium and cells [$/L]."""
    return (COST_PER_MM_GLUCOSE * glucose
            + COST_PER_MM_GLUTAMINE * glutamine
            + COST_PER_CELL_ML * vcd)


def space_time_metrics(mab, glucose, glutamine, vcd, time_h):
    """Return cost, value, space-time profit and space-time yield."""
    cost = cell_culture_cost(glucose, glutamine, vcd)
    value = VALUE_PER_MG_MAB * mab
    cycle_days = time_h / HOURS_PER_DAY + DOWNTIME_D
    return {
        "cost": cost,
        "value": value,
        "space_time_profit": (value - cost) / cycle_days,
        "space_time_yield": mab / cycle_days,
    }


# ----------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------

def compare_models(training_frame):
    """Cross-validate (grouped by batch) polynomial models and a random forest."""
    x = training_frame[FEATURES].to_numpy(dtype=float)
    y = training_frame[COL_MAB].to_numpy(dtype=float)
    groups = training_frame[COL_BATCH_ID].to_numpy()
    folds = GroupKFold(n_splits=CV_FOLDS)

    candidates = {f"Polynomial (degree {d})": (lambda d=d: PolynomialRegressor(d))
                  for d in CANDIDATE_DEGREES}
    candidates["Random forest"] = lambda: RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED, n_jobs=-1)

    rows, predictions = [], {}
    for name, make_model in candidates.items():
        predicted = np.zeros_like(y)
        for train_idx, test_idx in folds.split(x, y, groups):
            model = make_model().fit(x[train_idx], y[train_idx])
            predicted[test_idx] = model.predict(x[test_idx])
        predictions[name] = predicted
        rows.append({
            "Model": name,
            "CV RMSE [mg/L]": np.sqrt(mean_squared_error(y, predicted)),
            "CV R2": r2_score(y, predicted),
        })

    table = pd.DataFrame(rows)
    best_name = table.loc[table["CV RMSE [mg/L]"].idxmin(), "Model"]
    return table, best_name, predictions


def fit_final_model(training_frame, degree):
    """Fit the selected polynomial model on all the data."""
    return PolynomialRegressor(degree).fit(
        training_frame[FEATURES].to_numpy(dtype=float),
        training_frame[COL_MAB].to_numpy(dtype=float))


def predict_in_chunks(model, x):
    """Predict many rows without building one huge polynomial matrix."""
    parts = [model.predict(x[i:i + PREDICTION_CHUNK_SIZE])
             for i in range(0, len(x), PREDICTION_CHUNK_SIZE)]
    return np.concatenate(parts)


# ----------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------

def build_search_grid(training_frame):
    """All combinations of conditions inside the range that was explored."""
    def axis(column, step):
        return np.arange(training_frame[column].min(),
                         training_frame[column].max() + step / 2, step)

    times = np.sort(training_frame[COL_TIME].unique())
    mesh = np.meshgrid(axis(COL_GLUCOSE, GLUCOSE_STEP),
                       axis(COL_GLUTAMINE, GLUTAMINE_STEP),
                       axis(COL_VCD, VCD_STEP), times, indexing="ij")
    return pd.DataFrame({name: m.ravel() for name, m in zip(FEATURES, mesh)})


def find_optimum(model, training_frame):
    """Grid search for the conditions that maximize space-time profit."""
    grid = build_search_grid(training_frame)
    grid[COL_MAB] = np.clip(predict_in_chunks(model, grid[FEATURES].to_numpy()), 0, None)
    metrics = space_time_metrics(grid[COL_MAB], grid[COL_GLUCOSE], grid[COL_GLUTAMINE],
                                 grid[COL_VCD], grid[COL_TIME])
    grid["Space-time profit"] = metrics["space_time_profit"]
    return grid.loc[grid["Space-time profit"].idxmax()]


def optimum_summary(model, optimum):
    """Economics of the optimum, with 95% prediction intervals."""
    x = optimum[FEATURES].to_numpy(dtype=float).reshape(1, -1)
    mab, lower, upper = (v[0] for v in model.predict_interval(x))
    mab, lower = max(mab, 0.0), max(lower, 0.0)

    def economics(mab_value):
        return space_time_metrics(mab_value, optimum[COL_GLUCOSE], optimum[COL_GLUTAMINE],
                                  optimum[COL_VCD], optimum[COL_TIME])

    mid, low, high = economics(mab), economics(lower), economics(upper)
    ci = int(CONFIDENCE_LEVEL * 100)
    return pd.DataFrame([
        ("Initial glucose [mM]", optimum[COL_GLUCOSE], None, None),
        ("Initial glutamine [mM]", optimum[COL_GLUTAMINE], None, None),
        ("Initial VCD [cells/mL]", optimum[COL_VCD], None, None),
        ("Harvest time [d]", optimum[COL_TIME] / HOURS_PER_DAY, None, None),
        ("Cell culture cost [$/L]", mid["cost"], None, None),
        ("mAb at harvest [mg/L]", mab, lower, upper),
        ("Cell culture value [$/L]", mid["value"], low["value"], high["value"]),
        ("Space-time profit [$/(L*d)]", mid["space_time_profit"],
         low["space_time_profit"], high["space_time_profit"]),
        ("Space-time yield [mg/(L*d)]", mid["space_time_yield"],
         low["space_time_yield"], high["space_time_yield"]),
    ], columns=["Quantity", "Estimate", f"Lower {ci}% PI", f"Upper {ci}% PI"])


def sensitivity_table(model, optimum, training_frame):
    """Vary one initial condition at a time (others at the optimum).
    For each value, harvest at the time that maximizes space-time profit."""
    times = np.sort(training_frame[COL_TIME].unique())
    steps = {COL_GLUCOSE: GLUCOSE_STEP, COL_GLUTAMINE: GLUTAMINE_STEP, COL_VCD: VCD_STEP}
    rows = []
    for factor, step in steps.items():
        values = np.arange(training_frame[factor].min(),
                           training_frame[factor].max() + step / 2, step)
        for value in values:
            conditions = {f: optimum[f] for f in FACTORS}
            conditions[factor] = value
            x = np.column_stack([np.full(len(times), conditions[f]) for f in FACTORS] + [times])
            mab = np.clip(model.predict(x), 0, None)
            metrics = space_time_metrics(mab, *(x[:, i] for i in range(3)), times)
            best = int(np.argmax(metrics["space_time_profit"]))
            rows.append({
                "Factor": factor, "Value": value,
                "Best harvest time [d]": times[best] / HOURS_PER_DAY,
                "mAb [mg/L]": mab[best],
                "Space-time profit [$/(L*d)]": metrics["space_time_profit"][best],
                "Space-time yield [mg/(L*d)]": metrics["space_time_yield"][best],
            })
    return pd.DataFrame(rows)


def profile_over_time(model, optimum, training_frame):
    """Predicted mAb and space-time profit versus time at the optimal conditions."""
    times = np.sort(training_frame[COL_TIME].unique())
    x = np.column_stack([np.full(len(times), optimum[f]) for f in FACTORS] + [times])
    mab, lower, upper = model.predict_interval(x)
    mab, lower, upper = (np.clip(v, 0, None) for v in (mab, lower, upper))
    args = (optimum[COL_GLUCOSE], optimum[COL_GLUTAMINE], optimum[COL_VCD], times)
    return pd.DataFrame({
        "Time [d]": times / HOURS_PER_DAY,
        "mAb [mg/L]": mab, "mAb lower [mg/L]": lower, "mAb upper [mg/L]": upper,
        "Space-time profit [$/(L*d)]": space_time_metrics(mab, *args)["space_time_profit"],
        "Profit lower [$/(L*d)]": space_time_metrics(lower, *args)["space_time_profit"],
        "Profit upper [$/(L*d)]": space_time_metrics(upper, *args)["space_time_profit"],
    })


# ----------------------------------------------------------------------
# Exporting tables and figures
# ----------------------------------------------------------------------

def export_table(table, filename):
    TABLES_DIR.mkdir(exist_ok=True)
    table.to_csv(TABLES_DIR / filename, index=False)


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def plot_design(selected_conditions):
    """2D projections of the 30 selected conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pairs = [(COL_GLUCOSE, COL_GLUTAMINE), (COL_GLUCOSE, COL_VCD), (COL_GLUTAMINE, COL_VCD)]
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(selected_conditions[a], selected_conditions[b], s=40)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
    fig.suptitle("Initial conditions of the selected batch runs")
    save_figure(fig, "experimental_design.png")


def plot_time_series(batches):
    """mAb versus time for all selected batches, colored by initial VCD."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vcds = [b.vcd for b in batches]
    norm = plt.Normalize(min(vcds), max(vcds))
    for b in batches:
        ax.plot(b.time_series[COL_TIME] / HOURS_PER_DAY, b.time_series[COL_MAB],
                color=plt.cm.viridis(norm(b.vcd)), alpha=0.8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label=COL_VCD)
    ax.set_xlabel("Time [d]")
    ax.set_ylabel(COL_MAB)
    ax.set_title("mAb production of the selected batches")
    save_figure(fig, "mab_time_series.png")


def plot_parity(y_true, y_pred, model_name):
    """Cross-validated predictions versus measurements."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(limits, limits, color="black", linestyle="--")
    ax.set_xlabel("Measured mAb [mg/L]")
    ax.set_ylabel("Cross-validated prediction [mg/L]")
    ax.set_title(model_name)
    save_figure(fig, "cross_validation_parity.png")


def plot_optimum_profiles(profile, optimum_time_d):
    """mAb and space-time profit versus harvest time at the optimal conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    t = profile["Time [d]"]
    axes[0].plot(t, profile["mAb [mg/L]"])
    axes[0].fill_between(t, profile["mAb lower [mg/L]"], profile["mAb upper [mg/L]"], alpha=0.25)
    axes[0].set_ylabel("mAb [mg/L]")
    axes[1].plot(t, profile["Space-time profit [$/(L*d)]"])
    axes[1].fill_between(t, profile["Profit lower [$/(L*d)]"], profile["Profit upper [$/(L*d)]"],
                         alpha=0.25)
    axes[1].set_ylabel("Space-time profit [$/(L*d)]")
    for ax in axes:
        ax.axvline(optimum_time_d, color="red", linestyle="--", label="Optimal harvest")
        ax.set_xlabel("Time [d]")
    axes[1].legend()
    fig.suptitle("Optimal initial conditions with 95% prediction intervals")
    save_figure(fig, "optimum_profiles.png")


def plot_sensitivity(sensitivity):
    """Effect of each initial condition on space-time profit and yield."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="col")
    for col, factor in enumerate(FACTORS):
        subset = sensitivity[sensitivity["Factor"] == factor]
        axes[0, col].plot(subset["Value"], subset["Space-time profit [$/(L*d)]"])
        axes[1, col].plot(subset["Value"], subset["Space-time yield [mg/(L*d)]"], color="tab:orange")
        axes[1, col].set_xlabel(factor)
    axes[0, 0].set_ylabel("Space-time profit [$/(L*d)]")
    axes[1, 0].set_ylabel("Space-time yield [mg/(L*d)]")
    fig.suptitle("Influence of each initial condition (others at the optimum)")
    save_figure(fig, "sensitivity.png")