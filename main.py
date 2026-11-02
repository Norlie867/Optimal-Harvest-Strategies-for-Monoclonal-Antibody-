# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.constants import *  # This imports all constants from constants.py

# 1. Experimental design: choose which 30 batches to use
initial_conditions = sf.load_initial_conditions()
batch_ids = sf.select_batch_ids(initial_conditions, N_BATCHES)

# 2. Load the selected batches
batches = sf.load_batches(batch_ids, initial_conditions)
for batch in batches:
    batch.display_info()
training_frame = sf.build_training_frame(batches)

selected = initial_conditions[initial_conditions[COL_BATCH_ID].isin(batch_ids)]
sf.export_table(selected, "selected_batches.csv")
sf.plot_design(selected)
sf.plot_time_series(batches)

# 3. Compare models (cross-validation grouped by batch) and keep the best polynomial
cv_table, best_name, cv_predictions = sf.compare_models(training_frame)
sf.export_table(cv_table, "model_comparison.csv")
print("\n", cv_table.to_string(index=False))

polynomials = cv_table[cv_table["Model"].str.startswith("Polynomial")]
best_polynomial = polynomials.loc[polynomials["CV RMSE [mg/L]"].idxmin(), "Model"]
degree = int(best_polynomial.split("degree ")[1].rstrip(")"))
print(f"\nSelected model: {best_polynomial}")
sf.plot_parity(training_frame[COL_MAB].to_numpy(), cv_predictions[best_polynomial], best_polynomial)

# 4. Fit the final model on all the data
model = sf.fit_final_model(training_frame, degree)

# 5. Optimize space-time profit
optimum = sf.find_optimum(model, training_frame)
summary = sf.optimum_summary(model, optimum)
sf.export_table(summary, "optimal_conditions.csv")
print("\n", summary.to_string(index=False))

# 6. Evidence: profiles over time and influence of each initial condition
profile = sf.profile_over_time(model, optimum, training_frame)
sf.export_table(profile, "optimum_profile_over_time.csv")
sf.plot_optimum_profiles(profile, optimum[COL_TIME] / HOURS_PER_DAY)

sensitivity = sf.sensitivity_table(model, optimum, training_frame)
sf.export_table(sensitivity, "sensitivity_initial_conditions.csv")
sf.plot_sensitivity(sensitivity)