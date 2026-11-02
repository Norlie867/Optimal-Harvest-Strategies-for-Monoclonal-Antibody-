# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

from pathlib import Path

# ----- Paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
FIGURES_DIR = ROOT_DIR / "figures"
TABLES_DIR = ROOT_DIR / "tables"
INITIAL_CONDITIONS_FILE = DATASETS_DIR / "initial_conditions.csv"
BATCH_FILE_PATTERN = "batch_{:04d}.csv"

# ----- Column names (initial_conditions.csv and batch_####.csv) --------
COL_BATCH_ID = "Batch ID"
COL_TIME = "Time [h]"
COL_GLUCOSE = "Glucose [mM]"
COL_GLUTAMINE = "Glutamine [mM]"
COL_VCD = "VCD [cells/mL]"
COL_MAB = "mAb [mg/L]"

# Initial conditions that can be controlled (the 3 factors of the design)
FACTORS = [COL_GLUCOSE, COL_GLUTAMINE, COL_VCD]
# Inputs of the regression model: 3 initial conditions + time
FEATURES = [COL_GLUCOSE, COL_GLUTAMINE, COL_VCD, COL_TIME]

# ----- Experimental design ---------------------------------------------
N_BATCHES = 30  # Number of batch runs we are allowed to download
MAX_BATCH_ID = 1000  # Highest batch file available locally (None = use all 1029)

# ----- Economics -------------------------------------------------------
COST_PER_MM_GLUCOSE = 10.0      # $/L per mM of initial glucose
COST_PER_MM_GLUTAMINE = 20.0    # $/L per mM of initial glutamine
COST_PER_CELL_ML = 0.0005       # $/L per (cell/mL) of initial VCD
VALUE_PER_MG_MAB = 1000.0       # $/L per (mg/L) of mAb
DOWNTIME_D = 1.0                # Bioreactor downtime between batches [d]
HOURS_PER_DAY = 24.0

# ----- Modelling -------------------------------------------------------
CONFIDENCE_LEVEL = 0.95         # Prediction interval level
CANDIDATE_DEGREES = [2, 3, 4, 5]  # Polynomial degrees tested by cross-validation
CV_FOLDS = 5                    # Number of folds (grouped by batch)
RANDOM_SEED = 42
RF_N_ESTIMATORS = 200           # Alternative model: random forest
RF_MIN_SAMPLES_LEAF = 3

# ----- Optimization search grid (kept inside the explored range) -------
GLUCOSE_STEP = 1.0              # mM
GLUTAMINE_STEP = 0.5            # mM
VCD_STEP = 1.0e5                # cells/mL
PREDICTION_CHUNK_SIZE = 50_000  # Rows predicted at once (limits memory use)