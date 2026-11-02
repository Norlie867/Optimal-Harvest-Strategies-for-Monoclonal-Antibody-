# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.constants import (COL_BATCH_ID, COL_GLUCOSE, COL_GLUTAMINE, COL_VCD,
                           COL_TIME, COL_MAB, CONFIDENCE_LEVEL)


class Batch:
    """One bioreactor run: its initial conditions and its measured time series."""

    def __init__(self, batch_id, glucose, glutamine, vcd, time_series):
        self.batch_id = batch_id
        self.glucose = glucose
        self.glutamine = glutamine
        self.vcd = vcd
        self.time_series = time_series

    @classmethod
    def from_csv(cls, batch_id, glucose, glutamine, vcd, path):
        """Build a Batch by reading its batch_####.csv file."""
        return cls(batch_id, glucose, glutamine, vcd, pd.read_csv(path))

    def training_frame(self):
        """Return one row per time point: initial conditions + time + mAb."""
        frame = self.time_series[[COL_TIME, COL_MAB]].copy()
        frame[COL_BATCH_ID] = self.batch_id
        frame[COL_GLUCOSE] = self.glucose
        frame[COL_GLUTAMINE] = self.glutamine
        frame[COL_VCD] = self.vcd
        return frame

    def display_info(self):
        final_mab = self.time_series[COL_MAB].iloc[-1]
        print(f"Batch {self.batch_id:4d} | "
              f"Glucose: {self.glucose:5.1f} mM | "
              f"Glutamine: {self.glutamine:4.1f} mM | "
              f"VCD: {self.vcd:9.0f} cells/mL | "
              f"Final mAb: {final_mab:5.2f} mg/L")


class PolynomialRegressor:


    def __init__(self, degree=3):
        self.degree = degree
        self.pipeline = make_pipeline(StandardScaler(),
                                      PolynomialFeatures(degree))
        self.coefficients = None
        self.degrees_of_freedom = None
        self.residual_variance = None
        self.xtx_inverse = None

    def _design_matrix(self, x, fit=False):
        """Polynomial terms multiplied by time (in units of 100 h)."""
        x = np.asarray(x, dtype=float)
        phi = self.pipeline.fit_transform(x) if fit else self.pipeline.transform(x)
        return phi * (x[:, -1:] / 100.0)

    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
        phi = self._design_matrix(x, fit=True)
        self.coefficients = np.linalg.lstsq(phi, y, rcond=None)[0]
        residuals = y - phi @ self.coefficients
        self.degrees_of_freedom = len(y) - np.linalg.matrix_rank(phi)
        self.residual_variance = residuals @ residuals / self.degrees_of_freedom
        self.xtx_inverse = np.linalg.pinv(phi.T @ phi)
        return self

    def predict(self, x):
        return self._design_matrix(x) @ self.coefficients

    def predict_interval(self, x, confidence=CONFIDENCE_LEVEL):
        """Return (prediction, lower bound, upper bound) for each row of x."""
        phi = self._design_matrix(x)
        mean = phi @ self.coefficients
        leverage = np.einsum("ij,jk,ik->i", phi, self.xtx_inverse, phi)
        std_error = np.sqrt(self.residual_variance * (1.0 + leverage))
        t_value = stats.t.ppf(0.5 + confidence / 2.0, self.degrees_of_freedom)
        return mean, mean - t_value * std_error, mean + t_value * std_error