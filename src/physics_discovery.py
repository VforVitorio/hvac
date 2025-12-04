"""
Physics Discovery Module using PySINDy
Discovers governing equations of HVAC system dynamics from data.
"""

import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class HVACPhysicsDiscovery:
    """Discovers physical equations governing HVAC system dynamics using SINDy."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize physics discovery module.

        Args:
            feature_names: List of feature names for equation discovery
        """
        self.feature_names = feature_names
        self.model = None
        self.scaler = StandardScaler()
        self.discovered_equations = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        state_vars: List[str],
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for SINDy equation discovery.

        Args:
            df: Input dataframe with HVAC measurements
            state_vars: List of state variable column names
            dt: Time step (default 1 second based on data)

        Returns:
            X: State variables array
            X_dot: Time derivatives of state variables
        """
        # Extract state variables
        X = df[state_vars].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize data
        X_scaled = self.scaler.fit_transform(X)

        # Compute derivatives using finite differences
        X_dot = np.gradient(X_scaled, dt, axis=0)

        return X_scaled, X_dot

    def discover_equations(
        self,
        X: np.ndarray,
        X_dot: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.05,
        alpha: float = 0.01
    ) -> ps.SINDy:
        """
        Discover governing equations using SINDy.

        Args:
            X: State variables
            X_dot: Time derivatives
            feature_names: Names of features
            threshold: Sparsity threshold for STLSQ
            alpha: Regularization parameter

        Returns:
            Fitted SINDy model
        """
        # Define optimizer with sparsity-promoting regularization
        optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)

        # Define library of candidate functions
        # Includes polynomials up to degree 3 and trigonometric functions
        library = ps.PolynomialLibrary(degree=3) + ps.FourierLibrary(n_frequencies=2)

        # Create SINDy model
        model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library
        )

        # Fit the model
        model.fit(X, t=1.0, x_dot=X_dot, feature_names=feature_names)

        self.model = model
        self.feature_names = feature_names

        return model

    def extract_equations(self) -> Dict[str, str]:
        """Extract discovered equations as strings."""
        if self.model is None:
            raise ValueError("Model not fitted. Call discover_equations first.")

        equations = {}
        for i, feature in enumerate(self.feature_names):
            eq = self.model.equations()[i]
            equations[feature] = eq

        self.discovered_equations = equations
        return equations

    def get_coefficients(self) -> np.ndarray:
        """Get coefficient matrix of discovered equations."""
        if self.model is None:
            raise ValueError("Model not fitted. Call discover_equations first.")
        return self.model.coefficients()

    def simulate(self, X0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Simulate system using discovered equations.

        Args:
            X0: Initial conditions
            t: Time array

        Returns:
            Simulated trajectories
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call discover_equations first.")

        return self.model.simulate(X0, t)

    def save_equations(self, filepath: str):
        """Save discovered equations to file."""
        with open(filepath, 'w') as f:
            f.write("DISCOVERED HVAC SYSTEM EQUATIONS\n")
            f.write("="*60 + "\n\n")

            for var, eq in self.discovered_equations.items():
                f.write(f"d({var})/dt = {eq}\n\n")

            f.write("\nCOEFFICIENT MATRIX\n")
            f.write("="*60 + "\n")
            f.write(str(self.get_coefficients()))

    def plot_coefficients(self, save_path: Optional[str] = None):
        """Plot coefficient matrix as heatmap."""
        import seaborn as sns

        coefs = self.get_coefficients()

        plt.figure(figsize=(12, 8))
        sns.heatmap(coefs, cmap='coolwarm', center=0,
                   yticklabels=self.feature_names,
                   xticklabels=self.model.get_feature_names())
        plt.title("SINDy Coefficient Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class HVACPhysicsExtractor:
    """Extract and validate physics-based relationships for HVAC system."""

    @staticmethod
    def define_hvac_state_variables() -> Dict[str, List[str]]:
        """
        Define key state variables for HVAC physics discovery.

        Returns:
            Dictionary of variable categories
        """
        return {
            'temperatures': ['UCAIT', 'UCAOT', 'UCWIT', 'UCWOT', 'AMBT'],
            'humidity': ['UCAIH', 'UCAOH', 'AMBH'],
            'flow': ['UCWF', 'MVWF1', 'MVWF2'],
            'pressure': ['UCWDP', 'UCFDP', 'MVDP', 'CPPR'],
            'power': ['CPMEP', 'CPHP', 'CPHE'],
            'control': ['T_setpoint', 'MVCV']
        }

    @staticmethod
    def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived physical features.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with additional derived features
        """
        df = df.copy()

        # Temperature differences (driving forces)
        if 'UCWIT' in df.columns and 'UCWOT' in df.columns:
            df['delta_T_water'] = df['UCWOT'] - df['UCWIT']

        if 'UCAIT' in df.columns and 'UCAOT' in df.columns:
            df['delta_T_air'] = df['UCAOT'] - df['UCAIT']

        # Heat transfer (Q = m * cp * delta_T, simplified)
        if 'UCWF' in df.columns and 'delta_T_water' in df.columns:
            df['Q_water'] = df['UCWF'] * df['delta_T_water']

        # Efficiency metrics
        if 'CPHE' in df.columns and 'CPMEP' in df.columns:
            df['COP'] = np.where(df['CPMEP'] > 0,
                                df['CPHE'] / df['CPMEP'],
                                0)

        # Error signal (for control analysis)
        if 'T_setpoint' in df.columns and 'UCAOT' in df.columns:
            df['T_error'] = df['T_setpoint'] - df['UCAOT']

        return df


def run_physics_discovery_pipeline(
    df: pd.DataFrame,
    season: str,
    output_dir: str = "physics_models"
) -> Tuple[HVACPhysicsDiscovery, Dict[str, str]]:
    """
    Complete pipeline for physics discovery from HVAC data.

    Args:
        df: Consolidated HVAC dataframe
        season: 'summer' or 'winter'
        output_dir: Directory to save results

    Returns:
        Fitted physics discovery model and equations
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PHYSICS DISCOVERY PIPELINE - {season.upper()}")
    print(f"{'='*60}\n")

    # Get state variables
    extractor = HVACPhysicsExtractor()
    state_vars_dict = extractor.define_hvac_state_variables()

    # Add derived features
    print("Computing derived physical features...")
    df = extractor.compute_derived_features(df)

    # Select core state variables for equation discovery
    core_vars = (
        state_vars_dict['temperatures'][:3] +  # Key temperatures
        state_vars_dict['humidity'][:2] +       # Key humidity
        state_vars_dict['flow'][:1] +           # Water flow
        ['T_setpoint']                          # Control setpoint
    )

    # Filter to available columns
    available_vars = [v for v in core_vars if v in df.columns]
    print(f"Using {len(available_vars)} state variables: {available_vars}")

    # Initialize physics discovery
    discovery = HVACPhysicsDiscovery(feature_names=available_vars)

    # Prepare data
    print("Preparing data for SINDy...")
    X, X_dot = discovery.prepare_data(df, available_vars, dt=1.0)

    # Discover equations
    print("Discovering governing equations...")
    model = discovery.discover_equations(
        X, X_dot,
        feature_names=available_vars,
        threshold=0.05,
        alpha=0.01
    )

    # Extract equations
    equations = discovery.extract_equations()

    print(f"\n{'='*60}")
    print("DISCOVERED EQUATIONS")
    print(f"{'='*60}\n")
    for var, eq in equations.items():
        print(f"d({var})/dt = {eq}\n")

    # Save results
    eq_file = output_path / f"{season}_equations.txt"
    discovery.save_equations(str(eq_file))
    print(f"\n✓ Equations saved to: {eq_file}")

    # Plot and save coefficient matrix
    coef_plot = output_path / f"{season}_coefficients.png"
    discovery.plot_coefficients(save_path=str(coef_plot))
    print(f"✓ Coefficient plot saved to: {coef_plot}")

    return discovery, equations


if __name__ == "__main__":
    # Example usage
    print("Physics Discovery Module for HVAC Digital Twin")
    print("Run this after consolidating data using data_consolidation.py")