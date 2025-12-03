"""
Data Consolidation Pipeline for HVAC Digital Twin
Consolidates winter and summer CSV datasets with data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import glob
import warnings

warnings.filterwarnings('ignore')


class HVACDataConsolidator:
    """Consolidates HVAC experimental data by season."""

    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.summer_pattern = "*_sum_*.csv"
        self.winter_pattern = "*_win_*.csv"

    def load_and_clean_csv(self, filepath: Path) -> pd.DataFrame:
        """Load and clean individual CSV file."""
        try:
            # Read CSV with semicolon delimiter
            df = pd.read_csv(filepath, sep=';', encoding='utf-8')

            # Clean column names
            df.columns = df.columns.str.strip()

            # Handle date column (first column is date/time)
            date_col = df.columns[0]
            df['timestamp'] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
            df = df.drop(columns=[date_col])

            # Convert numeric columns (replace comma with dot for European format)
            for col in df.columns:
                if col != 'timestamp':
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove rows with all NaN values (except timestamp)
            df = df.dropna(how='all', subset=[c for c in df.columns if c != 'timestamp'])

            # Add metadata from filename
            filename = filepath.stem
            df['experiment_id'] = filename.split('_')[0]
            df['season'] = 'summer' if '_sum_' in filename else 'winter'

            # Extract parameters from filename
            params = self._extract_params_from_filename(filename)
            for key, value in params.items():
                df[key] = value

            return df

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

    def _extract_params_from_filename(self, filename: str) -> dict:
        """Extract experimental parameters from filename."""
        params = {}

        # Extract Tinitial
        if 'Tinitial=' in filename:
            try:
                t_initial = filename.split('Tinitial=')[1].split('º')[0]
                params['T_initial'] = float(t_initial.replace(',', '.'))
            except:
                pass

        # Extract Tsetpoint
        if 'Tsetpoint=' in filename:
            try:
                t_setpoint = filename.split('Tsetpoint=')[1].split('º')[0]
                params['T_setpoint'] = float(t_setpoint.replace(',', '.'))
            except:
                pass

        # Extract FF (frequency factor)
        if '_FF=' in filename:
            try:
                ff = filename.split('_FF=')[1].split('.csv')[0].split('_')[0]
                params['frequency_factor'] = float(ff)
            except:
                pass

        return params

    def consolidate_season(self, season: str) -> pd.DataFrame:
        """Consolidate all experiments for a given season."""
        pattern = self.summer_pattern if season == 'summer' else self.winter_pattern
        files = list(self.dataset_path.glob(pattern))

        print(f"\n{'='*60}")
        print(f"Consolidating {season.upper()} data")
        print(f"{'='*60}")
        print(f"Found {len(files)} {season} experiment files")

        all_dfs = []
        for filepath in sorted(files):
            print(f"  Loading: {filepath.name}")
            df = self.load_and_clean_csv(filepath)
            if not df.empty:
                all_dfs.append(df)
                print(f"    ✓ Loaded {len(df)} rows")

        if not all_dfs:
            print(f"  ✗ No valid data found for {season}")
            return pd.DataFrame()

        # Concatenate all dataframes
        consolidated = pd.concat(all_dfs, ignore_index=True)

        # Sort by timestamp
        consolidated = consolidated.sort_values('timestamp').reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"✓ {season.upper()} consolidation complete")
        print(f"  Total rows: {len(consolidated):,}")
        print(f"  Date range: {consolidated['timestamp'].min()} to {consolidated['timestamp'].max()}")
        print(f"  Experiments: {consolidated['experiment_id'].nunique()}")
        print(f"{'='*60}\n")

        return consolidated

    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """Get statistical summary of the dataset."""
        stats = {
            'total_rows': len(df),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'experiments': df['experiment_id'].unique().tolist(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
        }
        return stats

    def save_consolidated_datasets(self, output_dir: str = "consolidated_data"):
        """Consolidate and save both winter and summer datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Process summer data
        summer_df = self.consolidate_season('summer')
        if not summer_df.empty:
            summer_file = output_path / 'hvac_summer_consolidated.csv'
            summer_df.to_csv(summer_file, index=False)
            print(f"✓ Saved summer dataset: {summer_file}")

        # Process winter data
        winter_df = self.consolidate_season('winter')
        if not winter_df.empty:
            winter_file = output_path / 'hvac_winter_consolidated.csv'
            winter_df.to_csv(winter_file, index=False)
            print(f"✓ Saved winter dataset: {winter_file}")

        return summer_df, winter_df


def main():
    """Main execution function."""
    consolidator = HVACDataConsolidator()
    summer_df, winter_df = consolidator.save_consolidated_datasets()

    print("\n" + "="*60)
    print("DATA CONSOLIDATION COMPLETE")
    print("="*60)
    print(f"Summer dataset shape: {summer_df.shape}")
    print(f"Winter dataset shape: {winter_df.shape}")
    print("\nConsolidated datasets saved in 'consolidated_data' directory")
    print("="*60)


if __name__ == "__main__":
    main()