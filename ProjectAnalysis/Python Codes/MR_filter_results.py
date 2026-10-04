import pandas as pd
import os
import numpy as np

def filter_results():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "MR_results_multi_exposure.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")

    # Function to apply filters
    def filter_outcome_group(group_df, outcome_name):
        valid_exposures = []
        
        # Group by exposure to check consistency
        for exposure, exp_data in group_df.groupby("exposure"):
            # 1. Check IVW P-value < 0.05
            ivw_row = exp_data[exp_data["method"] == "Inverse-Variance Weighted"]
            
            if ivw_row.empty:
                continue
                
            if ivw_row.iloc[0]["pval"] >= (0.05 / 195):
                continue
            
            # 2. Check Direction Consistency
            # Get betas for MR methods (exclude Egger Intercept as it's a bias test, not an effect estimate)
            # Actually, usually we want consistency across effect estimates: IVW, MR Egger, Weighted Median.
            # Egger Intercept beta is the intercept, not the effect.
            
            # Required methods to check consistency
            required_methods = ["Inverse-Variance Weighted", "MR Egger", "Weighted Median", "Weighted mode"]
            effect_rows = exp_data[exp_data["method"].isin(required_methods)]
            
            if effect_rows.empty:
                continue
                
            betas = effect_rows["b"].values
            
            # Check signs: all > 0 or all < 0
            all_positive = np.all(betas > 0)
            all_negative = np.all(betas < 0)
            
            if all_positive or all_negative:
                valid_exposures.append(exposure)

        # Filter the original dataframe to keep only valid exposures
        filtered_df = group_df[group_df["exposure"].isin(valid_exposures)].copy()
        
        # Save
        output_filename = f"MR_results_{outcome_name}_filtered.csv"
        output_path = os.path.join(script_dir, output_filename)
        filtered_df.to_csv(output_path, index=False)
        print(f"[{outcome_name.upper()}] Saved {len(filtered_df)} rows ({len(valid_exposures)} exposures) to {output_filename}")
        
        return filtered_df

    # Split by outcome
    outcomes = df["outcome"].unique()
    
    for outcome in outcomes:
        print(f"\nProcessing Outcome: {outcome}")
        outcome_df = df[df["outcome"] == outcome].copy()
        filter_outcome_group(outcome_df, outcome)

if __name__ == "__main__":
    filter_results()
