import pandas as pd
import os
import numpy as np

def filter_difference_results():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "MR_difference_results.csv")
    output_file = os.path.join(script_dir, "MR_difference_filtered_consistent.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")

    valid_exposures = []
    
    # Analyze by Exposure
    for exposure, exp_data in df.groupby("exposure"):
        # 1. Check IVW Difference Significance
        ivw_row = exp_data[exp_data["method"] == "Inverse-Variance Weighted"]
        
        if ivw_row.empty:
            continue
            
        if ivw_row.iloc[0]["pval_diff"] >= (0.05 / 195):
            continue
        
        # 2. Check Direction Consistency of the Difference (b_diff)
        # Methods to check: IVW, MR Egger, Weighted Median, Weighted mode
        methods_check = ["Inverse-Variance Weighted", "MR Egger", "Weighted Median", "Weighted mode"]
        subset = exp_data[exp_data["method"].isin(methods_check)]
        
        # Ensure that ALL 3 methods are present to check consistency across "all methods"
        if len(subset["method"].unique()) < len(methods_check):
            continue
            
        diffs = subset["b_diff"].values # This is (b_adeno - b_neuro)
        
        all_positive = np.all(diffs > 0)
        all_negative = np.all(diffs < 0)
        
        if all_positive or all_negative:
            valid_exposures.append(exposure)

    # Filter original dataframe
    filtered_df = df[df["exposure"].isin(valid_exposures)].copy()
    
    # Save
    filtered_df.to_csv(output_file, index=False)
    print(f"Saved {len(filtered_df)} rows ({len(valid_exposures)} exposures) to {os.path.basename(output_file)}")
    
    # Show top
    if not filtered_df.empty:
        cols = ["exposure", "method", "b_diff", "pval_diff"]
        print("\nTop Consistent Differences (IVW method):")
        print(filtered_df[filtered_df["method"] == "Inverse-Variance Weighted"][cols].head(10).to_string(index=False))

if __name__ == "__main__":
    filter_difference_results()
