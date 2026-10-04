import pandas as pd
import numpy as np
from scipy.stats import norm
import os

def run_difference_test():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "MR_results_multi_exposure.csv")
    output_file = os.path.join(script_dir, "MR_difference_results.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Results file not found at {input_file}")
        return

    # Load Data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {os.path.basename(input_file)}")
    
    # Filter for IVW (Primary Method) - or we can do all. Let's do all matching methods.
    # We need to pivot independent of method first
    
    # Get unique methods
    methods = df["method"].unique()
    
    all_comparisons = []
    
    for method in methods:
        # Filter for this method
        method_df = df[df["method"] == method].copy()
        
        # Split by outcome
        adeno_df = method_df[method_df["outcome"] == "adeno"].set_index("exposure")
        neuro_df = method_df[method_df["outcome"] == "neuro"].set_index("exposure")
        
        # Find common exposures
        common_exposures = adeno_df.index.intersection(neuro_df.index)
        
        if len(common_exposures) == 0:
            print(f"No common exposures found for method: {method}")
            continue
            
        print(f"Comparing {len(common_exposures)} exposures for method: {method}")
        
        for exposure in common_exposures:
            row_adeno = adeno_df.loc[exposure]
            row_neuro = neuro_df.loc[exposure]
            
            b1, se1 = row_adeno["b"], row_adeno["se"]
            b2, se2 = row_neuro["b"], row_neuro["se"]
            
            # Difference Test
            # Z = (b1 - b2) / sqrt(se1^2 + se2^2)
            se_diff = np.sqrt(se1**2 + se2**2)
            z_diff = (b1 - b2) / se_diff
            p_diff = 2 * (1 - norm.cdf(abs(z_diff)))
            
            all_comparisons.append({
                "exposure": exposure,
                "method": method,
                "b_adeno": b1,
                "se_adeno": se1,
                "pval_adeno": row_adeno["pval"],
                "b_neuro": b2,
                "se_neuro": se2,
                "pval_neuro": row_neuro["pval"],
                "b_diff": b1 - b2,
                "z_diff": z_diff,
                "pval_diff": p_diff
            })

    # Save Results
    if all_comparisons:
        comp_df = pd.DataFrame(all_comparisons)
        
        # Sort by pval_diff
        comp_df.sort_values("pval_diff", inplace=True)
        
        comp_df.to_csv(output_file, index=False)
        print(f"\nDifference Analysis Complete. Saved to {output_file}")
        
        # Display Top Differences
        cols = ["exposure", "method", "b_adeno", "b_neuro", "pval_diff"]
        print("\nTop 10 Most Significant Differences:")
        print(comp_df[cols].head(10).to_string(index=False))
        
        # Check for significant ones
        sig_diff = comp_df[comp_df["pval_diff"] < 0.05]
        print(f"\nTotal Significant Differences (P < 0.05): {len(sig_diff)}")
    else:
        print("No comparisons could be made.")

if __name__ == "__main__":
    run_difference_test()
