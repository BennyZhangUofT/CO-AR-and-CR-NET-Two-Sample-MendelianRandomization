import pandas as pd
import numpy as np
import genal.MR as MR
from genal.MR_tools import harmonize_MR
from genal.Geno import Geno
import genal.tools as tools
import os

# --- Constants & Config ---
MR_RESULTS_FILE = "MR_results_multi_exposure.csv"

# --- Helper Functions ---
def clean_numeric(series):
    """Ensure numeric type, replacing errors with NaN."""
    return pd.to_numeric(series, errors='coerce')

# --- Main Analysis ---
def run_mr_analysis():
    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Cleaned")
    
    # Files
    exposure_path = os.path.join(data_dir, "MiBioGen_allHits.csv")
    output_file = os.path.join(script_dir, MR_RESULTS_FILE)

    # Ensure PLINK is installed
    try:
        tools.get_plink_path()
    except ValueError:
        print("PLINK2 not found. Installing...")
        tools.install_plink()
    
    # 2. Load Data
    print(f"Loading Exposure Data from: {os.path.basename(exposure_path)}")
    try:
        # Load Exposure Data
        full_exp_df = pd.read_csv(exposure_path)
        
        # Rename Columns (Exposure)
        exp_rename_map = {
            "rsID": "SNP", "beta": "BETA", "SE": "SE",
            "eff.allele": "EA", "ref.allele": "NEA",
            "P.weightedSumZ": "P", "N": "N"
        }
        full_exp_df.rename(columns=exp_rename_map, inplace=True)
        
        if "bac" not in full_exp_df.columns:
            print("Error: 'bac' column not found in exposure data.")
            return

        all_taxa = full_exp_df["bac"].unique()
        # Exclude unknown taxa (e.g., unknownfamily, unknowngenus)
        unique_exposures = [exp for exp in all_taxa if "unknown" not in str(exp).lower() and "unknow" not in str(exp).lower()]
        print(f"Found {len(unique_exposures)} named exposures (excluded {len(all_taxa) - len(unique_exposures)} unknown/unclassified taxa).")

    except FileNotFoundError:
        print(f"Error: Exposure file not found at {exposure_path}")
        return

    # Load All Outcomes (Pre-load from raw FinnGen files)
    finngen_dir = os.path.join(script_dir, "FinnGen")
    outcome_files = {
        "adeno": (
            os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_ADENO_EXALLC")
            if os.path.exists(os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_ADENO_EXALLC"))
            else os.path.join(finngen_dir, "C3_COLORECTAL_ADENO_EXALLC.tsv")
        ),
        "neuro": os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_NEUROENDO_EXALLC")
    }
    
    exp_snps = set(full_exp_df["SNP"].dropna().unique())
    outcomes_data = {}
    print("Loading Raw Outcome Data from FinnGen TSV summary stats...")
    use_cols = ["#chrom", "pos", "ref", "alt", "rsids", "pval", "beta", "sebeta"]
    out_rename_map = {
        "rsids": "SNP", "beta": "BETA", "sebeta": "SE",
        "alt": "EA", "ref": "NEA", "pval": "P",
        "#chrom": "CHR", "chrom": "CHR", "pos": "POS"
    }

    for out_name, out_path in outcome_files.items():
        if not os.path.exists(out_path):
            print(f"  Warning: Outcome file not found at {out_path}")
            continue
            
        print(f"  Reading raw file for '{out_name}': {os.path.basename(out_path)}...")
        try:
            chunks = []
            for chunk in pd.read_csv(out_path, sep="\t", chunksize=500000, usecols=use_cols):
                filt = chunk[chunk["rsids"].isin(exp_snps)].copy()
                if len(filt) > 0:
                    chunks.append(filt)
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                df.rename(columns=out_rename_map, inplace=True)
                df.dropna(subset=["SNP"], inplace=True)
                outcomes_data[out_name] = df
                print(f"  Loaded {out_name}: {len(df)} variants matching exposure dataset")
            else:
                print(f"  Warning: No matching variants found for {out_name}")
                
        except Exception as e:
            print(f"  Error loading {out_path}: {e}")

    if not outcomes_data:
        print("No outcome data loaded. Exiting.")
        return

    # 3. Initialize Results File
    # Create empty CSV with headers if it doesn't exist, or overwrite
    result_cols = ["exposure", "outcome", "method", "nSNP", "mean_F", "b", "se", "or", "or_lci95", "or_uci95", "pval", "Q", "Q_pval"]
    pd.DataFrame(columns=result_cols).to_csv(output_file, index=False)
    print(f"\nInitialized results file: {output_file}")

    # 4. Main Analysis Loop (Outer Loop = Exposure)
    print("\nStarting Analysis Loop (Exposure -> Clumping -> Outcomes)...")
    
    for i, exposure_name in enumerate(unique_exposures):
        print(f"Processing [{i+1}/{len(unique_exposures)}]: {exposure_name}")
        
        # Filter Exposure DF
        exp_df = full_exp_df[full_exp_df["bac"] == exposure_name].copy()
        
        # --- PRE-FILTERING (P < 1e-5) ---
        exp_df = exp_df[exp_df["P"] < 1e-5].copy()
        if len(exp_df) == 0:
            continue

        # --- LD CLUMPING (Run ONCE per exposure) ---
        clump_input = exp_df.rename(columns={"chr": "CHR", "bp": "POS"})
        
        geno = Geno(clump_input[["SNP", "P", "CHR", "POS"]], keep_columns=True)
        
        try:
            # Stricter LD: r2=0.001
            clumped_geno = geno.clump(kb=10000, r2=0.001, p1=1e-5, p2=1e-5)
            
            if clumped_geno is None:
                continue
                
            clumped_snps = clumped_geno.data["SNP"].tolist()
            exp_df = exp_df[exp_df["SNP"].isin(clumped_snps)].copy()
            
        except Exception as e:
            print(f"  Warning: Clumping failed ({e}). Skipping.")
            continue

        # --- F-STAT FILTER (F > 10) ---
        if len(exp_df) > 0:
            exp_df["F_stat"] = (exp_df["BETA"] ** 2) / (exp_df["SE"] ** 2)
            exp_df = exp_df[exp_df["F_stat"] > 10].copy()
        
        if len(exp_df) < 3:
            # Too few SNPs for MR
            continue
            
        mean_f_stat = exp_df["F_stat"].mean()
        
        # --- OUTCOME LOOP ---
        batch_results = []
        
        for outcome_name, out_df in outcomes_data.items():
            # --- ALLELE HARMONIZATION & PALINDROMIC VARIANT REMOVAL ---
            # action=3 excludes all palindromic SNPs (A/T and C/G) and aligns/flips effect alleles
            try:
                mr_data = harmonize_MR(exp_df, out_df, action=3)
            except Exception as e:
                print(f"  Warning: Harmonization failed for {exposure_name} vs {outcome_name}: {e}")
                continue
            
            if len(mr_data) < 3:
                continue

            # Ensure numeric & positive SE values
            cols_to_numeric = ["BETA_e", "SE_e", "BETA_o", "SE_o"]
            mr_data[cols_to_numeric] = mr_data[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
            mr_data.dropna(subset=cols_to_numeric, inplace=True)
            mr_data = mr_data[(mr_data["SE_e"] > 0) & (mr_data["SE_o"] > 0)].copy()
            
            if len(mr_data) < 3:
                continue
                
            # Run MR
            BETA_e, SE_e = mr_data["BETA_e"], mr_data["SE_e"]
            BETA_o, SE_o = mr_data["BETA_o"], mr_data["SE_o"]
            
            # Helper to format result
            def add_res(res_list):
                if res_list:
                    for r in res_list:
                        r['exposure'] = exposure_name
                        r['outcome'] = outcome_name
                        r['mean_F'] = mean_f_stat
                        batch_results.append(r)

            try:
                add_res(MR.mr_ivw(BETA_e, SE_e, BETA_o, SE_o))
            except Exception:
                add_res([{"method": "Inverse variance weighted", "b": np.nan, "se": np.nan, "pval": np.nan, "Q": np.nan, "Q_pval": np.nan}])

            try:
                add_res(MR.mr_egger_regression(BETA_e, SE_e, BETA_o, SE_o))
            except Exception:
                add_res([{"method": "MR Egger", "b": np.nan, "se": np.nan, "pval": np.nan, "Q": np.nan, "Q_pval": np.nan}])

            try:
                # Add Weighted Mode with required params: phi=1, nboot=100, cpus=1
                add_res(MR.mr_weighted_mode(BETA_e, SE_e, BETA_o, SE_o, 1, 100, 1, show_progress=False))
            except Exception:
                # Handle cases where weighted mode might fail (e.g., too few SNPs after filtering)
                add_res([{"method": "Weighted mode", "b": np.nan, "se": np.nan, "pval": np.nan, "Q": np.nan, "Q_pval": np.nan}])

            try:
                add_res(MR.mr_weighted_median(BETA_e, SE_e, BETA_o, SE_o, nboot=100))
            except Exception:
                add_res([{"method": "Weighted median", "b": np.nan, "se": np.nan, "pval": np.nan, "Q": np.nan, "Q_pval": np.nan}])
        
        # --- INCREMENTAL SAVE ---
        if batch_results:
            results_df = pd.DataFrame(batch_results)
            # Calculate Odds Ratio (OR) and 95% Confidence Intervals
            results_df["or"] = np.exp(results_df["b"])
            results_df["or_lci95"] = np.exp(results_df["b"] - 1.96 * results_df["se"])
            results_df["or_uci95"] = np.exp(results_df["b"] + 1.96 * results_df["se"])
            
            # Filter to relevant columns for size/speed
            cols_to_save = [c for c in result_cols if c in results_df.columns]
            
            # Append to CSV (header=False)
            results_df[cols_to_save].to_csv(output_file, mode='a', header=False, index=False)
            # print(f"  Saved {len(batch_results)} results.")

    print(f"\nAnalysis Complete. Results saved to {output_file}")

    # -------------------------------------------------
    # 5️⃣ Reverse MR analysis (outcome -> exposure)
    # -------------------------------------------------
    print("\nStarting Reverse Causal Testing (Outcome → Exposure)...")
    reverse_output_file = os.path.join(script_dir, "reverse_MR_results_multi_exposure.csv")
    # Initialise reverse results CSV with same columns
    pd.DataFrame(columns=result_cols).to_csv(reverse_output_file, index=False)

    # Loop over each outcome as the 'exposure'
    for i_rev, outcome_name in enumerate(outcomes_data.keys()):
        print(f"Reverse [{i_rev+1}/{len(outcomes_data)}]: {outcome_name}")
        # Load outcome data (will act as exposure in reverse)
        out_df = outcomes_data[outcome_name].copy()
        # Pre‑filter (P < 1e-4)
        p_thresh = 1e-4
        out_df_filt = out_df[out_df["P"] < p_thresh].copy()
        if len(out_df_filt) == 0:
            continue
        # LD clumping for the outcome (now exposure)
        geno_rev = Geno(out_df_filt[["SNP", "P", "CHR", "POS"]], keep_columns=True)
        try:
            clumped_rev = geno_rev.clump(kb=10000, r2=0.001, p1=p_thresh, p2=p_thresh)
            if clumped_rev is None:
                continue
            clumped_snps_rev = clumped_rev.data["SNP"].tolist()
            out_df_clumped = out_df_filt[out_df_filt["SNP"].isin(clumped_snps_rev)].copy()
        except Exception as e:
            print(f"  Warning: Reverse clumping failed ({e}). Skipping.")
            continue
        # F‑stat filter
        out_df_clumped["F_stat"] = (out_df_clumped["BETA"] ** 2) / (out_df_clumped["SE"] ** 2)
        out_df_clumped = out_df_clumped[out_df_clumped["F_stat"] > 10].copy()
        if len(out_df_clumped) < 1:
            continue
        mean_f_rev = out_df_clumped["F_stat"].mean()
        # Now iterate over original exposures as outcomes
        batch_rev = []
        for exposure_name in unique_exposures:
            # Filter original exposure data
            exp_df_rev = full_exp_df[full_exp_df["bac"] == exposure_name].copy()
            # Harmonise (exposure‑outcome swapped)
            try:
                mr_data_rev = harmonize_MR(out_df_clumped, exp_df_rev, action=3)
            except Exception as e:
                print(f"  Warning: Reverse harmonization failed for {outcome_name} vs {exposure_name}: {e}")
                continue
            if len(mr_data_rev) < 1:
                continue
            # Ensure numeric types
            cols_num = ["BETA_e", "SE_e", "BETA_o", "SE_o"]
            mr_data_rev[cols_num] = mr_data_rev[cols_num].apply(pd.to_numeric, errors='coerce')
            mr_data_rev.dropna(subset=cols_num, inplace=True)
            mr_data_rev = mr_data_rev[(mr_data_rev["SE_e"] > 0) & (mr_data_rev["SE_o"] > 0)].copy()
            if len(mr_data_rev) < 1:
                continue
            # Helper to add results with "reverse" prefix on exposure/outcome names
            def add_rev(res_list):
                if res_list:
                    for r in res_list:
                        r["exposure"] = outcome_name   # now treated as exposure
                        r["outcome"] = exposure_name   # original exposure becomes outcome
                        r["mean_F"] = mean_f_rev
                        batch_rev.append(r)
            # Run MR methods (same as forward loop)
            try:
                add_rev(MR.mr_ivw(mr_data_rev["BETA_e"], mr_data_rev["SE_e"], mr_data_rev["BETA_o"], mr_data_rev["SE_o"]))
            except Exception:
                pass
            try:
                add_rev(MR.mr_egger_regression(mr_data_rev["BETA_e"], mr_data_rev["SE_e"], mr_data_rev["BETA_o"], mr_data_rev["SE_o"]))
            except Exception:
                pass
            try:
                add_rev(MR.mr_weighted_mode(mr_data_rev["BETA_e"], mr_data_rev["SE_e"], mr_data_rev["BETA_o"], mr_data_rev["SE_o"], 1, 100, 1, show_progress=False))
            except Exception:
                pass
            try:
                add_rev(MR.mr_weighted_median(mr_data_rev["BETA_e"], mr_data_rev["SE_e"], mr_data_rev["BETA_o"], mr_data_rev["SE_o"], nboot=100))
            except Exception:
                pass
        # Append reverse results to CSV
        if batch_rev:
            rev_df = pd.DataFrame(batch_rev)
            # Calculate Odds Ratio (OR) and 95% Confidence Intervals
            rev_df["or"] = np.exp(rev_df["b"])
            rev_df["or_lci95"] = np.exp(rev_df["b"] - 1.96 * rev_df["se"])
            rev_df["or_uci95"] = np.exp(rev_df["b"] + 1.96 * rev_df["se"])
            
            cols_to_save = [c for c in result_cols if c in rev_df.columns]
            rev_df[cols_to_save].to_csv(reverse_output_file, mode='a', header=False, index=False)
            print(f"  Saved {len(batch_rev)} reverse results for {outcome_name}")

    print("\nReverse analysis complete. Results saved to reverse_MR_results_multi_exposure.csv")


if __name__ == "__main__":
    run_mr_analysis()
