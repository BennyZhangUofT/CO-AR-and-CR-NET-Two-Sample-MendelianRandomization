import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import genal.MR as MR
from genal.MR_tools import harmonize_MR
from genal.Geno import Geno
import os

def run_reverse_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Cleaned")
    exposure_path = os.path.join(data_dir, "MiBioGen_allHits.csv")

    full_exp_df = pd.read_csv(exposure_path)
    exp_rename_map = {
        "rsID": "SNP", "beta": "BETA", "SE": "SE",
        "eff.allele": "EA", "ref.allele": "NEA",
        "P.weightedSumZ": "P", "N": "N"
    }
    full_exp_df.rename(columns=exp_rename_map, inplace=True)
    all_taxa = full_exp_df["bac"].unique()
    unique_exposures = [exp for exp in all_taxa if "unknown" not in str(exp).lower() and "unknow" not in str(exp).lower()]

    finngen_dir = os.path.join(script_dir, "FinnGen")
    cancers_config = {
        "adeno": (
            os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_ADENO_EXALLC") 
            if os.path.exists(os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_ADENO_EXALLC"))
            else os.path.join(finngen_dir, "C3_COLORECTAL_ADENO_EXALLC.tsv"),
            1e-5
        ),
        "neuro": (
            os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_NEUROENDO_EXALLC"),
            1e-4
        )
    }

    result_cols = ["exposure", "outcome", "method", "nSNP", "mean_F", "b", "se", "or", "or_lci95", "or_uci95", "pval", "Q", "Q_pval"]
    reverse_output_file = os.path.join(script_dir, "reverse_MR_results_multi_exposure.csv")
    
    all_reverse_results = []

    print("Running Reverse Causal Testing (Outcome → Exposure)...")
    for i_rev, (outcome_name, (filepath, p_thresh)) in enumerate(cancers_config.items()):
        print(f"Processing Reverse [{i_rev+1}/{len(cancers_config)}]: {outcome_name} from raw file ({os.path.basename(filepath)})...")
        if not os.path.exists(filepath):
            print(f"  Warning: File not found {filepath}. Skipping.")
            continue
            
        chunks = []
        use_cols = ["#chrom", "pos", "ref", "alt", "rsids", "pval", "beta", "sebeta"]
        try:
            for chunk in pd.read_csv(filepath, sep="\t", chunksize=500000, usecols=use_cols):
                filt = chunk[chunk["pval"] < p_thresh].copy()
                if len(filt) > 0:
                    chunks.append(filt)
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            continue

        if not chunks:
            print(f"  No variants found below P < {p_thresh}.")
            continue

        out_df = pd.concat(chunks, ignore_index=True)
        out_rename_map = {
            "rsids": "SNP", "beta": "BETA", "sebeta": "SE",
            "alt": "EA", "ref": "NEA", "pval": "P",
            "#chrom": "CHR", "chrom": "CHR", "pos": "POS"
        }
        out_df.rename(columns=out_rename_map, inplace=True)
        out_df.dropna(subset=["SNP"], inplace=True)
        print(f"  Found {len(out_df)} total candidate variants below P < {p_thresh}.")

        geno_rev = Geno(out_df[["SNP", "P", "CHR", "POS"]], keep_columns=True)
        try:
            clumped_rev = geno_rev.clump(kb=10000, r2=0.001, p1=p_thresh, p2=p_thresh)
            if clumped_rev is None:
                continue
            clumped_snps_rev = clumped_rev.data["SNP"].tolist()
            out_df_clumped = out_df[out_df["SNP"].isin(clumped_snps_rev)].copy()
        except Exception as e:
            print(f"  Warning: Reverse clumping failed ({e}). Skipping.")
            continue
            
        out_df_clumped["F_stat"] = (out_df_clumped["BETA"] ** 2) / (out_df_clumped["SE"] ** 2)
        out_df_clumped = out_df_clumped[out_df_clumped["F_stat"] > 10].copy()
        print(f"  Clumped independent instruments (F > 10): {len(out_df_clumped)}")
        if len(out_df_clumped) < 1:
            continue
            
        mean_f_rev = out_df_clumped["F_stat"].mean()
        
        for exposure_name in unique_exposures:
            exp_df_rev = full_exp_df[full_exp_df["bac"] == exposure_name].copy()
            try:
                mr_data_rev = harmonize_MR(out_df_clumped, exp_df_rev, action=3)
            except Exception as e:
                continue
                
            if len(mr_data_rev) < 1:
                continue
                
            cols_num = ["BETA_e", "SE_e", "BETA_o", "SE_o"]
            mr_data_rev[cols_num] = mr_data_rev[cols_num].apply(pd.to_numeric, errors='coerce')
            mr_data_rev.dropna(subset=cols_num, inplace=True)
            mr_data_rev = mr_data_rev[(mr_data_rev["SE_e"] > 0) & (mr_data_rev["SE_o"] > 0)].copy()
            
            n = len(mr_data_rev)
            if n == 1:
                b_e, se_e = float(mr_data_rev["BETA_e"].iloc[0]), float(mr_data_rev["SE_e"].iloc[0])
                b_o, se_o = float(mr_data_rev["BETA_o"].iloc[0]), float(mr_data_rev["SE_o"].iloc[0])
                if b_e != 0 and se_e > 0 and se_o > 0:
                    b_wald = b_o / b_e
                    se_wald = se_o / abs(b_e)
                    z = b_wald / se_wald
                    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                    
                    all_reverse_results.append({
                        "exposure": outcome_name,
                        "outcome": exposure_name,
                        "method": "Wald ratio",
                        "nSNP": 1,
                        "mean_F": mean_f_rev,
                        "b": b_wald,
                        "se": se_wald,
                        "or": np.exp(b_wald),
                        "or_lci95": np.exp(b_wald - 1.96 * se_wald),
                        "or_uci95": np.exp(b_wald + 1.96 * se_wald),
                        "pval": p_val,
                        "Q": np.nan,
                        "Q_pval": np.nan
                    })
            elif n > 1:
                b_e_s = pd.Series(mr_data_rev["BETA_e"].values)
                se_e_s = pd.Series(mr_data_rev["SE_e"].values)
                b_o_s = pd.Series(mr_data_rev["BETA_o"].values)
                se_o_s = pd.Series(mr_data_rev["SE_o"].values)
                
                ivw_res = MR.mr_ivw(b_e_s, se_e_s, b_o_s, se_o_s)
                if ivw_res:
                    for r in ivw_res:
                        if pd.notnull(r.get("b")):
                            r["exposure"] = outcome_name
                            r["outcome"] = exposure_name
                            r["mean_F"] = mean_f_rev
                            r["or"] = np.exp(r["b"])
                            r["or_lci95"] = np.exp(r["b"] - 1.96 * r["se"])
                            r["or_uci95"] = np.exp(r["b"] + 1.96 * r["se"])
                            all_reverse_results.append(r)

    rev_df = pd.DataFrame(all_reverse_results)
    rev_df.to_csv(reverse_output_file, index=False)
    print(f"Saved {len(rev_df)} total reverse MR results to {reverse_output_file}")
    
    # Generate Academic PDF & CSV Tables for Reverse MR
    generate_reverse_academic_table(rev_df, script_dir)

def generate_reverse_academic_table(df, script_dir):
    def simplify_name(name):
        parts = str(name).split('.id.')
        return parts[0].replace('.', ' ')
        
    df['Outcome (Taxonomy)'] = df['outcome'].apply(simplify_name)
    df['Exposure (Cancer)'] = df['exposure'].apply(lambda x: 'Adenocarcinoma' if x == 'adeno' else 'Neuroendocrine')
    
    df['OR (95% CI)'] = df.apply(
        lambda r: f"{r['or']:.2f} ({r['or_lci95']:.2f}–{r['or_uci95']:.2f})" if pd.notnull(r['or']) else "NA", axis=1
    )
    df['Effect Size ($\\beta$)'] = df['b'].round(4)
    df['Standard Error (SE)'] = df['se'].round(4)
    df['P-value'] = df['pval'].apply(lambda x: f"{x:.2e}" if pd.notnull(x) and x < 0.001 else (f"{x:.4f}" if pd.notnull(x) else "NA"))
    
    cols_to_keep = [
        'Exposure (Cancer)',
        'Outcome (Taxonomy)',
        'Method',
        'OR (95% CI)',
        'Effect Size ($\\beta$)',
        'Standard Error (SE)',
        'P-value',
        'nSNP'
    ]
    df['Method'] = df['method']
    
    final_table = df[cols_to_keep].sort_values(by=['Exposure (Cancer)', 'P-value'], ascending=[True, True])
    
    output_csv = os.path.join(script_dir, "Table_Reverse_MR_IVW.csv")
    output_pdf = os.path.join(script_dir, "Table_Reverse_MR_IVW.pdf")
    final_table.to_csv(output_csv, index=False)
    print(f"Saved Reverse MR academic CSV table to {output_csv}")
    
    # Render PDF Table
    fig, ax = plt.subplots(figsize=(16.0, 0.42 * len(final_table) + 1.2))
    ax.axis('tight')
    ax.axis('off')
    
    num_cols = len(final_table.columns)
    col_widths = [0.15, 0.25, 0.15, 0.17, 0.11, 0.11, 0.08, 0.06]
    
    table = ax.table(cellText=final_table.values,
                     colLabels=final_table.columns,
                     colWidths=col_widths,
                     colColours=['#f2f2f2'] * num_cols,
                     loc='center',
                     cellLoc='center')
                     
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved Reverse MR academic PDF table to {output_pdf}")

    # Significant Reverse Table (p < 0.05)
    sig_df = final_table[final_table['P-value'].apply(lambda p: float(p) < 0.05 if 'e' not in p and p != 'NA' else True)]
    sig_csv = os.path.join(script_dir, "Table_Reverse_MR_Significant.csv")
    sig_pdf = os.path.join(script_dir, "Table_Reverse_MR_Significant.pdf")
    sig_df.to_csv(sig_csv, index=False)
    
    fig, ax = plt.subplots(figsize=(16.0, 0.42 * len(sig_df) + 1.2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=sig_df.values,
                     colLabels=sig_df.columns,
                     colWidths=col_widths,
                     colColours=['#f2f2f2'] * num_cols,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(sig_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved Reverse MR Significant PDF table to {sig_pdf}")

if __name__ == "__main__":
    run_reverse_analysis()
