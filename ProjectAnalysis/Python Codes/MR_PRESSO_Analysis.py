import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from genal.MR_tools import harmonize_MR
from genal.Geno import Geno

def simplify_name(name):
    parts = str(name).split('.id.')
    return parts[0].replace('.', ' ')

def run_mr_presso_single(b_e, se_e, b_o, se_o, snps, n_sim=2000, seed=42):
    """
    Implements MR-PRESSO (Mendelian Randomization Pleiotropy RESidual Sum and Outlier):
    1. Global Test: Residual Sum of Squares (RSS)
    2. Outlier Test: Individual variant residual P-values
    3. Distortion Test: Percent change in slope after removing outliers
    """
    np.random.seed(seed)
    N = len(b_e)
    if N < 3:
        return None

    weights = 1.0 / (se_o ** 2)
    w_sum = np.sum(weights * (b_e ** 2))
    b_ivw = np.sum(weights * b_e * b_o) / w_sum
    
    # Residuals from IVW slope
    res = b_o - b_ivw * b_e
    rss_obs = np.sum(weights * (res ** 2))
    d_obs = weights * (res ** 2)
    
    # Monte Carlo simulation under H0 (no pleiotropy)
    sim_b_o = np.random.normal(loc=b_ivw * b_e, scale=se_o, size=(n_sim, N))
    sim_w_sum = np.sum(weights * (b_e ** 2))
    sim_b_ivw = np.sum(weights * b_e * sim_b_o, axis=1) / sim_w_sum
    
    sim_res = sim_b_o - np.outer(sim_b_ivw, b_e)
    sim_rss = np.sum(weights * (sim_res ** 2), axis=1)
    sim_d = weights * (sim_res ** 2)
    
    # 1. Global Test P-value
    p_global = (1.0 + np.sum(sim_rss >= rss_obs)) / (n_sim + 1.0)
    
    # 2. Outlier Test per Variant
    p_outliers = []
    for i in range(N):
        p_out = (1.0 + np.sum(sim_d[:, i] >= d_obs[i])) / (n_sim + 1.0)
        p_outliers.append(p_out)
        
    outliers_idx = [i for i in range(N) if p_outliers[i] < 0.05]
    outlier_snps = [snps[i] for i in outliers_idx]
    
    # 3. Outlier-Corrected IVW
    keep_idx = [i for i in range(N) if i not in outliers_idx]
    if len(keep_idx) >= 2:
        b_e_c, se_e_c = b_e[keep_idx], se_e[keep_idx]
        b_o_c, se_o_c = b_o[keep_idx], se_o[keep_idx]
        
        w_c = 1.0 / (se_o_c ** 2)
        w_sum_c = np.sum(w_c * (b_e_c ** 2))
        b_ivw_c = np.sum(w_c * b_e_c * b_o_c) / w_sum_c
        q_c = np.sum(w_c * ((b_o_c - b_ivw_c * b_e_c) ** 2))
        res_var_c = max(1.0, q_c / (len(keep_idx) - 1))
        se_ivw_c = np.sqrt(res_var_c / w_sum_c)
        z_c = b_ivw_c / se_ivw_c
        p_ivw_c = 2 * (1 - stats.norm.cdf(abs(z_c)))
        
        distortion_pct = ((b_ivw_c - b_ivw) / abs(b_ivw)) * 100.0 if b_ivw != 0 else np.nan
    else:
        b_ivw_c, se_ivw_c, p_ivw_c, distortion_pct = np.nan, np.nan, np.nan, np.nan
        
    return {
        "nSNP_orig": N,
        "b_orig": b_ivw,
        "or_orig": np.exp(b_ivw),
        "rss_obs": rss_obs,
        "p_global": p_global,
        "n_outliers": len(outlier_snps),
        "outlier_snps": ", ".join(outlier_snps) if outlier_snps else "None",
        "b_corrected": b_ivw_c,
        "se_corrected": se_ivw_c,
        "or_corrected": np.exp(b_ivw_c) if pd.notnull(b_ivw_c) else np.nan,
        "or_lci95_corrected": np.exp(b_ivw_c - 1.96 * se_ivw_c) if pd.notnull(b_ivw_c) else np.nan,
        "or_uci95_corrected": np.exp(b_ivw_c + 1.96 * se_ivw_c) if pd.notnull(b_ivw_c) else np.nan,
        "p_corrected": p_ivw_c,
        "distortion_pct": distortion_pct
    }

def run_mr_presso_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Cleaned")
    finngen_dir = os.path.join(script_dir, "FinnGen")
    output_csv = os.path.join(script_dir, "MR_PRESSO_Results.csv")

    print("==================================================================")
    print("      Starting MR-PRESSO Outlier and Pleiotropy Analysis         ")
    print("==================================================================")

    # 1. Load Exposure Data
    exp_path = os.path.join(data_dir, "MiBioGen_allHits.csv")
    full_exp_df = pd.read_csv(exp_path)
    exp_rename_map = {
        "rsID": "SNP", "beta": "BETA", "SE": "SE",
        "eff.allele": "EA", "ref.allele": "NEA",
        "P.weightedSumZ": "P", "N": "N"
    }
    full_exp_df.rename(columns=exp_rename_map, inplace=True)
    exp_snps = set(full_exp_df["SNP"].dropna().unique())

    # 2. Load Outcome Data (raw FinnGen summary stats)
    use_cols = ["#chrom", "pos", "ref", "alt", "rsids", "pval", "beta", "sebeta"]
    out_rename_map = {
        "rsids": "SNP", "beta": "BETA", "sebeta": "SE",
        "alt": "EA", "ref": "NEA", "pval": "P",
        "#chrom": "CHR", "chrom": "CHR", "pos": "POS"
    }

    cancers_config = {
        "adeno": (
            os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_ADENO_EXALLC")
            if os.path.exists(os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_ADENO_EXALLC"))
            else os.path.join(finngen_dir, "C3_COLORECTAL_ADENO_EXALLC.tsv")
        ),
        "neuro": os.path.join(finngen_dir, "finngen_R12_C3_COLORECTAL_NEUROENDO_EXALLC")
    }

    outcomes_data = {}
    print("\n[1/3] Loading raw outcome summary statistics...")
    for out_name, out_path in cancers_config.items():
        if not os.path.exists(out_path):
            continue
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

    # 3. Read Forward MR Results to identify target exposures with directional pleiotropy (Egger Intercept P < 0.05)
    mr_results_path = os.path.join(script_dir, "MR_results_multi_exposure.csv")
    if not os.path.exists(mr_results_path):
        print(f"Error: {mr_results_path} not found.")
        return

    mr_df = pd.read_csv(mr_results_path)
    
    # Targets: Exposures with Egger Intercept P < 0.05 OR IVW P < 0.05
    pleio_targets = mr_df[(mr_df["method"] == "Egger Intercept") & (mr_df["pval"] < 0.05)][["exposure", "outcome"]].copy()
    print(f"\n[2/3] Identified {len(pleio_targets)} exposure-outcome pairs with significant directional pleiotropy (MR-Egger Intercept P < 0.05).")

    presso_rows = []

    print("\n[3/3] Running MR-PRESSO Global, Outlier, and Distortion Tests...")
    for idx, row in pleio_targets.iterrows():
        exp_name = row["exposure"]
        outcome_name = row["outcome"]
        clean_exp = simplify_name(exp_name)
        cancer_label = "Adenocarcinoma" if outcome_name == "adeno" else "Neuroendocrine"

        exp_df = full_exp_df[full_exp_df["bac"] == exp_name].copy()
        exp_df = exp_df[exp_df["P"] < 1e-5].copy()
        if len(exp_df) == 0:
            continue
            
        clump_input = exp_df.rename(columns={"chr": "CHR", "bp": "POS"})
        geno = Geno(clump_input[["SNP", "P", "CHR", "POS"]], keep_columns=True)
        try:
            clumped_geno = geno.clump(kb=10000, r2=0.001, p1=1e-5, p2=1e-5)
            if clumped_geno is None:
                continue
            clumped_snps = clumped_geno.data["SNP"].tolist()
            exp_df = exp_df[exp_df["SNP"].isin(clumped_snps)].copy()
        except Exception:
            continue

        if outcome_name not in outcomes_data:
            continue
            
        mr_data = harmonize_MR(exp_df, outcomes_data[outcome_name], action=3)
        cols_num = ["BETA_e", "SE_e", "BETA_o", "SE_o"]
        mr_data[cols_num] = mr_data[cols_num].apply(pd.to_numeric, errors="coerce")
        mr_data.dropna(subset=cols_num, inplace=True)
        mr_data = mr_data[(mr_data["SE_e"] > 0) & (mr_data["SE_o"] > 0)].copy()

        if len(mr_data) < 3:
            continue

        res = run_mr_presso_single(
            mr_data["BETA_e"].values,
            mr_data["SE_e"].values,
            mr_data["BETA_o"].values,
            mr_data["SE_o"].values,
            mr_data["SNP"].values
        )

        if res:
            res["exposure"] = clean_exp
            res["outcome"] = cancer_label
            presso_rows.append(res)
            print(f"  • {clean_exp} → {cancer_label}: Global P = {res['p_global']:.4f} | Outliers = {res['n_outliers']} ({res['outlier_snps']})")

    presso_df = pd.DataFrame(presso_rows)
    presso_df.to_csv(output_csv, index=False)
    print(f"\nSaved MR-PRESSO results to: {output_csv}")

    # Render Academic PDF Table
    generate_presso_pdf_table(presso_df, script_dir)

def generate_presso_pdf_table(df, script_dir):
    if df.empty:
        return

    cols_to_display = [
        "exposure", "outcome", "nSNP_orig", "rss_obs", "p_global",
        "n_outliers", "outlier_snps", "or_corrected", "p_corrected", "distortion_pct"
    ]
    
    tbl = df[cols_to_display].copy()
    tbl.columns = [
        "Exposure", "Outcome", "nSNP", "Global RSS", "Global P",
        "Outliers (N)", "Outlier SNPs", "Corrected OR", "Corrected P", "Distortion (%)"
    ]
    
    tbl["Global RSS"] = tbl["Global RSS"].round(2)
    tbl["Global P"] = tbl["Global P"].apply(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}")
    tbl["Corrected OR"] = tbl["Corrected OR"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "NA")
    tbl["Corrected P"] = tbl["Corrected P"].apply(lambda p: f"{p:.2e}" if pd.notnull(p) and p < 0.001 else (f"{p:.4f}" if pd.notnull(p) else "NA"))
    tbl["Distortion (%)"] = tbl["Distortion (%)"].apply(lambda d: f"{d:+.1f}%" if pd.notnull(d) else "NA")

    pdf_out = os.path.join(script_dir, "Table_MR_PRESSO_Outliers.pdf")
    csv_out = os.path.join(script_dir, "Table_MR_PRESSO_Outliers.csv")
    tbl.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(16, 0.45 * len(tbl) + 1.5))
    ax.axis("tight")
    ax.axis("off")
    
    col_widths = [0.22, 0.12, 0.06, 0.08, 0.08, 0.08, 0.18, 0.09, 0.09, 0.08]
    table = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        colWidths=col_widths,
        colColours=["#f2f2f2"] * len(tbl.columns),
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.close()
    print(f"Saved MR-PRESSO academic PDF table to: {pdf_out}")

if __name__ == "__main__":
    run_mr_presso_pipeline()
