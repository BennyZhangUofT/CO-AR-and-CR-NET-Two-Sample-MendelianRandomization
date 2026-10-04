import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from genal.MR_tools import harmonize_MR
from genal.Geno import Geno
import genal.MR as MR

def simplify_name(name):
    parts = str(name).split('.id.')
    return parts[0].replace('.', ' ')

def calculate_ivw(b_e, se_e, b_o, se_o):
    """Calculate fixed/random-effects IVW estimate and SE."""
    weights = 1.0 / (se_o ** 2)
    b_wald = b_o / b_e
    w_sum = np.sum(weights * (b_e ** 2))
    if w_sum <= 0:
        return np.nan, np.nan, np.nan
    
    b_ivw = np.sum(weights * b_e * b_o) / w_sum
    
    n = len(b_e)
    if n > 1:
        # Residual variance correction for random effects
        q_stat = np.sum(weights * ((b_o - b_ivw * b_e) ** 2))
        res_var = max(1.0, q_stat / (n - 1))
        se_ivw = np.sqrt(res_var / w_sum)
    else:
        se_ivw = np.sqrt(1.0 / w_sum)
        
    z = b_ivw / se_ivw
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return b_ivw, se_ivw, pval

def run_leave_one_out_analysis():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Cleaned")
    finngen_dir = os.path.join(script_dir, "FinnGen")
    output_dir = os.path.join(script_dir, "LeaveOneOut_Results")
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================================")
    print("      Starting Leave-One-Out (LOO) Mendelian Randomization       ")
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
    print("\n[1/3] Pre-loading raw outcome summary statistics from FinnGen...")
    for out_name, out_path in cancers_config.items():
        if not os.path.exists(out_path):
            print(f"  Warning: Outcome file not found {out_path}")
            continue
        print(f"  Reading raw TSV for '{out_name}' ({os.path.basename(out_path)})...")
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
            print(f"  Loaded {out_name}: {len(df)} variants matching exposure panel.")

    # 3. Read Forward MR Results to identify eligible significant exposures (P < 0.05 & nSNP >= 3)
    mr_results_path = os.path.join(script_dir, "MR_results_multi_exposure.csv")
    if not os.path.exists(mr_results_path):
        print(f"Error: {mr_results_path} not found.")
        return

    mr_df = pd.read_csv(mr_results_path)
    ivw_sig = mr_df[(mr_df["method"] == "Inverse-Variance Weighted") & (mr_df["pval"] < 0.05) & (mr_df["nSNP"] >= 3)].copy()
    print(f"\n[2/3] Found {len(ivw_sig)} significant exposure-outcome pairs eligible for LOO analysis (nSNP >= 3).")

    all_loo_summary = []

    # 4. Iterate over eligible exposures
    print("\n[3/3] Generating Leave-One-Out plots and statistics...")
    for idx, row in ivw_sig.iterrows():
        exp_name = row["exposure"]
        outcome_name = row["outcome"]
        clean_exp_name = simplify_name(exp_name)
        cancer_label = "Adenocarcinoma" if outcome_name == "adeno" else "Neuroendocrine"
        
        # Stricter LD Clumping on Exposure
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
        out_df = outcomes_data[outcome_name]
        
        try:
            mr_data = harmonize_MR(exp_df, out_df, action=3)
        except Exception:
            continue

        cols_num = ["BETA_e", "SE_e", "BETA_o", "SE_o"]
        mr_data[cols_num] = mr_data[cols_num].apply(pd.to_numeric, errors="coerce")
        mr_data.dropna(subset=cols_num, inplace=True)
        mr_data = mr_data[(mr_data["SE_e"] > 0) & (mr_data["SE_o"] > 0)].copy()

        n_snps = len(mr_data)
        if n_snps < 3:
            continue

        print(f"  [{idx+1}/{len(ivw_sig)}] Processing LOO for: {clean_exp_name} → {cancer_label} ({n_snps} SNPs)...")

        b_e = mr_data["BETA_e"].values
        se_e = mr_data["SE_e"].values
        b_o = mr_data["BETA_o"].values
        se_o = mr_data["SE_o"].values
        snps = mr_data["SNP"].values

        # Overall IVW Estimate
        b_overall, se_overall, p_overall = calculate_ivw(b_e, se_e, b_o, se_o)

        # Leave-One-Out Iteration
        loo_rows = []
        loo_rows.append({
            "Omitted_SNP": "All SNPs (Overall IVW)",
            "b": b_overall,
            "se": se_overall,
            "or": np.exp(b_overall),
            "or_lci95": np.exp(b_overall - 1.96 * se_overall),
            "or_uci95": np.exp(b_overall + 1.96 * se_overall),
            "pval": p_overall
        })

        for i in range(n_snps):
            omitted = snps[i]
            idx_keep = [j for j in range(n_snps) if j != i]
            b_e_sub, se_e_sub = b_e[idx_keep], se_e[idx_keep]
            b_o_sub, se_o_sub = b_o[idx_keep], se_o[idx_keep]

            b_sub, se_sub, p_sub = calculate_ivw(b_e_sub, se_e_sub, b_o_sub, se_o_sub)
            loo_rows.append({
                "Omitted_SNP": f"w/o {omitted}",
                "b": b_sub,
                "se": se_sub,
                "or": np.exp(b_sub),
                "or_lci95": np.exp(b_sub - 1.96 * se_sub),
                "or_uci95": np.exp(b_sub + 1.96 * se_sub),
                "pval": p_sub
            })
            
            all_loo_summary.append({
                "exposure": exp_name,
                "outcome": outcome_name,
                "omitted_snp": omitted,
                "b_loo": b_sub,
                "se_loo": se_sub,
                "or_loo": np.exp(b_sub),
                "pval_loo": p_sub,
                "b_overall": b_overall,
                "or_overall": np.exp(b_overall),
                "pval_overall": p_overall
            })

        loo_df = pd.DataFrame(loo_rows)

        # File Naming
        safe_name = exp_name.replace(".", "_").replace(" ", "_")
        scatter_pdf = os.path.join(output_dir, f"ScatterPlot_{safe_name}_{outcome_name}.pdf")
        scatter_png = os.path.join(output_dir, f"ScatterPlot_{safe_name}_{outcome_name}.png")
        forest_pdf = os.path.join(output_dir, f"LOO_ForestPlot_{safe_name}_{outcome_name}.pdf")
        forest_png = os.path.join(output_dir, f"LOO_ForestPlot_{safe_name}_{outcome_name}.png")

        # --- A. SCATTER PLOT ---
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        ax.errorbar(
            b_e, b_o, xerr=se_e, yerr=se_o,
            fmt='o', color='#1f77b4', ecolor='#aec7e8',
            elinewidth=1.2, capsize=3, markersize=6, alpha=0.85, label='Instruments'
        )

        # Fitted Regressions
        x_grid = np.linspace(min(b_e) * 1.1, max(b_e) * 1.1, 100)
        
        # IVW Line
        ax.plot(x_grid, b_overall * x_grid, color='#d9534f', linewidth=2.2, label=f'IVW Slope: {b_overall:.3f} (OR={np.exp(b_overall):.2f})')

        # Additional Sensitivity Slopes via genal.MR
        b_e_s = pd.Series(b_e)
        se_e_s = pd.Series(se_e)
        b_o_s = pd.Series(b_o)
        se_o_s = pd.Series(se_o)

        try:
            mre_res = MR.mr_egger_regression(b_e_s, se_e_s, b_o_s, se_o_s)
            if mre_res and pd.notnull(mre_res.get('b')):
                b_mre = mre_res['b']
                b_int = mre_res.get('intercept', 0.0)
                ax.plot(x_grid, b_int + b_mre * x_grid, color='#ff7f0e', linestyle='--', linewidth=1.8, label=f'MR-Egger Slope: {b_mre:.3f}')
        except Exception:
            pass

        try:
            wm_res = MR.mr_weighted_median(b_e_s, se_e_s, b_o_s, se_o_s)
            if wm_res and pd.notnull(wm_res[0].get('b')):
                b_wm = wm_res[0]['b']
                ax.plot(x_grid, b_wm * x_grid, color='#2ca02c', linestyle=':', linewidth=1.8, label=f'Weighted Median: {b_wm:.3f}')
        except Exception:
            pass

        # Annotate top variants on scatter plot
        for i_snp, snp_name in enumerate(snps):
            ax.annotate(
                snp_name, (b_e[i_snp], b_o[i_snp]),
                textcoords="offset points", xytext=(5, 5), ha='left', fontsize=7.5, alpha=0.85
            )

        ax.axhline(0, color='gray', linestyle=':', alpha=0.6)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.6)
        ax.set_xlabel(f'Genetic Effect on Exposure ({clean_exp_name})', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Genetic Effect on Outcome ({cancer_label})', fontsize=11, fontweight='bold')
        ax.set_title(f'MR Scatter Plot: {clean_exp_name} → {cancer_label}', fontsize=12, fontweight='bold', pad=12)
        ax.legend(loc='upper left', fontsize=9.0, frameon=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(scatter_pdf, bbox_inches='tight', dpi=300)
        plt.savefig(scatter_png, bbox_inches='tight', dpi=300)
        plt.close()

        # --- B. LEAVE-ONE-OUT FOREST PLOT ---
        fig_h = max(5.5, 0.4 * len(loo_df) + 1.8)
        fig, ax = plt.subplots(figsize=(10.5, fig_h))

        y_pos = np.arange(len(loo_df))
        
        for i_loo, r_loo in loo_df.iterrows():
            is_overall = r_loo['Omitted_SNP'] == "All SNPs (Overall IVW)"
            col = '#d9534f' if is_overall else '#1f77b4'
            fmt = 'D' if is_overall else 'o'
            msize = 8.0 if is_overall else 6.0
            
            ax.errorbar(
                x=r_loo['or'], y=y_pos[i_loo],
                xerr=[[r_loo['or'] - r_loo['or_lci95']], [r_loo['or_uci95'] - r_loo['or']]],
                fmt=fmt, color=col, ecolor=col,
                elinewidth=2.0 if is_overall else 1.5,
                capsize=4.0 if is_overall else 3.0,
                markersize=msize
            )
            
            p_str = f"{r_loo['pval']:.2e}" if r_loo['pval'] < 0.001 else f"{r_loo['pval']:.4f}"
            anno = f"OR: {r_loo['or']:.2f} [{r_loo['or_lci95']:.2f}, {r_loo['or_uci95']:.2f}]  (P = {p_str})"
            ax.text(
                max(loo_df['or_uci95']) * 1.05, y_pos[i_loo], anno,
                va='center', ha='left', fontsize=9.0, fontfamily='monospace',
                fontweight='bold' if is_overall else 'normal'
            )

        ax.axvline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(loo_df['Omitted_SNP'], fontsize=9.5)
        ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title(f'Leave-One-Out Sensitivity Analysis: {clean_exp_name} → {cancer_label}', fontsize=12, fontweight='bold', pad=15)
        
        min_x = max(0.1, min(loo_df['or_lci95'].min() * 0.85, 0.4))
        max_x = max(loo_df['or_uci95'].max() * 1.15, 2.0)
        ax.set_xlim(min_x, max_x)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle=':', alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(right=0.55)
        plt.savefig(forest_pdf, bbox_inches='tight', dpi=300)
        plt.savefig(forest_png, bbox_inches='tight', dpi=300)
        plt.close()

    # Save summary LOO CSV table
    loo_summary_df = pd.DataFrame(all_loo_summary)
    loo_csv_path = os.path.join(output_dir, "LeaveOneOut_Summary_Table.csv")
    loo_summary_df.to_csv(loo_csv_path, index=False)

    print("\n==================================================================")
    print(f" Leave-One-Out Analysis Complete!")
    print(f" Saved all PDF diagrams & LOO summary table to directory:")
    print(f" {output_dir}")
    print("==================================================================")

if __name__ == "__main__":
    run_leave_one_out_analysis()
