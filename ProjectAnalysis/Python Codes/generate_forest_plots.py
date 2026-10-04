import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def simplify_name(name):
    parts = str(name).split('.id.')
    return parts[0].replace('.', ' ')

def create_single_forest_plot(df, title, output_pdf, output_png):
    if df.empty:
        print(f"No data for {title}")
        return

    df = df.sort_values(by='pval', ascending=True).copy()

    if 'or' not in df.columns:
        df['or'] = np.exp(df['b'])
        df['or_lci95'] = np.exp(df['b'] - 1.96 * df['se'])
        df['or_uci95'] = np.exp(df['b'] + 1.96 * df['se'])

    df['label'] = df['exposure'].apply(simplify_name)
    df = df.iloc[::-1].reset_index(drop=True)

    n_rows = len(df)
    fig_height = max(6, 0.5 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(15, fig_height))

    y_pos = np.arange(n_rows)
    colors = ['#d9534f' if r['or'] > 1 else '#0275d8' for _, r in df.iterrows()]

    for i, row in df.iterrows():
        ax.errorbar(
            x=row['or'],
            y=y_pos[i],
            xerr=[[row['or'] - row['or_lci95']], [row['or_uci95'] - row['or']]],
            fmt='o',
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2.0,
            capsize=4.0,
            capthick=1.5,
            markersize=7
        )

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['label'], fontsize=10.5, fontweight='medium')
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Forest Plot: {title} (with Complete Test Statistics)', fontsize=14, fontweight='bold', pad=20)

    min_x = max(0.1, min(df['or_lci95'].min() * 0.85, 0.4))
    max_x = max(df['or_uci95'].max() * 1.15, 2.2)
    ax.set_xlim(min_x, max_x)

    # Column headers for right annotation table
    header_text = f"{'nSNP':<6}{'Mean F':<8}{'OR (95% CI)':<20}{'P-val':<10}{'P (Q)':<10}{'P (Pleio)':<10}"
    ax.text(max_x * 1.05, n_rows - 0.2, header_text, va='bottom', ha='left', fontsize=9.5, fontweight='bold', fontfamily='monospace')

    for i, row in df.iterrows():
        p_str = f"{row['pval']:.2e}" if row['pval'] < 0.001 else f"{row['pval']:.4f}"
        q_str = f"{row['Q_pval']:.3f}" if pd.notnull(row.get('Q_pval')) else "NA"
        pleio_str = f"{row['pleio_pval']:.3f}" if pd.notnull(row.get('pleio_pval')) else "NA"
        nsnp_str = f"{int(row['nSNP'])}" if pd.notnull(row.get('nSNP')) else "NA"
        f_str = f"{row['mean_F']:.1f}" if pd.notnull(row.get('mean_F')) else "NA"
        or_ci_str = f"{row['or']:.2f} [{row['or_lci95']:.2f}, {row['or_uci95']:.2f}]"

        line = f"{nsnp_str:<6}{f_str:<8}{or_ci_str:<20}{p_str:<10}{q_str:<10}{pleio_str:<10}"
        ax.text(
            max_x * 1.05, y_pos[i], line,
            va='center', ha='left', fontsize=9.0, fontfamily='monospace'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(right=0.45)

    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved detailed forest plot for {title} to {output_pdf} and {output_png}")


def create_combined_forest_plot(df_adeno, df_neuro, output_pdf, output_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 11), gridspec_kw={'width_ratios': [1.2, 1]})

    # --- Panel A: Adenocarcinoma ---
    df_a = df_adeno.sort_values(by='pval', ascending=True).iloc[::-1].reset_index(drop=True)
    y_pos_a = np.arange(len(df_a))
    colors_a = ['#d9534f' if r['or'] > 1 else '#0275d8' for _, r in df_a.iterrows()]

    for i, row in df_a.iterrows():
        ax1.errorbar(
            x=row['or'], y=y_pos_a[i],
            xerr=[[row['or'] - row['or_lci95']], [row['or_uci95'] - row['or']]],
            fmt='o', color=colors_a[i], ecolor=colors_a[i],
            elinewidth=1.8, capsize=3.5, capthick=1.2, markersize=6.5
        )
        p_str = f"{row['pval']:.2e}" if row['pval'] < 0.001 else f"{row['pval']:.4f}"
        q_str = f"{row['Q_pval']:.2f}" if pd.notnull(row.get('Q_pval')) else "NA"
        ax1.text(2.1, y_pos_a[i], f"OR:{row['or']:.2f} [{row['or_lci95']:.2f}-{row['or_uci95']:.2f}] P:{p_str} P_Q:{q_str}", va='center', fontsize=8.0, fontfamily='monospace')

    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_yticks(y_pos_a)
    ax1.set_yticklabels(df_a['exposure'].apply(simplify_name), fontsize=9.5)
    ax1.set_xlabel('Odds Ratio (95% CI)', fontsize=11, fontweight='bold')
    ax1.set_title('A: Colorectal Adenocarcinoma ($p < 0.05$)', fontsize=13, fontweight='bold')
    ax1.set_xlim(0.5, 3.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', linestyle=':', alpha=0.5)

    # --- Panel B: Neuroendocrine ---
    df_n = df_neuro.sort_values(by='pval', ascending=True).iloc[::-1].reset_index(drop=True)
    y_pos_n = np.arange(len(df_n))
    colors_n = ['#d9534f' if r['or'] > 1 else '#0275d8' for _, r in df_n.iterrows()]

    for i, row in df_n.iterrows():
        ax2.errorbar(
            x=row['or'], y=y_pos_n[i],
            xerr=[[row['or'] - row['or_lci95']], [row['or_uci95'] - row['or']]],
            fmt='o', color=colors_n[i], ecolor=colors_n[i],
            elinewidth=1.8, capsize=3.5, capthick=1.2, markersize=6.5
        )
        p_str = f"{row['pval']:.2e}" if row['pval'] < 0.001 else f"{row['pval']:.4f}"
        q_str = f"{row['Q_pval']:.2f}" if pd.notnull(row.get('Q_pval')) else "NA"
        ax2.text(5.5, y_pos_n[i], f"OR:{row['or']:.2f} [{row['or_lci95']:.2f}-{row['or_uci95']:.2f}] P:{p_str} P_Q:{q_str}", va='center', fontsize=8.0, fontfamily='monospace')

    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_yticks(y_pos_n)
    ax2.set_yticklabels(df_n['exposure'].apply(simplify_name), fontsize=9.5)
    ax2.set_xlabel('Odds Ratio (95% CI)', fontsize=11, fontweight='bold')
    ax2.set_title('B: Colorectal Neuroendocrine ($p < 0.05$)', fontsize=13, fontweight='bold')
    ax2.set_xlim(0.2, 9.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', linestyle=':', alpha=0.5)

    plt.suptitle('Mendelian Randomization Forest Plots with Comprehensive Test Statistics', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved combined detailed forest plot to {output_pdf} and {output_png}")


def create_multi_method_forest_plot(df_all, outcome_target, title, output_pdf, output_png):
    # Filter for target outcome and significant IVW exposures
    df_target = df_all[df_all['outcome'] == outcome_target].copy()
    ivw_sig = df_target[(df_target['method'] == 'Inverse-Variance Weighted') & (df_target['pval'] < 0.05)]
    
    if ivw_sig.empty:
        print(f"No significant IVW exposures for {outcome_target}")
        return

    sig_exposures = ivw_sig['exposure'].unique()
    sub_df = df_target[df_target['exposure'].isin(sig_exposures)].copy()
    
    # Exclude Egger Intercept from forest plot errorbars (it is a pleiotropy test, not an OR effect)
    sub_df = sub_df[sub_df['method'] != 'Egger Intercept'].copy()

    methods_order = ['Inverse-Variance Weighted', 'MR Egger', 'Weighted Median', 'Weighted mode']
    method_colors = {
        'Inverse-Variance Weighted': '#1f77b4',
        'MR Egger': '#ff7f0e',
        'Weighted Median': '#2ca02c',
        'Weighted mode': '#d62728'
    }

    exposures_sorted = ivw_sig.sort_values(by='pval')['exposure'].tolist()
    
    fig_rows = len(exposures_sorted) * 5
    fig_height = max(7, 0.35 * fig_rows + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y_ticks = []
    y_labels = []
    current_y = 0

    for exp in reversed(exposures_sorted):
        exp_label = simplify_name(exp)
        exp_sub = sub_df[sub_df['exposure'] == exp]
        
        # Section header label
        current_y += 1
        
        for m in reversed(methods_order):
            m_data = exp_sub[exp_sub['method'] == m]
            if not m_data.empty:
                r = m_data.iloc[0]
                or_val = r['or']
                or_lci = r['or_lci95']
                or_uci = r['or_uci95']
                p_val = r['pval']
                
                ax.errorbar(
                    x=or_val, y=current_y,
                    xerr=[[or_val - or_lci], [or_uci - or_val]],
                    fmt='s' if m == 'Inverse-Variance Weighted' else 'o',
                    color=method_colors.get(m, 'gray'),
                    ecolor=method_colors.get(m, 'gray'),
                    elinewidth=1.6, capsize=3.0, markersize=6
                )
                
                p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
                annotation = f"{m:<26} OR: {or_val:.2f} [{or_lci:.2f}, {or_uci:.2f}]  P = {p_str}"
                ax.text(3.5, current_y, annotation, va='center', ha='left', fontsize=8.5, fontfamily='monospace')
                
                y_ticks.append(current_y)
                y_labels.append(f"{exp_label} ({m})")
                current_y += 1

        current_y += 1 # Spacing between taxa

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Multi-Method Sensitivity Forest Plot: {title}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0.3, 3.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(right=0.48)
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved multi-method forest plot for {title} to {output_pdf} and {output_png}")


def generate_all_forest_plots():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "MR_results_multi_exposure.csv")

    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    df = pd.read_csv(input_file)
    df_ivw = df[df['method'] == 'Inverse-Variance Weighted'].copy()
    
    # Extract Egger Intercept pleiotropy p-values
    egger_df = df[df['method'] == 'Egger Intercept'][['exposure', 'outcome', 'pval']].rename(columns={'pval': 'pleio_pval'})
    df_ivw = pd.merge(df_ivw, egger_df, on=['exposure', 'outcome'], how='left')

    df_adeno_sig = df_ivw[(df_ivw['outcome'] == 'adeno') & (df_ivw['pval'] < 0.05)].copy()
    pdf_adeno = os.path.join(script_dir, "Forest_Plot_Adenocarcinoma.pdf")
    png_adeno = os.path.join(script_dir, "Forest_Plot_Adenocarcinoma.png")
    create_single_forest_plot(df_adeno_sig, "Colorectal Adenocarcinoma", pdf_adeno, png_adeno)

    df_neuro_sig = df_ivw[(df_ivw['outcome'] == 'neuro') & (df_ivw['pval'] < 0.05)].copy()
    pdf_neuro = os.path.join(script_dir, "Forest_Plot_Neuroendocrine.pdf")
    png_neuro = os.path.join(script_dir, "Forest_Plot_Neuroendocrine.png")
    create_single_forest_plot(df_neuro_sig, "Colorectal Neuroendocrine", pdf_neuro, png_neuro)

    # Combined multi-panel forest plot
    pdf_comb = os.path.join(script_dir, "Forest_Plot_Combined.pdf")
    png_comb = os.path.join(script_dir, "Forest_Plot_Combined.png")
    create_combined_forest_plot(df_adeno_sig, df_neuro_sig, pdf_comb, png_comb)

    # Multi-Method Sensitivity Forest Plots
    pdf_multi_adeno = os.path.join(script_dir, "Forest_Plot_MultiMethod_Adenocarcinoma.pdf")
    png_multi_adeno = os.path.join(script_dir, "Forest_Plot_MultiMethod_Adenocarcinoma.png")
    create_multi_method_forest_plot(df, "adeno", "Colorectal Adenocarcinoma", pdf_multi_adeno, png_multi_adeno)

    pdf_multi_neuro = os.path.join(script_dir, "Forest_Plot_MultiMethod_Neuroendocrine.pdf")
    png_multi_neuro = os.path.join(script_dir, "Forest_Plot_MultiMethod_Neuroendocrine.png")
    create_multi_method_forest_plot(df, "neuro", "Colorectal Neuroendocrine", pdf_multi_neuro, png_multi_neuro)


if __name__ == "__main__":
    generate_all_forest_plots()
