import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os

def generate_volcano_plot(input_file, outcome_name, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    # Load data
    df = pd.read_csv(input_file)
    
    # Filter for the primary method, Inverse-Variance Weighted, to represent the main effect size
    # Volcano plots usually represent one method's effect size to avoid plotting the same trait 4 times
    df_ivw = df[df['method'] == 'Inverse-Variance Weighted'].copy()
    
    if df_ivw.empty:
        print(f"No Inverse-Variance Weighted results found in {input_file}")
        return

    # Calculate -log10(p-value)
    # To avoid log(0) in case of extremely small p-values, we can add a small constant or clip
    df_ivw['nlog10_p'] = -np.log10(df_ivw['pval'].clip(lower=1e-300))

    # Keep the taxonomy but replace periods with spaces, and remove the '.id.XYZ' suffix
    def simplify_name(name):
        # Example: 'genus.Bifidobacterium.id.436' -> 'genus Bifidobacterium'
        parts = name.split('.id.')
        # Take the part before '.id.' and replace periods with spaces
        return parts[0].replace('.', ' ')
        
    df_ivw['label'] = df_ivw['exposure'].apply(simplify_name)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points with horizontal error bars
    # xerr is the standard error (SE)
    ax.errorbar(
        df_ivw['b'], 
        df_ivw['nlog10_p'], 
        xerr=df_ivw['se'], 
        fmt='o', 
        color='royalblue',
        ecolor='lightgray',
        elinewidth=1.5,
        capsize=3,
        markersize=6,
        alpha=0.8
    )

    # Add labels
    texts = []
    for i, row in df_ivw.iterrows():
        if row['label'] not in ['unknownfamily', 'unknowngenus']:
            texts.append(ax.text(row['b'], row['nlog10_p'], row['label'], fontsize=9))

    # Adjust text to prevent overlap
    print(f"Adjusting labels for {outcome_name} plot... this might take a few seconds.")
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # Add lines for significance thresholds (e.g., P < 0.05/195)
    sig_threshold = -np.log10(0.05 / 195)
    ax.axhline(y=sig_threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Formalize axes
    ax.set_xlabel('Effect Size (B)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$-log_{10}(P-value)$', fontsize=12, fontweight='bold')
    ax.set_title(f'Volcano Plot: {outcome_name}', fontsize=14, fontweight='bold')
    
    # Clean up right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if output_file.endswith('.pdf'):
        plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {outcome_name} volcano plot to {output_file} and {output_file.replace('.pdf', '.png')}")

def generate_diff_volcano_plot(bg_file, fg_file, output_file):
    if not os.path.exists(bg_file):
        print(f"Error: Could not find {bg_file}")
        return
    if not os.path.exists(fg_file):
        print(f"Error: Could not find {fg_file}")
        return

    # Load data
    df_bg = pd.read_csv(bg_file)
    df_fg = pd.read_csv(fg_file)
    
    # Filter for the primary method, Inverse-Variance Weighted
    df_ivw_bg = df_bg[df_bg['method'] == 'Inverse-Variance Weighted'].copy()
    df_ivw_fg = df_fg[df_fg['method'] == 'Inverse-Variance Weighted'].copy()
    
    if df_ivw_bg.empty:
        print(f"No Inverse-Variance Weighted results found in {bg_file}")
        return

    # Calculate -log10(p-value)
    df_ivw_bg['nlog10_p'] = -np.log10(df_ivw_bg['pval_diff'].clip(lower=1e-300))
    df_ivw_fg['nlog10_p'] = -np.log10(df_ivw_fg['pval_diff'].clip(lower=1e-300))

    # Keep the taxonomy but replace periods with spaces, and remove the '.id.XYZ' suffix
    def simplify_name(name):
        # Example: 'genus.Bifidobacterium.id.436' -> 'genus Bifidobacterium'
        parts = name.split('.id.')
        # Take the part before '.id.' and replace periods with spaces
        return parts[0].replace('.', ' ')
        
    df_ivw_bg['label'] = df_ivw_bg['exposure'].apply(simplify_name)
    df_ivw_fg['label'] = df_ivw_fg['exposure'].apply(simplify_name)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Plot all background points in gray
    ax.scatter(
        df_ivw_bg['z_diff'], 
        df_ivw_bg['nlog10_p'], 
        color='lightgray',
        alpha=0.6,
        s=30,
        label='Background (All IVW)'
    )
    
    # 2. Extract strictly consistent exposures (foreground)
    sig_left = df_ivw_fg['z_diff'] < 0
    sig_right = df_ivw_fg['z_diff'] >= 0
    
    # Plot strictly consistent (Z < 0)
    if sig_left.any():
        ax.scatter(
            df_ivw_fg.loc[sig_left, 'z_diff'], 
            df_ivw_fg.loc[sig_left, 'nlog10_p'], 
            color='mediumblue',
            alpha=0.8,
            s=40,
            label='More Associated with Neuroendocrine Tumors (Z < 0)'
        )

    # Plot strictly consistent (Z >= 0)
    if sig_right.any():
        ax.scatter(
            df_ivw_fg.loc[sig_right, 'z_diff'], 
            df_ivw_fg.loc[sig_right, 'nlog10_p'], 
            color='crimson',
            alpha=0.8,
            s=40,
            label='More Associated with Adenocarcinoma (Z > 0)'
        )

    # Add labels ONLY for the strictly consistent hits
    texts = []
    for i, row in df_ivw_fg.iterrows():
        if row['label'] not in ['unknownfamily', 'unknowngenus']:
            texts.append(ax.text(row['z_diff'], row['nlog10_p'], row['label'], fontsize=9))

    # Adjust text to prevent overlap
    print(f"Adjusting labels for Difference plot... this might take a few seconds.")
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

    # Add lines for significance thresholds (e.g., P < 0.05/195)
    sig_threshold = -np.log10(0.05 / 195)
    ax.axhline(y=sig_threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Formalize axes
    ax.set_xlabel('Z-Score ($Z_{diff}$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$-log_{10}(P_{diff})$', fontsize=12, fontweight='bold')
    ax.set_title('Volcano Plot: Difference (Adeno vs Neuro)', fontsize=14, fontweight='bold')
    
    # Clean up right and top borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    if output_file.endswith('.pdf'):
        plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Difference volcano plot to {output_file} and {output_file.replace('.pdf', '.png')}")

def generate_academic_table(input_file, output_csv, output_pdf):
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return
        
    df = pd.read_csv(input_file)
    
    # Filter for IVW only
    df_ivw = df[df['method'] == 'Inverse-Variance Weighted'].copy()
    
    # Sort by P-value (smallest to largest)
    df_ivw = df_ivw.sort_values(by='pval', ascending=True)
    
    if df_ivw.empty:
        print(f"No Inverse-Variance Weighted results found in {input_file}")
        return
        
    # Clean up exposure names
    def simplify_name(name):
        parts = name.split('.id.')
        return parts[0].replace('.', ' ')
        
    df_ivw['Exposure (Taxonomy)'] = df_ivw['exposure'].apply(simplify_name)
    
    # Format the statistics for academic presentation
    df_ivw['Effect Size ($\\beta$)'] = df_ivw['b'].round(4)
    df_ivw['Standard Error (SE)'] = df_ivw['se'].round(4)
    df_ivw['P-value'] = df_ivw['pval'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    
    # Calculate OR (95% CI)
    if 'or' in df_ivw.columns and 'or_lci95' in df_ivw.columns and 'or_uci95' in df_ivw.columns:
        df_ivw['OR (95% CI)'] = df_ivw.apply(
            lambda r: f"{r['or']:.2f} ({r['or_lci95']:.2f}–{r['or_uci95']:.2f})" if pd.notnull(r['or']) else "NA", axis=1
        )
    else:
        df_ivw['OR (95% CI)'] = df_ivw.apply(
            lambda r: f"{np.exp(r['b']):.2f} ({np.exp(r['b'] - 1.96 * r['se']):.2f}–{np.exp(r['b'] + 1.96 * r['se']):.2f})" if pd.notnull(r['b']) and pd.notnull(r['se']) else "NA", axis=1
        )

    # Select and reorder columns
    cols_to_keep = [
        'Exposure (Taxonomy)',
        'OR (95% CI)',
        'Effect Size ($\\beta$)',
        'Standard Error (SE)',
        'P-value',
        'nSNP'
    ]
    
    # If Q stats exist, add them
    if 'Q' in df_ivw.columns and 'Q_pval' in df_ivw.columns:
        df_ivw["Cochran's Q"] = df_ivw['Q'].round(2)
        df_ivw['Q P-value'] = df_ivw['Q_pval'].apply(lambda x: f"{x:.2e}" if pd.notnull(x) and x < 0.001 else (f"{x:.4f}" if pd.notnull(x) else "NA"))
        cols_to_keep.extend(["Cochran's Q", 'Q P-value'])
        
    final_table = df_ivw[cols_to_keep]
    
    # Save the table as CSV
    final_table.to_csv(output_csv, index=False)
    print(f"Saved academic CSV table to {output_csv}")
    
    # Render and save the table as PDF
    fig, ax = plt.subplots(figsize=(15.0, 0.42 * len(final_table) + 1.0)) # Scaled size
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate column widths: allocate 25% to first column, 18% to OR (95% CI)
    num_cols = len(final_table.columns)
    first_col_w = 0.25
    or_col_w = 0.18
    remaining_w = (1.0 - first_col_w - or_col_w) / (num_cols - 2)
    col_widths = [first_col_w, or_col_w] + [remaining_w] * (num_cols - 2)
    
    table = ax.table(cellText=final_table.values,
                     colLabels=final_table.columns,
                     colWidths=col_widths,
                     colColours=['#f2f2f2'] * num_cols,
                     loc='center',
                     cellLoc='center')
                     
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5) # Scale width and height
    
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    if output_pdf.endswith('.pdf'):
        plt.savefig(output_pdf.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved academic PDF and PNG table to {output_pdf}")

def generate_diff_academic_table(input_file, output_csv, output_pdf):
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return
        
    df = pd.read_csv(input_file)
    
    # Filter for IVW only
    df_ivw = df[df['method'] == 'Inverse-Variance Weighted'].copy()
    
    # Sort by P-value diff (smallest to largest)
    df_ivw = df_ivw.sort_values(by='pval_diff', ascending=True)
    
    if df_ivw.empty:
        print(f"No Inverse-Variance Weighted results found in {input_file}")
        return
        
    # Clean up exposure names
    def simplify_name(name):
        parts = name.split('.id.')
        return parts[0].replace('.', ' ')
        
    df_ivw['Exposure (Taxonomy)'] = df_ivw['exposure'].apply(simplify_name)
    
    # Format the statistics for academic presentation
    df_ivw['Z-Score ($Z_{diff}$)'] = df_ivw['z_diff'].round(4)
    df_ivw['P-value ($P_{diff}$)'] = df_ivw['pval_diff'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    
    # Select and reorder columns
    cols_to_keep = [
        'Exposure (Taxonomy)',
        'Z-Score ($Z_{diff}$)',
        'P-value ($P_{diff}$)',
    ]
    
    final_table = df_ivw[cols_to_keep]
    
    # Save the table as CSV
    final_table.to_csv(output_csv, index=False)
    print(f"Saved difference academic CSV table to {output_csv}")
    
    # Render and save the table as PDF
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(final_table) + 1)) # Scale height to rows
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate column widths: increase first column width
    num_cols = len(final_table.columns)
    base_width = 1.0 / num_cols
    col_widths = [base_width] * num_cols
    col_widths[0] = base_width * 1.65
    
    table = ax.table(cellText=final_table.values,
                     colLabels=final_table.columns,
                     colWidths=col_widths,
                     colColours=['#f2f2f2'] * num_cols,
                     loc='center',
                     cellLoc='center')
                     
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5) # Scale width and height
    
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight', dpi=300)
    if output_pdf.endswith('.pdf'):
        plt.savefig(output_pdf.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved difference academic PDF and PNG table to {output_pdf}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Primary Results File
    main_results_file = os.path.join(script_dir, "MR_results_multi_exposure.csv")
    diff_results_file = os.path.join(script_dir, "MR_difference_results.csv")

    if os.path.exists(main_results_file):
        df_all = pd.read_csv(main_results_file)
        
        # Split by outcome for Adeno and Neuro
        df_adeno = df_all[df_all['outcome'] == 'adeno']
        df_neuro = df_all[df_all['outcome'] == 'neuro']
        
        adeno_temp = os.path.join(script_dir, "temp_adeno_results.csv")
        neuro_temp = os.path.join(script_dir, "temp_neuro_results.csv")
        
        df_adeno.to_csv(adeno_temp, index=False)
        df_neuro.to_csv(neuro_temp, index=False)

        # Adeno Plot & Tables
        out_adeno_plot = os.path.join(script_dir, "Volcano_Adeno.pdf")
        out_adeno_table_csv = os.path.join(script_dir, "Table_Adeno_IVW.csv")
        out_adeno_table_pdf = os.path.join(script_dir, "Table_Adeno_IVW.pdf")
        generate_volcano_plot(adeno_temp, "Colorectal Adenocarcinoma", out_adeno_plot)
        generate_academic_table(adeno_temp, out_adeno_table_csv, out_adeno_table_pdf)
        
        # Adeno Significant Table Only (p < 0.05)
        df_adeno_sig = df_adeno[df_adeno['pval'] < 0.05].copy()
        adeno_sig_temp = os.path.join(script_dir, "temp_adeno_sig.csv")
        df_adeno_sig.to_csv(adeno_sig_temp, index=False)
        out_adeno_sig_csv = os.path.join(script_dir, "Table_Adeno_Significant.csv")
        out_adeno_sig_pdf = os.path.join(script_dir, "Table_Adeno_Significant.pdf")
        generate_academic_table(adeno_sig_temp, out_adeno_sig_csv, out_adeno_sig_pdf)
        if os.path.exists(adeno_sig_temp): os.remove(adeno_sig_temp)

        # Neuro Plot & Tables
        out_neuro_plot = os.path.join(script_dir, "Volcano_Neuro.pdf")
        out_neuro_table_csv = os.path.join(script_dir, "Table_Neuro_IVW.csv")
        out_neuro_table_pdf = os.path.join(script_dir, "Table_Neuro_IVW.pdf")
        generate_volcano_plot(neuro_temp, "Colorectal Neuroendocrine", out_neuro_plot)
        generate_academic_table(neuro_temp, out_neuro_table_csv, out_neuro_table_pdf)
        
        # Neuro Significant Table Only (p < 0.05)
        df_neuro_sig = df_neuro[df_neuro['pval'] < 0.05].copy()
        neuro_sig_temp = os.path.join(script_dir, "temp_neuro_sig.csv")
        df_neuro_sig.to_csv(neuro_sig_temp, index=False)
        out_neuro_sig_csv = os.path.join(script_dir, "Table_Neuro_Significant.csv")
        out_neuro_sig_pdf = os.path.join(script_dir, "Table_Neuro_Significant.pdf")
        generate_academic_table(neuro_sig_temp, out_neuro_sig_csv, out_neuro_sig_pdf)
        if os.path.exists(neuro_sig_temp): os.remove(neuro_sig_temp)
        
        # Clean up temp files
        if os.path.exists(adeno_temp): os.remove(adeno_temp)
        if os.path.exists(neuro_temp): os.remove(neuro_temp)

    if os.path.exists(diff_results_file):
        df_diff = pd.read_csv(diff_results_file)
        # Filter foreground hits (p_diff < 0.05)
        df_diff_fg = df_diff[df_diff['pval_diff'] < 0.05].copy()
        diff_fg_temp = os.path.join(script_dir, "temp_diff_fg.csv")
        df_diff_fg.to_csv(diff_fg_temp, index=False)

        out_diff = os.path.join(script_dir, "Volcano_Difference.pdf")
        generate_diff_volcano_plot(diff_results_file, diff_fg_temp, out_diff)

        out_diff_table_csv = os.path.join(script_dir, "Table_Difference_IVW.csv")
        out_diff_table_pdf = os.path.join(script_dir, "Table_Difference_IVW.pdf")
        generate_diff_academic_table(diff_fg_temp, out_diff_table_csv, out_diff_table_pdf)

        if os.path.exists(diff_fg_temp): os.remove(diff_fg_temp)

