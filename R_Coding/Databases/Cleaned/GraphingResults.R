library(dplyr)

# Demonstrating the Results

### Results from MR analysis 
raw_MR <- read_csv("~/Documents/LINA PROJECT/R_Coding/Databases/MR_results_multi_exposure.csv")
#Individual results
# Adeno_MR_results <- MR_Multi_exposure_analysis_results[MR_Multi_exposure_analysis_results$outcome == "adeno",]
# Neuro_MR_results <- MR_Multi_exposure_analysis_results[MR_Multi_exposure_analysis_results$outcome == "neuro",]

# Filtered
filtered_MR_adeno <-read_csv("~/Documents/LINA PROJECT/R_Coding/Databases/MR_results_adeno_filtered.csv") 
filtered_MR_neuro <- read_csv("~/Documents/LINA PROJECT/R_Coding/Databases/MR_results_neuro_filtered.csv")
  
# Raw Difference between the two (Z-tested)
raw_MR_comparison <- read_csv("~/Documents/LINA PROJECT/R_Coding/Databases/MR_difference_results.csv")
# Filtered for significance (P< 0.05 and consistent directional difference)
filtered_MR_comparison <- read_csv("~/Documents/LINA PROJECT/R_Coding/Databases/MR_difference_filtered_consistent.csv")

filtered_



# Table 1: Table 1: Significant SNPs related to exposures


# Table 2: consistent results across different methods and IVW p < 0.05
# adeno: 
filtered_MR_adenoDisplay <- filtered_MR_adeno[filtered_MR_adeno$method == "Inverse-Variance Weighted",]

filtered_MR_adenoDisplay_sig <- filtered_MR_adenoDisplay %>%
  arrange(pval)

filtered_MR_adenoDisplay_corr <- filtered_MR_adenoDisplay %>%
  arrange(desc(b))




# neuro 
filtered_MR_neuroDisplay <- filtered_MR_neuro[filtered_MR_neuro$method == "Inverse-Variance Weighted",]

filtered_MR_neuroDisplay_sig <- filtered_MR_neuroDisplay %>%
  arrange(pval)

filtered_MR_neuroDisplay_corr <- filtered_MR_neuroDisplay %>%
  arrange(desc(b))

# Table 3: Filtered comparison 

filtered_MR_comparisonDisplay <- filtered_MR_comparison[filtered_MR_comparison$method == "Inverse-Variance Weighted",]

filtered_MR_comparisonDisplay_sig <- filtered_MR_comparisonDisplay %>%
  arrange(pval_diff)

filtered_MR_comparisonDisplay_corr <- filtered_MR_comparisonDisplay %>%
  arrange(desc(b_diff))












