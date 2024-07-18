# %% Imports & Constants
import os
import numpy as np
from args import (dDir, Nasr_Numerosity, nFeatures, Models, Layer, nImg)
from utils import (find_Number_Selective_Features)

nNumerosity = len(Nasr_Numerosity)

fDir = os.path.join(dDir, 'CNN_Representations', 'Dot_Patterns_Dataset')
sDir = os.path.join(dDir, 'Decoding_Results', 'Dot_Patterns_Dataset')

# %% Find the Number Selective Units using two-way ANOVA as Nasr
for Model in Models:

    Features = np.zeros((nNumerosity, 3, nImg, nFeatures))
    for i in range(nImg):
        for n in range(nNumerosity):

            standard_set_Path = os.path.join(fDir, f'{Model}_{Layer}_Standard_Set_N-{Nasr_Numerosity[n]}_Stim-{1 + i}.npy')
            control_set1_Path = os.path.join(fDir, f'{Model}_{Layer}_Control-1_Set_N-{Nasr_Numerosity[n]}_Stim-{1 + i}.npy')
            control_set2_Path = os.path.join(fDir, f'{Model}_{Layer}_Control-2_Set_N-{Nasr_Numerosity[n]}_Stim-{1 + i}.npy')
            
            Features[n, 0, i, :] = np.load(standard_set_Path).copy()
            Features[n, 1, i, :] = np.load(control_set1_Path).copy()
            Features[n, 2, i, :] = np.load(control_set2_Path).copy()

    ## Removing Features that are non-activated across all stimuli
    overall_activity = np.sum(Features, axis=(0,1,2))
    activated_Features = Features[:,:,:, ~(overall_activity == 0)]
    activated_units_cumsum = np.cumsum(overall_activity.astype(bool)); nActivatedUnits = overall_activity.astype(bool).sum()
    activated_units_idx = np.array([np.where(activated_units_cumsum == idx)[0][0] for idx in range(1, nActivatedUnits+1)])

    significance_units, non_significant_units, number_pvalues, significative_of_sets, sets_pvalues = find_Number_Selective_Features(activated_Features, alpha=0.01)
    significant_units_cumsum = np.cumsum(significance_units); nSignificantUnits = significance_units.sum()
    sigificant_units_idx = np.array([np.where(significant_units_cumsum == idx)[0][0] for idx in range(1, nSignificantUnits+1)])
    non_significant_units_cumsum = np.cumsum(non_significant_units); nNonSignificantUnits = non_significant_units.sum()
    non_sigificant_units_idx = np.array([np.where(non_significant_units_cumsum == idx)[0][0] for idx in range(1, nNonSignificantUnits+1)])
    significative_of_sets_cumsum = np.cumsum(significative_of_sets); nSignificantofSetsUnits = significative_of_sets.sum()
    significative_of_sets_idx = np.array([np.where(significative_of_sets_cumsum == idx)[0][0] for idx in range(1, nSignificantofSetsUnits+1)])

    # ## Saving uncorrected units in their original indexation that are significant according to the ANOVA test
    np.save(os.path.join(sDir, f'{Model}_{Layer}_Nars_Number_Selective_Units_Uncorrected_Reproduction_nImgs-{nImg}.npy'), activated_units_idx[sigificant_units_idx])
    np.save(os.path.join(sDir, f'{Model}_{Layer}_Nars_Number_Non_Selective_Units_Uncorrected_Reproduction_nImgs-{nImg}.npy'), activated_units_idx[non_sigificant_units_idx])
    np.save(os.path.join(sDir, f'{Model}_{Layer}_Nars_Significative_of_Sets_Units_Uncorrected_Reproduction_nImgs-{nImg}.npy'), activated_units_idx[significative_of_sets_idx])

    ## Saving all Features by croissant ordered of pvalues - ie. by number selectivity
    non_activated_units_idx = np.array([idx for idx in range(nFeatures) if idx not in activated_units_idx])
    activated_units_idx_pvalues = [(activated_units_idx[i],number_pvalues[i]) for i in range(nActivatedUnits)]
    activated_units_idx_pvalues.sort(key=lambda x:x[1])
    ordered_units_idx_by_pvalues = np.concatenate((np.array([activated_units_idx_pvalues[i][0] for i in range(nActivatedUnits)]), non_activated_units_idx))
    np.save(os.path.join(sDir, f'{Model}_{Layer}_Number_Selective_Units_Idx_Nars_Stimuli_nImgs-{nImg}.npy'), ordered_units_idx_by_pvalues)

    ## Keeping only units whose corrected p-value (Bonferroni correction) is lower than 0.05 
    corrected_units_idx = np.where(number_pvalues[significance_units]*nFeatures < 0.05)[0]
    corrected_non_selective_units_idx = np.where(sets_pvalues[non_significant_units]*nFeatures < 0.05)[0]

    ## Going Back to the original Units indexation & saving them
    kept_units_idx = activated_units_idx[sigificant_units_idx[corrected_units_idx]]
    np.save(os.path.join(sDir, f'{Model}_{Layer}_Nars_Number_Selective_Units_Corrected_Reproduction_nImgs-{nImg}.npy'), kept_units_idx)
    kept_units_idx = activated_units_idx[non_sigificant_units_idx[corrected_non_selective_units_idx]]
    np.save(os.path.join(sDir, f'{Model}_{Layer}_Nars_Number_Non_Selective_Units_Corrected_Reproduction_nImgs-{nImg}.npy'), kept_units_idx)

    ## Resulting p-values of the two-way ANOVA
    selective_units = [(number_pvalues[i], significance_units[i]) for i in range(len(number_pvalues))]
    selective_units.sort(key=lambda x:x[0])
    significant_pvalues = (np.array(selective_units)[:,0])[np.array(selective_units)[:,1].astype(bool)]
    np.save(os.path.join(sDir, f'{Model}_{Layer}_Anova_Pvalues_nImgs-{nImg}.npy'), significant_pvalues)