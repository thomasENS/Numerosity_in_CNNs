# %% Imports
import numpy as np
from scipy.stats import f

# %% Useful Methods to reproduce Nasr's artificial units selection
def anova2(data, alpha):
    """
        Performs a two-way ANOVA on the given 3D Numpy Array of shape (n_fA, n_fB, n_samples)
        and assesses the significance of the p-values obtained for each factor A, B as well as 
        their interaction AB.

        - Remark :
            - This implementation considers the balanced case of the ANOVA i.e. same number n_samples
            of samples for every condition Yij.
            - We followed these formula : https://statweb.stanford.edu/~jtaylo/courses/stats203/notes/anova.fixed.pdf

        - Parameters :
            - data [np.array] : n_fX is the number of distinct level of factor X
            - alpha [float] : significance level to compare the p-values obtained from the F-test against.

    """

    n_fA, n_fB, n_samples = data.shape
    
    # Calculate the means
    grand_mean = np.mean(data)
    fA_mean = np.mean(data, axis=(1,2), keepdims=True) # average across the samples & factor B levels
    fB_mean = np.mean(data, axis=(0,2), keepdims=True) # average across the samples & factor A levels
    fAB_mean = np.mean(data, axis=2, keepdims=True)    # average across the samples

    # Calculate the sum of squares
    SST = np.sum((data - grand_mean) ** 2)
    SSA = n_samples * n_fB * np.sum((fA_mean - grand_mean) ** 2)
    SSB = n_samples * n_fA * np.sum((fB_mean - grand_mean) ** 2)
    SSAB = n_samples * np.sum((fAB_mean - fA_mean - fB_mean + grand_mean) ** 2)
    SSE = SST - SSA - SSB - SSAB

    # Calculate the degrees of freedom
    df_a = n_fA - 1
    df_b = n_fB - 1
    df_ab = df_a * df_b
    df_error = (n_samples - 1) * n_fA * n_fB
    
    # Calculate the mean squares
    MSA = SSA / df_a
    MSB = SSB / df_b
    MSAB = SSAB / df_ab
    MSE = SSE / df_error
    
    # Calculate the F-statistics and p-values
    f_a = MSA / MSE
    p_a = 1 - f.cdf(f_a, df_a, df_error)
    f_b = MSB / MSE
    p_b = 1 - f.cdf(f_b, df_b, df_error)
    f_ab = MSAB / MSE
    p_ab = 1 - f.cdf(f_ab, df_ab, df_error)
    
    return p_a, p_a < alpha, p_b, p_b < alpha, p_ab, p_ab < alpha

def find_Number_Selective_Features(features, alpha=0.01):
    '''
        Find the Number Selective Features following Nieder Approach upon Nars Dataset of Dot-Patterns Stimuli.

        - Remark : 
            - Nars Dataset is composed of 3 Control Sets of Stimuli (StandardSet, ControlSet1, ControlSet2)
            which equates some distinct non-numerical properties of the Numerosity stimulus.
            - A feature is considered to be Number Selective if it exhibited a significant change 
            for numerosity (P < alpha) but no significant change for the stimulus control
            set or interaction between two factors.

        - Parameters : 
            - features [np.array] : Representation of Nars stimuli dataset by the HCNN model. It should
            be of shape (n_Numerosity, n_Sets, n_samples, n_Features) i.e. Factor A is Numerosity & Factor B is Set.
            - alpha [float] : significance level to compare the p-values obtained from the F-test against.

    '''
    
    number_selective_features, number_non_selective_features, number_pvalues, significative_of_sets, sets_pvalues = [], [], [], [], []
    for k in range(features.shape[3]):

        pNumber, sigNumber, pSets, sigSets, pInteraction, sigInteraction = anova2(features[:,:,:,k], alpha)
        number_selective_features.append(sigNumber and (not sigSets) and (not sigInteraction))
        number_pvalues.append(pNumber)
        number_non_selective_features.append((not sigNumber) and sigSets and (not sigInteraction))
        sets_pvalues.append(pSets)
        significative_of_sets.append(sigSets)
        
        
    return np.array(number_selective_features), np.array(number_non_selective_features), np.array(number_pvalues), np.array(significative_of_sets), np.array(sets_pvalues)
