# %% Imports and Constants
from sklearn.linear_model import LinearRegression
import os
import numpy as np
from utils import (_read_ParkSpace_log, _compute_park_space_point)
from args import (mDir, dDir, Backgrounds, Objects, PS_Ranges, Versions,
                  Training_Modes, Models, Layers, Model_Names,
                  Low_Level_Statistics, Numerosity, Spacing, Size_Area)

tDir = os.path.join(dDir, 'Decoding_Results', 'Photorealistic_Dataset')
wDir = os.path.join(mDir, 'derivatives',  'Low_Level_Statistics',  'Local_Contrast_Measures')
fDir = os.path.join(mDir, 'derivatives',  'Low_Level_Statistics',  'Frequency_Measures')
lDir = os.path.join(mDir, 'derivatives',  'Low_Level_Statistics',  'Luminance_Measures')


nBackgrounds, nObjects = len(Backgrounds), len(Objects)

## Modality that we will decode from each stimulus
Modality = 'N'

# %% Useful Methods
def _distance_to_axis_function(ps_range):
    pDir = os.path.join(wDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')

    Beta, Gamma = [], []
    for bg_idx, bg_alpha in Backgrounds:
        bg_name = f'bg-{bg_idx}_alpha{bg_alpha}'
        for obj_name in Objects:
            Beta.append(np.load(os.path.join(pDir, f'Sigma-12_{obj_name}_{bg_name}_Beta.npy')).flatten())
            Gamma.append(np.load(os.path.join(pDir, f'Sigma-12_{obj_name}_{bg_name}_Gamma.npy')).flatten())
    Beta = np.concatenate(Beta)
    Gamma = np.concatenate(Gamma)

    M = np.zeros([len(Beta), 2])
    M[:,0] = Gamma.copy()
    M[:,1] = Beta.copy()
    mB, mG = np.mean(Beta), np.mean(Gamma)
    c = np.array([mG, mB])

    u, s, vh = np.linalg.svd(M, full_matrices=True)
    wC, wT = np.array(vh[0,:]), np.array(vh[1,:]) 
    norm_wC, norm_wT = np.sqrt(np.sum(wC**2)), np.sqrt(np.sum(wT**2))
    dC = lambda p:np.dot(wC, p-c)/norm_wC
    dT = lambda p:np.dot(wT, p-c)/norm_wT
    return dT, dC

def _load_features(ParkSpace_Description, statistic, ps_range, object_name, bg_name, v_idx, dT, dC, UseSegMask):
    '''
        Load the features row associated with the low-level statistic of interest for a given stimuli
    '''

    x = []

    if statistic == 'Mean_Lum':

        uDir = os.path.join(lDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')
        result_path = os.path.join(uDir, f'Mean_Lum{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy')
        Mean_Lum = np.load(result_path)

        for N, ID, FD in ParkSpace_Description:

            Sp, SzA = _compute_park_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = Numerosity[ps_range].index(N), Spacing[ps_range].index(Sp), Size_Area[ps_range].index(SzA)
            x.append(Mean_Lum[idx_N, idx_Sp, idx_SzA])

    elif statistic == 'Std_Lum':

        uDir = os.path.join(lDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')
        result_path = os.path.join(uDir, f'Std_Lum{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy')
        Std_Lum = np.load(result_path)

        for N, ID, FD in ParkSpace_Description:

            Sp, SzA = _compute_park_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = Numerosity[ps_range].index(N), Spacing[ps_range].index(Sp), Size_Area[ps_range].index(SzA)
            x.append(Std_Lum[idx_N, idx_Sp, idx_SzA])

    elif statistic == 'Energy_Low_SF':

        uDir = os.path.join(fDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')
        result_path = os.path.join(uDir, f'Energy_Low_SF{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy')
        NRJ_Low_SF = np.load(result_path)

        for N, ID, FD in ParkSpace_Description:

            Sp, SzA = _compute_park_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = Numerosity[ps_range].index(N), Spacing[ps_range].index(Sp), Size_Area[ps_range].index(SzA)
            x.append(NRJ_Low_SF[idx_N, idx_Sp, idx_SzA])

    elif statistic == 'Energy_High_SF':

        uDir = os.path.join(fDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')
        result_path = os.path.join(uDir, f'Energy_High_SF{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy')
        NRJ_High_SF = np.load(result_path)

        for N, ID, FD in ParkSpace_Description:

            Sp, SzA = _compute_park_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = Numerosity[ps_range].index(N), Spacing[ps_range].index(Sp), Size_Area[ps_range].index(SzA)
            x.append(NRJ_High_SF[idx_N, idx_Sp, idx_SzA])

    elif statistic == 'Dist_Texture':

        uDir = os.path.join(wDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')
        result_path = os.path.join(uDir, f'Sigma-12_{object_name}_{bg_name}')
        Gamma, Beta = np.load(result_path + '_Gamma.npy'), np.load(result_path + '_Beta.npy')

        for N, ID, FD in ParkSpace_Description:

            Sp, SzA = _compute_park_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = Numerosity[ps_range].index(N), Spacing[ps_range].index(Sp), Size_Area[ps_range].index(SzA)
            p = np.array([Gamma[idx_N, idx_Sp, idx_SzA], Beta[idx_N, idx_Sp, idx_SzA]])
            x.append(dT(p))
                
    elif statistic == 'Dist_Complexity':

        uDir = os.path.join(wDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')
        result_path = os.path.join(uDir, f'Sigma-12_{object_name}_{bg_name}')
        Gamma, Beta = np.load(result_path + '_Gamma.npy'), np.load(result_path + '_Beta.npy')

        for N, ID, FD in ParkSpace_Description:

            Sp, SzA = _compute_park_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = Numerosity[ps_range].index(N), Spacing[ps_range].index(Sp), Size_Area[ps_range].index(SzA)
            p = np.array([Gamma[idx_N, idx_Sp, idx_SzA], Beta[idx_N, idx_Sp, idx_SzA]])
            x.append(dC(p))

    elif statistic == 'Agg_Mag_Fourier':

        uDir = os.path.join(fDir, f'PS_{ps_range[0].upper() + ps_range[1:]}_Range')
        result_path = os.path.join(uDir, f'Aggregate_rAvg_Fourier_Magnitude{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy')
        Agg_Mag = np.load(result_path)

        for N, ID, FD in ParkSpace_Description:

            Sp, SzA = _compute_park_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = Numerosity[ps_range].index(N), Spacing[ps_range].index(Sp), Size_Area[ps_range].index(SzA)
            x.append(Agg_Mag[idx_N, idx_Sp, idx_SzA])

    else:
        raise NotImplementedError(f'{statistic} Statistic not currently used.')

    return x

def _load_stimuli_statistics(ps_range, statistics, v_idx, object_name=None, bg_name=None, dT=None, dC=None, UseSegMask=''):

    assert ps_range in ['subitizing', 'estimation'], 'range should be either "subitizing" or "estimation"'
    assert (object_name is not None and bg_name is None) or (object_name is None and bg_name is not None), 'at least, and only one, object_name or bg_name should be provided.'
    assert UseSegMask in ['', '_Mask'], "UseSegMask as '' for use of the PhotoRealistic Stimuli & as '_Mask' for associated Segmentation Masks"

    PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{ps_range}_range.csv')   
    ParkSpace_Description = _read_ParkSpace_log(PS_path)

    X = []
    for statistic in statistics:

        features = []
        ## Fine Grained Generalisation Across Objects
        if object_name is not None:
            for bg_idx, bg_alpha in Backgrounds:
                features.append(_load_features(ParkSpace_Description, statistic, ps_range, object_name, f'bg-{bg_idx}_alpha{bg_alpha}', v_idx, dT, dC, UseSegMask))
        ## Fine Grained Generalisation Across Backgrounds
        else:
            for object_name in Objects:
                features.append(_load_features(ParkSpace_Description, statistic, ps_range, object_name, bg_name, v_idx, dT, dC, UseSegMask))
        
        X.append(np.concatenate(features).copy())
        
    return X

def _standardize(X):

    ## standardizing input to verify the Product Measures (Variance Decomposition) Hypotheses
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return X

# %% Assess the explained variance accounted by linear mapping of the low-level statistics on their associated fine-grained predictions

UseSegMask = '' # Just Remove (or replace '_Mask' by '') this Arg for using the photorealistic Stimuli (rather than their binary Masks)

clf = LinearRegression()

for PS_range in PS_Ranges:

    sDir = os.path.join(tDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

    dT, dC = _distance_to_axis_function(PS_range)

    #### Generalisation Across Objects (Collasped Backgrounds Dimension) :

    ## Loading Statistics of the Stimuli used to obtain Predictions (Shared for every Training - once Concatenated)
    X_test = []
    for j in range(nObjects):
        for v_idx in Versions: ## Careful X_test was corresponding to "3-v_idx"

            objTest = Objects[j]
            Xs = _load_stimuli_statistics(PS_range, Low_Level_Statistics, 3 - v_idx, object_name=objTest, bg_name=None, dT=dT, dC=dC, UseSegMask=UseSegMask)
            
            ## Xs : array of size (1280, nStatistics) 
            X_test.append(np.stack(Xs, axis=1)) 
    
    ## Reformat the input X_test as a array of size (20 * 2 * 1280, nStatistics) & Standardize it
    X_test = _standardize(np.vstack(X_test))

    for Model_Name in Models:
        for Mode in Training_Modes:

            Model = Model_Names[Mode][Model_Name]

            for Layer in Layers:

                results_path = os.path.join(sDir, f'FGG_OLS_{Model}_{Layer}_Decoding_Log_{Modality}') if Model != 'RawPixels' else os.path.join(sDir, f'FGG_OLS_{Model}_Decoding_Log_{Modality}')
                predictions_path = os.path.join(sDir, f'Finer_Grain_Generalisation{UseSegMask}_{Model}_{Layer}_Decoding_Log_{Modality}') if Model != 'RawPixels' else os.path.join(sDir, f'Finer_Grain_Generalisation{UseSegMask}_{Model}_Decoding_Log_{Modality}')
                save_path = results_path + f'_Across_Objects_LLS_Explained_Variance{UseSegMask}.npy'

                if not os.path.isfile(save_path):

                    Explained_Variance = np.zeros([nObjects])
                    predictions = np.load(predictions_path + '_Predictions_Across_Objects.npy') # shape : (20, 20, 2560) i.e. pred[i, j, :1280] correspond to the predictions of the 2nd version of the stimuli (since X_test on 3-v_idx !)

                    ## Loading & Fitting the Predictions (y_test_pred) for each Training (X_train may use different confounders) independantly. 
                    for i in range(nObjects):

                        y_test_pred = predictions[i, :, :].flatten() # Careful: pred[i*1280:(i+1)*1280] correspond to the "3-v_idx" stimuli whereas pred[(i+1)*1280:(i+2)*1280] correspond to the "v_idx"
                        ## centering the target in order to verify the hypothesis of the product measure (variance decompositon)
                        y_test_pred -= y_test_pred.mean()
                        y_test_pred = y_test_pred.reshape(-1, 1)

                        ## Fitting the BRR between the Predictions (y_test_pred) and the Statistics of the associated Stimuli (X_test).
                        clf.fit(X_test, y_test_pred)

                        ## Estimating the Explained Variance of each Low-Level Statistics accounting for the mapping from X_test to y_test_pred
                        Explained_Variance[i] = clf.score(X_test, y_test_pred)

                    np.save(save_path, Explained_Variance)

    #### Generalisation Across Backgrounds (collapsed Objects Dimension) - Only for PhotoRealistic Stimuli
    if UseSegMask == '':

        ## Loading Statistics of the Stimuli used to obtain Predictions (Shared for every Training - once Concatenated)
        X_test = []
        for j in range(nBackgrounds):
            for v_idx in Versions: ## Careful X_test was corresponding to "3-v_idx"

                bg_idx, bg_alpha = Backgrounds[j]
                bgTest = f'bg-{bg_idx}_alpha{bg_alpha}'
                Xs = _load_stimuli_statistics(PS_range, Low_Level_Statistics, 3 - v_idx, object_name=None, bg_name=bgTest, dT=dT, dC=dC, UseSegMask=UseSegMask)
                
                ## Xs : array of size (1280, nStatistics) 
                X_test.append(np.stack(Xs, axis=1)) 
        
        ## Reformat the input X_test as a array of size (20 * 2 * 1280, nStatistics) & Standardize it
        X_test = _standardize(np.vstack(X_test))

        for Model_Name in Models:
            for Mode in Training_Modes:

                Model = Model_Names[Mode][Model_Name]

                for Layer in Layers:

                    results_path = os.path.join(sDir, f'FGG_OLS_{Model}_{Layer}_Decoding_Log_{Modality}') if Model != 'RawPixels' else os.path.join(sDir, f'FGG_OLS_{Model}_Decoding_Log_{Modality}')
                    predictions_path = os.path.join(sDir, f'Finer_Grain_Generalisation{UseSegMask}_{Model}_{Layer}_Decoding_Log_{Modality}') if Model != 'RawPixels' else os.path.join(sDir, f'Finer_Grain_Generalisation{UseSegMask}_{Model}_Decoding_Log_{Modality}')
                    save_path = results_path + f'_Across_Backgrounds_LLS_Explained_Variance{UseSegMask}.npy'

                    if not os.path.isfile(save_path):

                        Explained_Variance = np.zeros([nBackgrounds])
                        predictions = np.load(predictions_path + '_Predictions_Across_Backgrounds.npy') # shape : (20, 20, 2560) i.e. pred[i, j, :1280] correspond to the predictions of the 2nd version of the stimuli (since X_test on 3-v_idx !)

                        ## Loading & Fitting the Predictions (y_test_pred) for each Training (X_train may use different confounders) independantly. 
                        for i in range(nBackgrounds):

                            y_test_pred = predictions[i, :, :].flatten() # Careful: pred[i*1280:(i+1)*1280] correspond to the "3-v_idx" stimuli whereas pred[(i+1)*1280:(i+2)*1280] correspond to the "v_idx"
                            ## centering the target in order to verify the hypothesis of the product measure (variance decompositon)
                            y_test_pred -= y_test_pred.mean()
                            y_test_pred = y_test_pred.reshape(-1, 1)

                            ## Fitting the BRR between the Predictions (y_test_pred) and the Statistics of the associated Stimuli (X_test).
                            clf.fit(X_test, y_test_pred)

                            ## Estimating the Explained Variance of each Low-Level Statistics accounting for the mapping from X_test to y_test_pred
                            Explained_Variance[i] = clf.score(X_test, y_test_pred)

                        np.save(save_path, Explained_Variance)