# %% Imports and Constants
from sklearn.linear_model import RidgeCV
import os
import numpy as np
from utils import (_read_ParkSpace_log, _load_labels, _compute_park_space_point)
from args import (mDir, dDir, PS_Ranges, Versions, Alphas, Training_Modes, Models,
                  Layers, Model_Names, SubSpace, subSpaceTrain, subSpaceTest,
                  Stimulus_Types)

nLayers = len(Layers)

clf = RidgeCV(alphas=Alphas)

# Controlled Park Space Description for Congruent / Incongruent Non-Numerical Parameters (N, Sp, SzA)
Yellow_Set = {'estimation':[(6, 11.3, 8.3), (10, 11.1, 8.5), (15, 11.,  8.7), (24, 10.7, 8.9)],
              'subitizing':[(1, 12.1, 9.1), (2, 12., 9.4), (3, 11.8, 9.5), (4, 11.5, 9.7)]}
Orange_Set = {'estimation':[(6, 10.7, 8.9), (10, 11.,  8.7), (15, 11.1, 8.5), (24, 11.3, 8.3)],
              'subitizing':[(1, 11.5, 9.7), (2, 11.8, 9.5), (3, 12.,  9.4), (4, 12.1, 9.1)]}

# Yellow_Set['estimation'] corresponds to (N, ID, FD) in [(6, 75, 604), (10, 75, 610), (15, 75, 610), (24, 75, 604)]
# Orange_Set['estimation'] corresponds to (N, ID, FD) in [(6, 106, 427), (10, 83, 551), (15, 68, 675), (24, 53, 855)]

# Yellow_Set['subitizing'] corresponds to (N, ID, FD) in [(1, 184, 604), (2, 184, 669), (3, 184, 669), (4, 184, 604)]
# Orange_Set['subitizing'] corresponds to (N, ID, FD) in [(1, 260, 427), (2, 203, 604), (3, 166, 740), (4, 130, 855)]

# %% Useful Methods
def _load_features(features_path, model, layer, modality, sub_space_name):
    '''
        Load the features row associated with the features_path, applying features selection to reduce
        the dimension if necessary i.e. for Conv1, Conv2 and Conv3.
    '''
    if (model == 'RawPixels') or (model in ['AlexNet', 'Random_AlexNet'] and layer in ['Conv4', 'Conv5']):
            return np.load(features_path)
    else:
        selectedFeaturesIdx = np.load(
            os.path.join(iDir, f'{model}_{layer}_subSpace-{sub_space_name}_{modality}_selected_features_idx.npy')
        )
        return np.load(features_path)[selectedFeaturesIdx]

def _load_controlled_non_numerical_parameters_stimuli_dataset(model, layer, ps_range, load_space_idx, selected_features_space_idx, controlling_condition_set, _mask='', target_scale='Log'):
    '''
        Load all the representations extracted from a Network's [layer] of the stimuli pasted on the given [background] grouped by Object used to create the stimuli.
    '''

    assert ps_range in ['subitizing', 'estimation'], 'range should be either "subitizing" or "estimation"'
    assert layer in ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'], 'Network layers are ConvX with X in {1 ... 5}'

    PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{ps_range}_range.csv')
    ParkSpace_Description = _read_ParkSpace_log(PS_path)
    Backgrounds, Objects = SubSpace[load_space_idx]

    X, y = [], []
    for object_name in Objects:
        for bg_idx, bg_alpha in Backgrounds:

            for N, ID, FD in ParkSpace_Description:

                Sp, SzA = _compute_park_space_point(N, ID, FD)
                if (N, Sp, SzA) in controlling_condition_set:

                    for v_idx in Versions:

                        if model != 'RawPixels':
                            features_path = os.path.join(fDir, model, layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                        else:
                            features_path = os.path.join(fDir, 'RawPixels', object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'RawPixels{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')

                        if os.path.isfile(features_path): 
                            X.append(_load_features(features_path, model, layer, 'N', selected_features_space_idx))
                            y.append(_load_labels(N, ID, FD, 'N', target_scale))

    return np.array(X), np.array(y) - np.array(y).mean()

MAE = lambda y_pred, y_true: np.mean(np.abs(y_true - y_pred))

# %% Ridge Decoding of Numerosity while Controlling the Park Space to decorrelate N from Non-Numerical Parameters !
TargetScale = 'Log' # 'Log' for log(N) as target (or replace by '' for y=N directly)

for PS_range in PS_Ranges:

    iDir = os.path.join(dDir, 'Features_Selection')
    rDir = os.path.join(dDir, 'CNN_Representations', 'Photorealistic_Dataset')
    lDir = os.path.join(dDir, 'Decoding_Results', 'Photorealistic_Dataset')
    fDir = os.path.join(rDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')
    iDir = os.path.join(iDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')
    lDir = os.path.join(lDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

    Controlling_Set = {0:Yellow_Set[PS_range], 1:Orange_Set[PS_range]}; Control_Condition = [0, 1]

    for UseSegMask in Stimulus_Types:

        for Model_Name in Models:
            for Mode in Training_Modes:

                Model = Model_Names[Mode][Model_Name]

                results_path = os.path.join(lDir, f'Full_Generalisation_Controlled_Non_Numerical_Params{UseSegMask}_{Model}_across_Hierarchy_Decoding_{TargetScale}_N')
                score_MAE, score_Std = np.zeros([nLayers]), np.zeros([nLayers])

                if not os.path.isfile(results_path + '_Score_MAE.npy'):

                    ## Perform the Predictions Analysis across the Hierarchy
                    for i in range(nLayers):

                        ## Controlling (IA, TA, FA, Spar) to ensure that they are decorrelated from N (i.e. constant/varying or congruent/incongruent between train/test)
                        MAE_InterCV = []
                        for control in Control_Condition:
                        
                            ## Compute Predictions for the Generalisation Pattern - Either Full either Partial and Average the Errors across the cases.
                            for sTrain in subSpaceTrain:

                                X_train, y_train = _load_controlled_non_numerical_parameters_stimuli_dataset(Model, Layers[i], PS_range, sTrain, sTrain, Controlling_Set[control], _mask=UseSegMask, target_scale=TargetScale)

                                ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                                clf.fit(X_train, y_train)

                                ## Compute Outer Loop Ridge Regression MAE score
                                for sTest in subSpaceTest[sTrain]:
                                    X_test, y_test = _load_controlled_non_numerical_parameters_stimuli_dataset(Model, Layers[i], PS_range, sTest, sTrain, Controlling_Set[1-control], _mask=UseSegMask, target_scale=TargetScale)
                                    MAE_InterCV.append(MAE(clf.predict(X_test), y_test))

                        score_MAE[i] = np.mean(MAE_InterCV)
                        score_Std[i] = np.std(MAE_InterCV)

                    np.save(results_path + '_Score_MAE.npy', score_MAE)
                    np.save(results_path + '_Score_Std.npy', score_Std)