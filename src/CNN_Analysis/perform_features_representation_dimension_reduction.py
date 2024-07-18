# %% Imports and Constants
from sklearn.feature_selection import f_classif
import os
import numpy as np
from utils import (_read_ParkSpace_log, _load_labels)
from args import (mDir, dDir, Versions, Naturals, Artificials, Animals, Tools, PS_Ranges,
                  Training_Modes, Models, Layers, Model_Names, Stimulus_Types,
                  Modalities, nFeatures, SubSpace)

iDir = os.path.join(dDir, 'Features_Selection')
rDir = os.path.join(dDir, 'CNN_Representations', 'Photorealistic_Dataset')

# %% Useful Methods to load all the stimulus dataset representation for all generalization scheme
def _load_stimuli_features(model, layer, ps_range, load_space_idx, v_idx=1, _mask=''):
    '''
        Load all the representations extracted from a Network's [layer] of the stimuli pasted on the given [background] grouped by Object used to create the stimuli.
    '''

    assert ps_range in ['subitizing', 'estimation'], 'range should be either "subitizing" or "estimation"'
    assert layer in ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'], 'Network layers are ConvX with X in {1 ... 5}'

    PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{ps_range}_range.csv')
    ParkSpace_Description = _read_ParkSpace_log(PS_path)
    Backgrounds, Objects = SubSpace[load_space_idx]

    X = []
    for i in range(len(Objects)):
        for j in range(len(Backgrounds)):

            object_name = Objects[i]
            bg_idx, bg_alpha = Backgrounds[j]

            for N, ID, FD in ParkSpace_Description:
                if model != 'RawPixels':
                    features_path = os.path.join(fDir, model, layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                else:
                    features_path = os.path.join(fDir, 'RawPixels', object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'RawPixels{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')

                if os.path.isfile(features_path): 
                    X.append(np.load(features_path))

    return np.array(X)

def _load_stimuli_labels(model, layer, ps_range, modality, load_space_idx, v_idx=1, _mask=''):

    assert ps_range in ['subitizing', 'estimation'], 'range should be either "subitizing" or "estimation"'
    assert layer in ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'], 'Network layers are ConvX with X in {1 ... 5}'

    PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{ps_range}_range.csv')
    ParkSpace_Description = _read_ParkSpace_log(PS_path)
    Backgrounds, Objects = SubSpace[load_space_idx]

    y = []
    for i in range(len(Objects)):
        for j in range(len(Backgrounds)):

            object_name = Objects[i]
            bg_idx, bg_alpha = Backgrounds[j]

            for N, ID, FD in ParkSpace_Description:
                if model != 'RawPixels':
                    features_path = os.path.join(fDir, model, layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                else:
                    features_path = os.path.join(fDir, 'RawPixels', object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'RawPixels{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')

                if os.path.isfile(features_path): 
                    y.append(_load_labels(N, ID, FD, ps_range, modality))

    return np.array(y) - np.array(y).mean()

def _load_stimuli_object_dataset(model, layer, ps_range, modality, object_name, _mask=''):
    '''
        Load the representations extracted from a Network's [layer] of the stimuli that have a given [object]
    '''

    assert ps_range in ['subitizing', 'estimation'], 'range should be either "subitizing" or "estimation"'
    assert layer in ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'], 'Network layers are ConvX with X in {1 ... 5}'

    PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{ps_range}_range.csv')
    ParkSpace_Description = _read_ParkSpace_log(PS_path)

    X, y = [], []
    for bg_idx, bg_alpha in Naturals + Artificials:

        for v_idx in Versions:
            for N, ID, FD in ParkSpace_Description:
                if model != 'RawPixels':
                    features_path = os.path.join(fDir, model, layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                else:
                    features_path = os.path.join(fDir, 'RawPixels', object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'RawPixels{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')

                if os.path.isfile(features_path): 
                    X.append(np.load(features_path))
                    y.append(_load_labels(N, ID, FD, ps_range, modality))

    return np.array(X), np.array(y) - np.array(y).mean()

def _load_stimuli_background_dataset(model, layer, ps_range, modality, bg_name, _mask=''):
    '''
        Load all the representations extracted from a Network's [layer] of the stimuli pasted on the given [background].
    '''

    assert ps_range in ['subitizing', 'estimation'], 'range should be either "subitizing" or "estimation"'
    assert layer in ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5'], 'Network layers are ConvX with X in {1 ... 5}'

    PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{ps_range}_range.csv')
    ParkSpace_Description = _read_ParkSpace_log(PS_path)

    bg_idx, bg_alpha = int(bg_name.split('bg-')[-1].split('_')[0]), int(bg_name.split('alpha')[-1])

    X, y = [], []
    for object_name in Animals + Tools:

        for v_idx in Versions:
            for N, ID, FD in ParkSpace_Description:
                if model != 'RawPixels':
                    features_path = os.path.join(fDir, model, layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                else:
                    features_path = os.path.join(fDir, 'RawPixels', object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'RawPixels{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')

                if os.path.isfile(features_path): 
                    X.append(np.load(features_path))
                    y.append(_load_labels(N, ID, FD, ps_range, modality))

    return np.array(X), np.array(y) - np.array(y).mean()

# %% ANOVA Feature Selections for the Coarse-Grained Generalization
for PS_range in PS_Ranges:
 
    fDir = os.path.join(rDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')
    iDir = os.path.join(iDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

    for UseSegMask in Stimulus_Types:

        for Mode in Training_Modes:
        
            for Model in Models:

                for Layer in Layers:

                    for sub_space_idx in [1, 2, 3, 4]:

                        if not os.path.isfile(os.path.join(iDir, f'{Model_Names[Mode][Model]}_{Layer}{UseSegMask}_subSpace-{sub_space_idx}_N_selected_features_idx.npy')): ## Do not compute Features Idx that were already computed
                            
                            print(Model_Names[Mode][Model], Layer, PS_range, sub_space_idx)
                            X = _load_stimuli_features(Model_Names[Mode][Model], Layer, PS_range, sub_space_idx, _mask=UseSegMask)
                            
                            for modality in Modalities:
                                
                                idx_path = os.path.join(iDir, f'{Model_Names[Mode][Model]}_{Layer}{UseSegMask}_subSpace-{sub_space_idx}_{modality}_selected_features_idx.npy')

                                if not os.path.isfile(idx_path): ## Do not compute Features Idx that were already computed

                                    y = _load_stimuli_labels(Model_Names[Mode][Model], Layer, PS_range, modality, sub_space_idx, _mask=UseSegMask)

                                    f_stat, p_value = f_classif(X,y)
                                    f_stat /= np.max(f_stat[~np.isnan(f_stat)])

                                    sorted_pValue_Idx = np.argsort(-f_stat) # NaN values are put at the end, so, not causing any issue as long as there is more non-NaN than nFeatures
                                    selected_features_Idx = sorted_pValue_Idx[:nFeatures]

                                    if Model == 'alexnet' and Layer in ['Conv4', 'Conv5']:
                                        np.save(idx_path, selected_features_Idx)
                                    else:
                                        nNaN = len(p_value[np.isnan(p_value)])
                                        if len(p_value) - nNaN < nFeatures:
                                            print(PS_range, Model, Layer, sub_space_idx, modality, f'Careful, {nNaN} nNaN values relative to {len(p_value) - nFeatures} Features.')
                                        else:
                                            np.save(idx_path, selected_features_Idx)
                            
# %% ANOVA Features Selection for Object/Background Finer Grain Generalisation
for PS_range in PS_Ranges:
 
    fDir = os.path.join(rDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')
    iDir = os.path.join(iDir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

    for Modality in Modalities:

        for UseSegMask in Stimulus_Types:

            for Model in Models:

                for Mode in Training_Modes:

                    for Layer in Layers:

                        ## Features Selection for Finer Grain Generalisatio Across Backgrounds
                        for bg_idx, bg_alpha in Artificials + Naturals:
                            bg_name = f'bg-{bg_idx}_alpha{bg_alpha}'

                            idx_path = os.path.join(iDir, f'{Model_Names[Mode][Model]}_{Layer}{UseSegMask}_subSpace-{bg_name}_{Modality}_selected_features_idx.npy')
                            
                            if not os.path.isfile(idx_path): ## Do not compute Features Idx that were already computed
                                
                                print(Model_Names[Mode][Model], Layer, PS_range, bg_name)

                                X, y = _load_stimuli_background_dataset(Model_Names[Mode][Model], Layer, PS_range, Modality, bg_name, _mask=UseSegMask)

                                f_stat, p_value = f_classif(X,y)
                                f_stat /= np.max(f_stat[~np.isnan(f_stat)])

                                sorted_pValue_Idx = np.argsort(-f_stat) # NaN values are put at the end, so, not causing any issue as long as there is more non-NaN than nFeatures
                                selected_features_Idx = sorted_pValue_Idx[:nFeatures]

                                if Model == 'alexnet' and Layer in ['Conv4', 'Conv5']:
                                    np.save(idx_path, selected_features_Idx)
                                else:
                                    nNaN = len(p_value[np.isnan(p_value)])
                                    if len(p_value) - nNaN < nFeatures:
                                        print(PS_range, Model, Layer, bg_name, Modality, f'Careful, {nNaN} nNaN values relative to {len(p_value) - nFeatures} Features.')
                                    else:
                                        np.save(idx_path, selected_features_Idx)

                        ## Features Selection for Finer Grain Generalisatio Across Objects
                        for obj_name in Animals + Tools:

                            idx_path = os.path.join(iDir, f'{Model_Names[Mode][Model]}_{Layer}{UseSegMask}_subSpace-{obj_name}_{Modality}_selected_features_idx.npy')
                            
                            if not os.path.isfile(idx_path): ## Do not compute Features Idx that were already computed
                                
                                print(Model_Names[Mode][Model], Layer, PS_range, obj_name)

                                X, y = _load_stimuli_object_dataset(Model_Names[Mode][Model], Layer, PS_range, Modality, obj_name, _mask=UseSegMask)

                                f_stat, p_value = f_classif(X,y)
                                f_stat /= np.max(f_stat[~np.isnan(f_stat)])

                                sorted_pValue_Idx = np.argsort(-f_stat) # NaN values are put at the end, so, not causing any issue as long as there is more non-NaN than nFeatures
                                selected_features_Idx = sorted_pValue_Idx[:nFeatures]

                                if Model == 'alexnet' and Layer in ['Conv4', 'Conv5']:
                                    np.save(idx_path, selected_features_Idx)
                                else:
                                    nNaN = len(p_value[np.isnan(p_value)])
                                    if len(p_value) - nNaN < nFeatures:
                                        print(PS_range, Model, Layer, obj_name, Modality, f'Careful, {nNaN} nNaN values relative to {len(p_value) - nFeatures} Features.')
                                    else:
                                        np.save(idx_path, selected_features_Idx)