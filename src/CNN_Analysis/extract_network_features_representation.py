# %% Imports & Constants
import os
import numpy as np
from utils import (_read_ParkSpace_log, TorchHub_FeatureExtractor, Nars_Dataset_Preprocessing,
                   ImageNet_Stimuli_PreProcessing, Mask_Dataset_Preprocessing)
from args import (mDir, dDir, Versions, Backgrounds, Objects, Stimulus_Types,
                  PS_Ranges, Training_Modes, Models, Layers, Model_Names, Nasr_Numerosity)

sDir = os.path.join(mDir, 'data', 'Stimuli')
rDir = os.path.join(dDir, 'CNN_Representations')

# %% Extract Representation for Selected Objects & Backgrounds for the Networks' Layers
preprocess = ImageNet_Stimuli_PreProcessing()

for Model in Models:
    for Mode in Training_Modes:

        FeaturesExtractor = TorchHub_FeatureExtractor(Model, Mode)

        for PS_range in PS_Ranges:
    
            iDir = os.path.join(sDir, 'Photorealistic_Dataset', f'PS_{PS_range[0].upper() + PS_range[1:]}_Range', 'Images')
            fDir = os.path.join(rDir, 'Photorealistic_Dataset', f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

            PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{PS_range}_range.csv')
            ParkSpace_Description = _read_ParkSpace_log(PS_path)

            for object_name in Objects:
                for bg_idx, bg_alpha in Backgrounds:
                    for N, ID, FD in ParkSpace_Description:
                        for v_idx in Versions:

                            features_paths = [os.path.join(fDir, Model_Names[Mode][Model], layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{Model_Names[Mode][Model]}_{layer}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy') for layer in Layers]
                            is_not_files = [not os.path.isfile(features_path) for features_path in features_paths]
                            if np.any(is_not_files): # Do not compute a representation that was already created !

                                img_path = os.path.join(iDir, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.png')

                                if os.path.isfile(img_path): # Do not try to compute representation of an image that does not exist !
                                    img_input = preprocess(img_path)
                                    features  = FeaturesExtractor(img_input)

                                    for layer in Layers:
                                        features_path = os.path.join(fDir, Model_Names[Mode][Model], layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{Model_Names[Mode][Model]}_{layer}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                                        np.save(features_path, features[layer])

# %% Create Representation for Binary Segmentation Mask Stimuli
preprocess = Mask_Dataset_Preprocessing

for Model in Models:
    for Mode in Training_Modes:

        FeaturesExtractor = TorchHub_FeatureExtractor(Model, Mode)

        for PS_range in PS_Ranges:

            iDir = os.path.join(sDir, 'Photorealistic_Dataset', f'PS_{PS_range[0].upper() + PS_range[1:]}_Range', 'Masks')
            fDir = os.path.join(rDir, 'Photorealistic_Dataset', f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

            PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{PS_range}_range.csv')
            ParkSpace_Description = _read_ParkSpace_log(PS_path)

            for object_name in Objects:
                for bg_idx, bg_alpha in Backgrounds:
                    for N, ID, FD in ParkSpace_Description:
                        for v_idx in Versions:

                            features_paths = [os.path.join(fDir, Model_Names[Mode][Model], layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{Model_Names[Mode][Model]}_{layer}_Mask_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy') for layer in Layers]
                            is_not_files = [not os.path.isfile(features_path) for features_path in features_paths]
                            if np.any(is_not_files): # Do not compute a representation that was already created !

                                img_path = os.path.join(iDir, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')

                                if os.path.isfile(img_path): # Do not try to compute representation of an image that does not exist !
                                    img_input = preprocess(img_path)
                                    features  = FeaturesExtractor(img_input)

                                    for layer in Layers:
                                        features_path = os.path.join(fDir, Model_Names[Mode][Model], layer, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{Model_Names[Mode][Model]}_{layer}_Mask_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                                        np.save(features_path, features[layer])     

# %% Extract the Raw Pixels Values for Selected Objects and Backgrounds
preprocess = {'':ImageNet_Stimuli_PreProcessing(), '_Mask':Mask_Dataset_Preprocessing}

for PS_range in PS_Ranges:
    for UseSegMask in Stimulus_Types:

        uDir = os.path.join(sDir, 'Photorealistic_Dataset', f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')
        iDir = os.path.join(uDir, 'Masks') if UseSegMask == '_Mask' else os.path.join(uDir, 'Images')
        fDir = os.path.join(rDir, 'Photorealistic_Dataset', f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

        PS_path = os.path.join(mDir, 'src', 'Stimulus_Creation', f'PS_{PS_range}_range.csv')
        ParkSpace_Description = _read_ParkSpace_log(PS_path)

        for object_name in Objects:
            for bg_idx, bg_alpha in Backgrounds:
                for N, ID, FD in ParkSpace_Description:
                    for v_idx in Versions:

                        img_pixels_path = os.path.join(fDir, 'RawPixels', object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'RawPixels{UseSegMask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy')
                        if not os.path.isfile(img_pixels_path): # Do not compute a representation that was already created !

                            base_path = os.path.join(iDir, object_name, f'Bg-{bg_idx}_Alpha{bg_alpha}', f'{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}')
                            img_path = base_path + '.png' if UseSegMask == '' else base_path + '.npy'

                            if os.path.isfile(img_path): # Do not try to compute representation of an image that does not exist !
                                img_input  = preprocess[UseSegMask](img_path)
                                img_pixels = img_input.numpy()[0][0].flatten()
                                np.save(img_pixels_path, img_pixels)

# %% Extract Representation from Nasr Dataset Using AlexNet Conv5 Layer to reproduce Nasr's Analysis on the 50 first stimuli of the dataset
Model =  'alexnet'
Layers = ['Conv5']

n_imgs = 900 # Number of stimulus to extract the representations per numerosity

## Using our generated stimuli of dot-arrays stimulus Dataset
iDir = os.path.join(sDir, 'Dot_Patterns_Dataset')
fDir = os.path.join(rDir, 'Dot_Patterns_Dataset')

preprocess = Nars_Dataset_Preprocessing

for Mode in Training_Modes:

    FeaturesExtractor = TorchHub_FeatureExtractor(Model, Mode)

    for i in range(n_imgs):
        for n in Nasr_Numerosity:
            for Set_Name in ['Standard', 'Control-1', 'Control-2']:
            
                features_path = os.path.join(fDir, Model_Names[Mode][Model], Layers[-1], f'{Model_Names[Mode][Model]}_{Layers[-1]}_Set_Name_Set_N-{n}_Stim-{i+1}.npy')
                if not os.path.isfile(features_path): # Do not compute a representation that was already created !

                    img_path = os.path.join(iDir, f'{Set_Name}_Set_N-{n}_Stim-{i+1}.npy')

                    if os.path.isfile(img_path): # Do not try to compute representation of an image that does not exist !
                        img_input = preprocess(img_path)
                        features  = FeaturesExtractor(img_input)

                        for layer in Layers:
                            features_path = os.path.join(fDir, Model_Names[Mode][Model], layer, f'{Model_Names[Mode][Model]}_{layer}_Set_Name_Set_N-{n}_Stim-{i+1}.npy')
                            np.save(features_path, features[layer])