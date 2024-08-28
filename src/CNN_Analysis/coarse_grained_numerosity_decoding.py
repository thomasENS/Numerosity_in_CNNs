# %% Imports and Constants
from sklearn.linear_model import RidgeCV
import os
import numpy as np
from joblib import Parallel, delayed
from utils import _read_param_space_log, _load_labels
from args import (
    mDir,
    dDir,
    Naturals,
    Animals,
    PS_Ranges,
    nSampled,
    nCVloop,
    Alphas,
    Training_Modes,
    Models,
    Layers,
    Model_Names,
    SubSpace,
    subSpaceTrain,
    subSpaceTest,
    Modalities,
    Stimulus_Types,
)

nBackgrounds, nObjects = len(Naturals), len(Animals)


# %% Useful Methods to perform the coarse-grained generalization decoding
def _load_features(features_path, model, layer, modality, sub_space_name):
    """
    Load the features row associated with the features_path, applying features selection to reduce
    the dimension if necessary i.e. for Conv1, Conv2 and Conv3.
    """
    if (model == "RawPixels") or (
        model in ["AlexNet", "Random_AlexNet"] and layer in ["Conv4", "Conv5"]
    ):
        return np.load(features_path)
    else:
        selectedFeaturesIdx = np.load(
            os.path.join(
                iDir,
                f"{model}_{layer}_subSpace-{sub_space_name}_{modality}_selected_features_idx.npy",
            )
        )
        return np.load(features_path)[selectedFeaturesIdx]


def _load_stimuli_dataset(
    model,
    layer,
    ps_range,
    modality,
    load_space_idx,
    selected_features_space_idx,
    sampling_matrix,
    v_idx=1,
    _mask="",
    target_scale="Log",
):
    """
    Load all the representations extracted from a network's [layer] of the stimuli of a given subspace.
    """

    assert ps_range in [
        "subitizing",
        "estimation",
    ], 'range should be either "subitizing" or "estimation"'
    assert layer in [
        "Conv1",
        "Conv2",
        "Conv3",
        "Conv4",
        "Conv5",
    ], "AlexNet layers are ConvX with X in {1 ... 5}"

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)
    Backgrounds, Objects = SubSpace[load_space_idx]

    X, y = [], []
    for i in range(nObjects):
        for j in range(nBackgrounds):

            if sampling_matrix[i, j]:

                object_name = Objects[i]
                bg_idx, bg_alpha = Backgrounds[j]

                for N, ID, FD in Param_Space_Description:
                    if model != "RawPixels":
                        features_path = os.path.join(
                            fDir,
                            model,
                            layer,
                            object_name,
                            f"Bg-{bg_idx}_Alpha{bg_alpha}",
                            f"{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                        )
                    else:
                        features_path = os.path.join(
                            fDir,
                            "RawPixels",
                            object_name,
                            f"Bg-{bg_idx}_Alpha{bg_alpha}",
                            f"RawPixels{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                        )

                    if os.path.isfile(features_path):
                        try:
                            X.append(
                                _load_features(
                                    features_path,
                                    model,
                                    layer,
                                    modality,
                                    selected_features_space_idx,
                                )
                            )
                            y.append(_load_labels(N, ID, FD, modality, target_scale))
                        except:
                            pass

    return np.array(X), np.array(y) - np.array(y).mean()


def _create_sampling_matrix(nSampled):
    """
    Fct that create randomly a sampling matrix of nSampled {object x background} pairs among
    the 100 pairs available for one given (out of the four) coarse-grained subspaces.
    """

    assert 1 <= nSampled <= 100, "nSampled should be an integer between 1 and 100."

    Sampling_Matrix = np.zeros((nObjects, nBackgrounds))
    while np.sum(Sampling_Matrix) < nSampled:
        i, j = np.random.randint(nObjects), np.random.randint(nBackgrounds)
        Sampling_Matrix[i, j] = 1

    return Sampling_Matrix.astype(bool)


MAE = lambda y_pred, y_true: np.mean(np.abs(y_true - y_pred))


# %% Decoding linearly all the non-numerical Parameters across the hierarchy of Layers
def _ridge_decoding_across_hierarchy_JOBLIB(
    PS_range,
    Model,
    Modality,
    Layers,
    subSpaceTrain,
    subSpaceTest,
    UseSegMask="",
    target_scale="Log",
):

    clf = RidgeCV(alphas=Alphas)

    results_path = os.path.join(
        lDir,
        f"Full_Generalisation{UseSegMask}_{Model}_Decoding_{target_scale}_{Modality}",
    )
    score_MAE, score_Std = np.zeros([len(Layers)]), np.zeros([len(Layers)])

    ## Perform the Predictions Analysis for Several number of Training Samples
    for i in range(len(Layers)):
        Layer = Layers[i]

        ## Compute Predictions for the Generalisation Pattern - Either Full either Partial and Average the Errors across the cases.
        MAE_InterCV = []
        for sTrain in subSpaceTrain:

            ## Outer CV Loop - Just multiple loops to explore most of the [Objects x Backgrounds] space
            for k in range(nCVloop):

                Sampling_Matrix = _create_sampling_matrix(nSampled)

                X_train, y_train = _load_stimuli_dataset(
                    Model,
                    Layer,
                    PS_range,
                    Modality,
                    sTrain,
                    sTrain,
                    Sampling_Matrix,
                    _mask=UseSegMask,
                    target_scale=target_scale,
                )

                ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                clf.fit(X_train, y_train)

                ## Compute Outer Loop Ridge Regression MAE score
                for sTest in subSpaceTest[sTrain]:
                    X_test, y_test = _load_stimuli_dataset(
                        Model,
                        Layer,
                        PS_range,
                        Modality,
                        sTest,
                        sTrain,
                        Sampling_Matrix,
                        _mask=UseSegMask,
                        target_scale=target_scale,
                    )
                    MAE_InterCV.append(MAE(clf.predict(X_test), y_test))

        score_MAE[i] = np.mean(MAE_InterCV)
        score_Std[i] = np.std(MAE_InterCV)

    np.save(results_path + "_Across_Hierarchy_Score_MAE.npy", score_MAE)
    np.save(results_path + "_Across_Hierarchy_Score_Std.npy", score_Std)

    return None


# %% Decode all the Parametric Parameters across the Hierarchy of HCNNs Convolutional Layerss

TargetScale = "Log"  # 'Log' for log(N) as target (or replace by '' for y=N directly)

for PS_range in PS_Ranges:

    iDir = os.path.join(dDir, "Features_Selection")
    rDir = os.path.join(dDir, "CNN_Representations", "Photorealistic_Dataset")
    lDir = os.path.join(dDir, "Decoding_Results", "Photorealistic_Dataset")
    fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    iDir = os.path.join(iDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    lDir = os.path.join(lDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Model_Name in Models:
        for Mode in Training_Modes:

            Model = Model_Names[Mode][Model_Name]

            for UseSegMask in Stimulus_Types:

                Parallel(n_jobs=60)(
                    delayed(_ridge_decoding_across_hierarchy_JOBLIB)(
                        PS_range,
                        Model,
                        Modality,
                        Layers,
                        subSpaceTrain,
                        subSpaceTest,
                        UseSegMask=UseSegMask,
                        target_scale=TargetScale,
                    )
                    for Modality in Modalities
                )
