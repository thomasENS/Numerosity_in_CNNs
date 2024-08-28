# %% Imports and Constants
from sklearn.linear_model import RidgeCV
import os
import numpy as np
from joblib import Parallel, delayed
from utils import _read_param_space_log, _load_labels
from args import (
    mDir,
    dDir,
    Backgrounds,
    Objects,
    PS_Ranges,
    Versions,
    nVersions,
    Alphas,
    Training_Modes,
    Models,
    Layers,
    Model_Names,
    Stimulus_Types,
)

nBackgrounds, nObjects = len(Backgrounds), len(Objects)

## Modality that we will decode from each stimulus
Modality = "N"


## Useful Methods
def _load_features(iDir, features_path, model, layer, modality, sub_space_name):
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


def _load_stimuli_object(
    iDir,
    fDir,
    model,
    layer,
    ps_range,
    modality,
    object_name,
    selected_features_space_obj,
    v_idx=1,
    _mask="",
    target_scale="Log",
):
    """
    Load all the representations extracted from a Network's [layer] of the stimuli containing a given [object].
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
    ], "Network layers are ConvX with X in {1 ... 5}"

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)

    X, y = [], []
    for bg_idx, bg_alpha in Backgrounds:

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
                X.append(
                    _load_features(
                        iDir,
                        features_path,
                        model,
                        layer,
                        modality,
                        selected_features_space_obj,
                    )
                )
                y.append(_load_labels(N, ID, FD, modality, target_scale))

    return np.array(X), np.array(y) - np.array(y).mean()  # centering target


def _load_stimuli_background(
    iDir,
    fDir,
    model,
    layer,
    ps_range,
    modality,
    bg_name,
    selected_features_space_bg,
    v_idx=1,
    _mask="",
    target_scale="Log",
):
    """
    Load all the representations extracted from a Network's [layer] of the stimuli pasted on the given [background]
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
    ], "Network layers are ConvX with X in {1 ... 5}"

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)

    bg_idx, bg_alpha = (
        bg_name.split("_")[0].split("bg-")[1],
        bg_name.split("_")[1].split("alpha")[1],
    )

    X, y = [], []
    for object_name in Objects:

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
                X.append(
                    _load_features(
                        iDir,
                        features_path,
                        model,
                        layer,
                        modality,
                        selected_features_space_bg,
                    )
                )
                y.append(_load_labels(N, ID, FD, modality, target_scale))

    return np.array(X), np.array(y) - np.array(y).mean()  # centering target


MAE = lambda y_pred, y_true: np.mean(np.abs(y_true - y_pred))


def _ridge_finer_grain_generalisation_JOBLIB(
    Model, Layer, PS_range, UseSegMask="", target_scale="Log"
):
    clf = RidgeCV(alphas=Alphas)

    iDir = os.path.join(dDir, "Features_Selection")
    rDir = os.path.join(dDir, "CNN_Representations", "Photorealistic_Dataset")
    lDir = os.path.join(dDir, "Decoding_Results", "Photorealistic_Dataset")
    fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    iDir = os.path.join(iDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    lDir = os.path.join(lDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    results_path = (
        os.path.join(
            lDir,
            f"Finer_Grain_Generalisation{UseSegMask}_{Model}_{Layer}_Decoding_{target_scale}_{Modality}",
        )
        if Model != "RawPixels"
        else os.path.join(
            lDir,
            f"Finer_Grain_Generalisation_{Model}_Decoding_{target_scale}_{Modality}",
        )
    )

    score_MAE, score_Std = np.zeros([nObjects, nObjects]), np.zeros(
        [nObjects, nObjects]
    )
    predictions = np.zeros(
        (nObjects, nObjects, nBackgrounds * 64 * nVersions)
    )  # shape : (20, 20, 2560) i.e. pred[i, j, :1280] correspond to the predictions of the 2nd version of the stimuli (since X_test on 3-v_idx !)

    if not os.path.isfile(results_path + "_Across_Objects" + "_Score_MAE.npy"):

        for i in range(nObjects):
            objTrain = Objects[i]

            y_pred, MAE_InterCV = {j: [] for j in range(nObjects)}, {
                j: [] for j in range(nObjects)
            }
            for v_idx in Versions:

                X_train, y_train = _load_stimuli_object(
                    iDir,
                    fDir,
                    Model,
                    Layer,
                    PS_range,
                    Modality,
                    objTrain,
                    objTrain,
                    v_idx,
                    _mask=UseSegMask,
                    target_scale=target_scale,
                )

                ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                clf.fit(X_train, y_train)

                ## Compute Outer Loop Ridge Regression MAE score
                for j in range(nObjects):

                    objTest = Objects[j]
                    print(objTrain, objTest)

                    ## Diagonal Generalises across the Version of the SAME Object Stimuli to see influence of Position
                    ## Off-Diagonal already takes the Generelasation across Objects into Account (also across Version even though it does not affect the Finer Grain Generalisation)
                    X_test, y_test = _load_stimuli_object(
                        iDir,
                        fDir,
                        Model,
                        Layer,
                        PS_range,
                        Modality,
                        objTest,
                        objTrain,
                        3 - v_idx,
                        _mask=UseSegMask,
                        target_scale=target_scale,
                    )  ## y=3-x i.e. (x=1,y=2 & x=2,y=1)

                    MAE_InterCV[j].append(MAE(clf.predict(X_test), y_test))
                    y_pred[j].append(clf.predict(X_test))

            for j in range(nObjects):
                score_MAE[i, j] = np.mean(MAE_InterCV[j])
                score_Std[i, j] = np.std(MAE_InterCV[j])

            for j in range(nObjects):
                predictions[i, j] = np.concatenate(y_pred[j])

        np.save(results_path + "_Across_Objects" + "_Score_MAE.npy", score_MAE)
        np.save(results_path + "_Across_Objects" + "_Score_Std.npy", score_Std)
        np.save(results_path + "_Predictions_Across_Objects.npy", predictions)

    # Generalisation Across Backgrounds (collapsed Objects Dimension)

    score_MAE, score_Std = np.zeros([nBackgrounds, nBackgrounds]), np.zeros(
        [nBackgrounds, nBackgrounds]
    )
    predictions = np.zeros(
        (nBackgrounds, nObjects, nBackgrounds * 64 * nVersions)
    )  # shape : (20, 20, 2560) i.e. pred[i, j, :1280] correspond to the predictions of the 2nd version of the stimuli (since X_test on 3-v_idx !)

    if not os.path.isfile(results_path + "_Across_Backgrounds" + "_Score_MAE.npy"):

        for i in range(nBackgrounds):
            bg_idx, bg_alpha = Backgrounds[i]
            bgTrain = f"bg-{bg_idx}_alpha{bg_alpha}"

            y_pred, MAE_InterCV = {j: [] for j in range(nBackgrounds)}, {
                j: [] for j in range(nBackgrounds)
            }
            for v_idx in Versions:

                X_train, y_train = _load_stimuli_background(
                    iDir,
                    fDir,
                    Model,
                    Layer,
                    PS_range,
                    Modality,
                    bgTrain,
                    bgTrain,
                    v_idx,
                    _mask=UseSegMask,
                    target_scale=target_scale,
                )

                ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                clf.fit(X_train, y_train)

                ## Compute Outer Loop Ridge Regression MAE score
                for j in range(nBackgrounds):

                    bg_idx, bg_alpha = Backgrounds[j]
                    bgTest = f"bg-{bg_idx}_alpha{bg_alpha}"
                    print(bgTrain, bgTest)

                    X_test, y_test = _load_stimuli_background(
                        iDir,
                        fDir,
                        Model,
                        Layer,
                        PS_range,
                        Modality,
                        bgTest,
                        bgTrain,
                        3 - v_idx,
                        _mask=UseSegMask,
                        target_scale=target_scale,
                    )

                    MAE_InterCV[j].append(MAE(clf.predict(X_test), y_test))
                    y_pred[j].append(clf.predict(X_test))

            for j in range(nBackgrounds):
                score_MAE[i, j] = np.mean(MAE_InterCV[j])
                score_Std[i, j] = np.std(MAE_InterCV[j])

            for j in range(nBackgrounds):
                predictions[i, j] = np.concatenate(y_pred[j])

        np.save(results_path + "_Across_Backgrounds" + "_Score_MAE.npy", score_MAE)
        np.save(results_path + "_Across_Backgrounds" + "_Score_Std.npy", score_Std)
        np.save(results_path + "_Predictions_Across_Backgrounds.npy", predictions)

    return None


# %% JOBLIB Optimisation of the Inner CV loops

TargetScale = "Log"  # 'Log' for log(N) as target (or replace by '' for y=N directly)

for Model_Name in Models:
    for Mode in Training_Modes:

        Model = Model_Names[Mode][Model_Name]

        for UseSegMask in Stimulus_Types:

            Parallel(n_jobs=-1)(
                delayed(_ridge_finer_grain_generalisation_JOBLIB)(
                    Model,
                    Layer,
                    PS_range,
                    UseSegMask=UseSegMask,
                    target_scale=TargetScale,
                )
                for Layer in Layers
                for PS_range in PS_Ranges
            )
