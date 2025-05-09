# %% Imports and Constants
from sklearn.linear_model import RidgeCV
import os
import numpy as np
from utils import _read_param_space_log
from args import (
    mDir,
    dDir,
    Versions,
    Naturals,
    Animals,
    PS_Ranges,
    nSampled,
    nCVloop,
    Alphas,
    SubSpace,
    subSpaceTrain,
    subSpaceTest,
)

nBackgrounds, nObjects = len(Naturals), len(Animals)

lDir = os.path.join(dDir, "Decoding_Results", "Revision")
rDir = os.path.join(dDir, "Representations", "Revision")

Model = "Random_AlexNet"
Layer = "Conv5"
UseSegMask = ""
UseDistribution = "_uniform"  # or "" for normal distribution

Gains = (
    [1 / 500, 1 / 10, 2, 10] if UseDistribution == "" else [1 / 500, 1 / 10, 1, 2, 10]
)
nGains = len(Gains)


# %% Useful Methods to perform the coarse-grained generalization decoding
def _load_stimuli_dataset(
    model,
    layer,
    ps_range,
    load_space_idx,
    sampling_matrix,
    idx_gain,
    _mask="",
    _distribution="",
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

    PS_path = os.path.join(mDir, "Stimulus_Creation", f"new_PS_{ps_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)
    Backgrounds, Objects = SubSpace[load_space_idx]

    X, y = [], []
    for i in range(nObjects):
        for j in range(nBackgrounds):

            if sampling_matrix[i, j]:

                object_name, _ = Objects[i]
                bg_idx, bg_alpha = Backgrounds[j]

                for N, ID, FD in Param_Space_Description:
                    for v_idx in Versions:

                        features_path = os.path.join(
                            fDir,
                            model,
                            layer,
                            f"{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}_gain{_distribution}-{idx_gain}.npy",
                        )

                        if os.path.isfile(features_path):
                            X.append(np.load(features_path))
                            y.append(np.log(N))

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
    Layer,
    subSpaceTrain,
    subSpaceTest,
    UseSegMask="",
    UseDistribution="",
):

    clf = RidgeCV(alphas=Alphas)

    results_path = os.path.join(
        lDir,
        f"Coarse_Generalisation{UseSegMask}_{Model}{UseDistribution}_Decoding_Log_N",
    )
    score_MAE, score_Std = np.zeros([nGains]), np.zeros([nGains])

    ## Perform the Predictions Analysis for Several number of Training Samples
    for idx_gain in range(nGains):

        ## Compute Predictions for the Generalisation Pattern - Either Coarse either Partial and Average the Errors across the cases.
        MAE_InterCV = []
        for sTrain in subSpaceTrain:

            ## Outer CV Loop - Just multiple loops to explore most of the [Objects x Backgrounds] space
            for k in range(nCVloop):

                Sampling_Matrix = _create_sampling_matrix(nSampled)

                X_train, y_train = _load_stimuli_dataset(
                    Model,
                    Layer,
                    PS_range,
                    sTrain,
                    Sampling_Matrix,
                    idx_gain,
                    _mask=UseSegMask,
                    _distribution=UseDistribution,
                )

                ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                clf.fit(X_train, y_train)

                ## Compute Outer Loop Ridge Regression MAE score
                for sTest in subSpaceTest[sTrain]:
                    X_test, y_test = _load_stimuli_dataset(
                        Model,
                        Layer,
                        PS_range,
                        sTest,
                        Sampling_Matrix,
                        idx_gain,
                        _mask=UseSegMask,
                        _distribution=UseDistribution,
                    )
                    MAE_InterCV.append(MAE(clf.predict(X_test), y_test))

        score_MAE[idx_gain] = np.mean(MAE_InterCV)
        score_Std[idx_gain] = np.std(MAE_InterCV)

    np.save(results_path + "_Variable_Initialization_Score_MAE.npy", score_MAE)
    np.save(results_path + "_Variable_Initialization_Score_Std.npy", score_Std)

    return None


# %% Decode all the Parametric Parameters across the Hierarchy of HCNNs Convolutional Layerss
for PS_range in PS_Ranges:

    fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    lDir = os.path.join(lDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    _ridge_decoding_across_hierarchy_JOBLIB(
        PS_range,
        Model,
        Layer,
        subSpaceTrain,
        subSpaceTest,
        UseSegMask=UseSegMask,
        UseDistribution=UseDistribution,
    )
