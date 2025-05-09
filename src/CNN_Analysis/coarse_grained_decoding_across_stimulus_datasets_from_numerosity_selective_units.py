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
    nFeatures,
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
rDir = os.path.join(dDir, "Representations")

Models = ["AlexNet", "Random_AlexNet"]
Layer = "Conv5"

Mode = "_3Way"
Correction = "_Uncorrected"


# %% Useful Methods to perform the coarse-grained generalization decoding
def _load_stimuli_dataset(
    model, layer, ps_range, load_space_idx, sampling_matrix, _mask="", v_idx=2
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

                    features_path = os.path.join(
                        fDir,
                        model,
                        layer,
                        f"{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
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


def _train_test(Model, Layer, PS_range, Idx_Units, SegMaskTrain, SegMaskTest, v_idx):

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
                _mask=SegMaskTrain,
                v_idx=v_idx,
            )

            ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
            clf.fit(X_train[:, Idx_Units], y_train)

            ## Compute Outer Loop Ridge Regression MAE score
            for sTest in subSpaceTest[sTrain]:
                X_test, y_test = _load_stimuli_dataset(
                    Model,
                    Layer,
                    PS_range,
                    sTest,
                    Sampling_Matrix,
                    _mask=SegMaskTest,
                    v_idx=v_idx,
                )
                MAE_InterCV.append(MAE(clf.predict(X_test[:, Idx_Units]), y_test))

    return MAE_InterCV


# %% Decoding linearly all the non-numerical Parameters across the hierarchy of Layers
clf = RidgeCV(alphas=Alphas)

for PS_range in PS_Ranges:

    fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    sDir = os.path.join(lDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Model in Models:

        results_path = os.path.join(
            sDir,
            f"Coarse_Generalisation_{Model}_Decoding_Log_N_Across_Stimulus_Dataset",
        )

        Score_MAE, Score_Std = np.zeros(6), np.zeros(6)

        MAE_OuterCV = [[] for k in range(6)]
        for v_idx in Versions:

            ## Indices of Numerosoty-Selective Units
            Idx_NS_Photorealistic = np.load(
                os.path.join(
                    sDir,
                    f"{Model}_{Layer}{Mode}_Number_Selective_Units{Correction}_Stimuli_v{v_idx}.npy",
                )
            )
            Idx_Other_Photorealistic = np.array(
                [idx for idx in range(nFeatures) if idx not in Idx_NS_Photorealistic]
            )
            Idx_NS_Binary = np.load(
                os.path.join(
                    sDir,
                    f"{Model}_{Layer}{Mode}_Number_Selective_Units{Correction}_Mask_Stimuli_v{v_idx}.npy",
                )
            )
            Idx_Other_Binary = np.array(
                [idx for idx in range(nFeatures) if idx not in Idx_NS_Binary]
            )

            ## Photorealistic Stimuli - Photorealitc Numerosity-Selective Units
            MAE_OuterCV[0] += _train_test(
                Model,
                Layer,
                PS_range,
                Idx_NS_Photorealistic,
                SegMaskTrain="",
                SegMaskTest="",
                v_idx=v_idx,
            )

            ## Photorealistic Stimuli - Other Units
            MAE_OuterCV[1] += _train_test(
                Model,
                Layer,
                PS_range,
                Idx_Other_Photorealistic,
                SegMaskTrain="",
                SegMaskTest="",
                v_idx=v_idx,
            )

            ## Photorealistic Stimuli - Binary Numerosity-Selective Units
            MAE_OuterCV[2] += _train_test(
                Model,
                Layer,
                PS_range,
                Idx_NS_Binary,
                SegMaskTrain="",
                SegMaskTest="",
                v_idx=v_idx,
            )

            ## Binary Stimuli - Binary Numerosity-Selective Units
            MAE_OuterCV[3] += _train_test(
                Model,
                Layer,
                PS_range,
                Idx_NS_Binary,
                SegMaskTrain="_Mask",
                SegMaskTest="_Mask",
                v_idx=v_idx,
            )

            ## Binary Stimuli - Other Units
            MAE_OuterCV[4] += _train_test(
                Model,
                Layer,
                PS_range,
                Idx_Other_Binary,
                SegMaskTrain="_Mask",
                SegMaskTest="_Mask",
                v_idx=v_idx,
            )

            ## Binary Stimuli - Photorealistic Numerosity-Selective Units
            MAE_OuterCV[5] += _train_test(
                Model,
                Layer,
                PS_range,
                Idx_NS_Photorealistic,
                SegMaskTrain="_Mask",
                SegMaskTest="_Mask",
                v_idx=v_idx,
            )

        for k in range(6):
            Score_MAE[k] = np.mean(MAE_OuterCV[k])
            Score_Std[k] = np.std(MAE_OuterCV[k])

        np.save(
            results_path + f"_From{Mode}_Numerosity_Selective_Units_Score_MAE.npy",
            Score_MAE,
        )
        np.save(
            results_path + f"_From{Mode}_Numerosity_Selective_Units_Score_Std.npy",
            Score_Std,
        )
