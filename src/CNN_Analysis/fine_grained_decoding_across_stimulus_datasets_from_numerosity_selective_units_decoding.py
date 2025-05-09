# %% Imports and Constants
from sklearn.linear_model import RidgeCV
import os, time
import numpy as np
from utils import _read_param_space_log
from args import (
    mDir,
    dDir,
    Versions,
    Objects,
    Backgrounds,
    nObjects,
    nFeatures,
    PS_Ranges,
    Alphas,
)


lDir = os.path.join(dDir, "Decoding_Results", "Revision")
rDir = os.path.join(dDir, "Representations")

Models = ["AlexNet", "Random_AlexNet"]
SegMasks = ["", "_Mask"]
nSegMasks = len(SegMasks)
Layer = "Conv5"

Mode = "_3Way"
Correction = "_Uncorrected"


# %% Useful Methods to perform the fine-grained generalization decoding
def _load_stimuli_dataset(model, layer, ps_range, v_idx, _mask=""):
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

    X, y = [[] for i in range(nObjects)], []
    for i, object_info in enumerate(Objects):

        object_name, _ = object_info

        for bg_idx, bg_alpha in Backgrounds:

            for N, ID, FD in Param_Space_Description:

                features_path = os.path.join(
                    fDir,
                    model,
                    layer,
                    f"{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                if os.path.isfile(features_path):
                    X[i].append(np.load(features_path))

                    ## Same Numerosity targets for all objects
                    if i == 0:
                        y.append(np.log(N))

    for i in range(nObjects):
        X[i] = np.array(X[i])

    return X, np.array(y) - np.array(y).mean()  # centering target


MAE = lambda y_pred, y_true: np.mean(np.abs(y_true - y_pred))


def _train_test(X_train, X_test, y, Idx_Units):

    clf = RidgeCV(alphas=Alphas)

    MAE_InterCV = []
    for i in range(nObjects):

        ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
        clf.fit(X_train[i][:, Idx_Units], y)

        ## Compute the Predictions Performances on Left-out Objects
        for j in range(nObjects):

            if j != i:

                MAE_InterCV.append(MAE(clf.predict(X_test[j][:, Idx_Units]), y))

    return MAE_InterCV


# %% Execute
for PS_range in PS_Ranges:

    fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    sDir = os.path.join(lDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Model in Models:

        print(time.asctime(time.localtime()), PS_range[0].upper() + PS_range[1:], Model)

        results_path = os.path.join(
            sDir,
            f"Fine_Generalisation_{Model}_Decoding_Log_N_Across_Stimulus_Dataset",
        )

        Score_MAE, Score_Std = np.zeros(6), np.zeros(6)

        MAE_OuterCV = [[] for k in range(6)]
        for v_idx in Versions:

            X_Photorealistic, y = _load_stimuli_dataset(
                Model,
                Layer,
                PS_range,
                v_idx=v_idx,
                _mask=SegMasks[0],
            )

            X_Binary, _ = _load_stimuli_dataset(
                Model,
                Layer,
                PS_range,
                v_idx=v_idx,
                _mask=SegMasks[1],
            )

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
                X_train=X_Photorealistic,
                X_test=X_Photorealistic,
                y=y,
                Idx_Units=Idx_NS_Photorealistic,
            )

            ## Photorealistic Stimuli - Other Units
            MAE_OuterCV[1] += _train_test(
                X_train=X_Photorealistic,
                X_test=X_Photorealistic,
                y=y,
                Idx_Units=Idx_Other_Photorealistic,
            )

            ## Photorealistic Stimuli - Binary Numerosity-Selective Units
            MAE_OuterCV[2] += _train_test(
                X_train=X_Photorealistic,
                X_test=X_Photorealistic,
                y=y,
                Idx_Units=Idx_NS_Binary,
            )

            ## Binary Stimuli - Binary Numerosity-Selective Units
            MAE_OuterCV[3] += _train_test(
                X_train=X_Binary,
                X_test=X_Binary,
                y=y,
                Idx_Units=Idx_NS_Binary,
            )

            ## Binary Stimuli - Other Units
            MAE_OuterCV[4] += _train_test(
                X_train=X_Binary,
                X_test=X_Binary,
                y=y,
                Idx_Units=Idx_Other_Binary,
            )

            ## Binary Stimuli - Photorealistic Numerosity-Selective Units
            MAE_OuterCV[5] += _train_test(
                X_train=X_Binary,
                X_test=X_Binary,
                y=y,
                Idx_Units=Idx_NS_Photorealistic,
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
