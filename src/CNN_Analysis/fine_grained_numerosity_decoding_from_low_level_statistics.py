# %% Imports and Constants
from sklearn.linear_model import RidgeCV
import os
import numpy as np
from utils import _read_param_space_log, _load_labels, _compute_param_space_point
from args import (
    mDir,
    dDir,
    Backgrounds,
    Objects,
    PS_Ranges,
    Versions,
    Alphas,
    Low_Level_Statistics,
    Numerosity,
    Spacing,
    Size_Area,
)

tDir = os.path.join(dDir, "Decoding_Results", "Photorealistic_Dataset")
wDir = os.path.join(
    mDir, "derivatives", "Low_Level_Statistics", "Local_Contrast_Measures"
)
fDir = os.path.join(mDir, "derivatives", "Low_Level_Statistics", "Frequency_Measures")
lDir = os.path.join(mDir, "derivatives", "Low_Level_Statistics", "Luminance_Measures")

nBackgrounds, nObjects = len(Backgrounds), len(Objects)

## Modality that we will decode from each stimulus
Modality = "N"


# %% Useful Methods
def _distance_to_axis_function(ps_range):
    """
    Fct that defines for a given numerosity range stimulus dataset (subitizing/estimation)
    the distance functions to the "image complexity" and "texture similarity" axes 
    """
    
    pDir = os.path.join(wDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")

    Beta, Gamma = [], []
    for bg_idx, bg_alpha in Backgrounds:
        bg_name = f"bg-{bg_idx}_alpha{bg_alpha}"
        for obj_name in Objects:
            Beta.append(
                np.load(
                    os.path.join(pDir, f"Sigma-12_{obj_name}_{bg_name}_Beta.npy")
                ).flatten()
            )
            Gamma.append(
                np.load(
                    os.path.join(pDir, f"Sigma-12_{obj_name}_{bg_name}_Gamma.npy")
                ).flatten()
            )
    Beta = np.concatenate(Beta)
    Gamma = np.concatenate(Gamma)

    M = np.zeros([len(Beta), 2])
    M[:, 0] = Gamma.copy()
    M[:, 1] = Beta.copy()
    mB, mG = np.mean(Beta), np.mean(Gamma)
    c = np.array([mG, mB])

    u, s, vh = np.linalg.svd(M, full_matrices=True)
    wC, wT = np.array(vh[0, :]), np.array(vh[1, :])
    norm_wC, norm_wT = np.sqrt(np.sum(wC**2)), np.sqrt(np.sum(wT**2))
    dC = lambda p: np.dot(wC, p - c) / norm_wC
    dT = lambda p: np.dot(wT, p - c) / norm_wT
    return dT, dC

def _load_features(
    Param_Space_Description,
    statistic,
    ps_range,
    object_name,
    bg_name,
    v_idx,
    dT,
    dC,
    UseSegMask,
):
    """
    Load the features row associated with the low-level statistic of interest for a given stimuli
    """

    x = []

    if statistic == "Mean_Lum":

        uDir = os.path.join(lDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
        result_path = os.path.join(
            uDir, f"Mean_Lum{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy"
        )
        Mean_Lum = np.load(result_path)

        for N, ID, FD in Param_Space_Description:

            Sp, SzA = _compute_param_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = (
                Numerosity[ps_range].index(N),
                Spacing[ps_range].index(Sp),
                Size_Area[ps_range].index(SzA),
            )
            x.append(Mean_Lum[idx_N, idx_Sp, idx_SzA])

    elif statistic == "Std_Lum":

        uDir = os.path.join(lDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
        result_path = os.path.join(
            uDir, f"Std_Lum{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy"
        )
        Std_Lum = np.load(result_path)

        for N, ID, FD in Param_Space_Description:

            Sp, SzA = _compute_param_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = (
                Numerosity[ps_range].index(N),
                Spacing[ps_range].index(Sp),
                Size_Area[ps_range].index(SzA),
            )
            x.append(Std_Lum[idx_N, idx_Sp, idx_SzA])

    elif statistic == "Energy_Low_SF":

        uDir = os.path.join(fDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
        result_path = os.path.join(
            uDir,
            f"Energy_Low_SF{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy",
        )
        NRJ_Low_SF = np.load(result_path)

        for N, ID, FD in Param_Space_Description:

            Sp, SzA = _compute_param_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = (
                Numerosity[ps_range].index(N),
                Spacing[ps_range].index(Sp),
                Size_Area[ps_range].index(SzA),
            )
            x.append(NRJ_Low_SF[idx_N, idx_Sp, idx_SzA])

    elif statistic == "Energy_High_SF":

        uDir = os.path.join(fDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
        result_path = os.path.join(
            uDir,
            f"Energy_High_SF{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy",
        )
        NRJ_High_SF = np.load(result_path)

        for N, ID, FD in Param_Space_Description:

            Sp, SzA = _compute_param_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = (
                Numerosity[ps_range].index(N),
                Spacing[ps_range].index(Sp),
                Size_Area[ps_range].index(SzA),
            )
            x.append(NRJ_High_SF[idx_N, idx_Sp, idx_SzA])

    elif statistic == "Dist_Texture":

        uDir = os.path.join(wDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
        result_path = os.path.join(uDir, f"Sigma-12_{object_name}_{bg_name}")
        Gamma, Beta = np.load(result_path + "_Gamma.npy"), np.load(
            result_path + "_Beta.npy"
        )

        for N, ID, FD in Param_Space_Description:

            Sp, SzA = _compute_param_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = (
                Numerosity[ps_range].index(N),
                Spacing[ps_range].index(Sp),
                Size_Area[ps_range].index(SzA),
            )
            p = np.array([Gamma[idx_N, idx_Sp, idx_SzA], Beta[idx_N, idx_Sp, idx_SzA]])
            x.append(dT(p))

    elif statistic == "Dist_Complexity":

        uDir = os.path.join(wDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
        result_path = os.path.join(uDir, f"Sigma-12_{object_name}_{bg_name}")
        Gamma, Beta = np.load(result_path + "_Gamma.npy"), np.load(
            result_path + "_Beta.npy"
        )

        for N, ID, FD in Param_Space_Description:

            Sp, SzA = _compute_param_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = (
                Numerosity[ps_range].index(N),
                Spacing[ps_range].index(Sp),
                Size_Area[ps_range].index(SzA),
            )
            p = np.array([Gamma[idx_N, idx_Sp, idx_SzA], Beta[idx_N, idx_Sp, idx_SzA]])
            x.append(dC(p))

    elif statistic == "Agg_Mag_Fourier":

        uDir = os.path.join(fDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
        result_path = os.path.join(
            uDir,
            f"Aggregate_rAvg_Fourier_Magnitude{UseSegMask}_{object_name}_{bg_name}_version-{v_idx}.npy",
        )
        Agg_Mag = np.load(result_path)

        for N, ID, FD in Param_Space_Description:

            Sp, SzA = _compute_param_space_point(N, ID, FD)
            idx_N, idx_Sp, idx_SzA = (
                Numerosity[ps_range].index(N),
                Spacing[ps_range].index(Sp),
                Size_Area[ps_range].index(SzA),
            )
            x.append(Agg_Mag[idx_N, idx_Sp, idx_SzA])

    else:
        raise NotImplementedError(f"{statistic} Statistic not currently used.")

    return x


def _load_stimuli(
    ps_range,
    modality,
    statistics,
    v_idx,
    object_name=None,
    bg_name=None,
    dT=None,
    dC=None,
    UseSegMask="",
):

    assert ps_range in [
        "subitizing",
        "estimation",
    ], 'range should be either "subitizing" or "estimation"'
    assert (object_name is not None and bg_name is None) or (
        object_name is None and bg_name is not None
    ), "at least, and only one, object_name or bg_name should be provided."

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)

    X = []
    for statistic in statistics:

        features = []
        ## Fine Grained Generalisation Across Objects
        if object_name is not None:
            for bg_idx, bg_alpha in Backgrounds:
                features.append(
                    _load_features(
                        Param_Space_Description,
                        statistic,
                        ps_range,
                        object_name,
                        f"bg-{bg_idx}_alpha{bg_alpha}",
                        v_idx,
                        dT,
                        dC,
                        UseSegMask,
                    )
                )
        ## Fine Grained Generalisation Across Backgrounds
        else:
            for object_name in Objects:
                features.append(
                    _load_features(
                        Param_Space_Description,
                        statistic,
                        ps_range,
                        object_name,
                        bg_name,
                        v_idx,
                        dT,
                        dC,
                        UseSegMask,
                    )
                )

        X.append(np.concatenate(features).copy())

    y = []
    for N, ID, FD in Param_Space_Description:
        y.append(_load_labels(N, ID, FD, modality))
    y = y * len(features)

    ## normalizing input
    X = np.stack(X, axis=1)
    X /= X.std(axis=0)
    ## centering target
    y = np.array(y) - np.array(y).mean()

    assert len(X) == len(
        y
    ), f"Mismatch in the number of samples between Features : {len(X)} & Targets : {len(y)}"

    return X, y


MAE = lambda y_pred, y_true: np.mean(np.abs(y_true - y_pred))

# %% Execute the fine-grained generalization using low-level statistics as input
clf = RidgeCV(alphas=Alphas)

for PS_range in PS_Ranges:

    sDir = os.path.join(tDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Statistics in [[stat] for stat in Low_Level_Statistics] + [
        Low_Level_Statistics
    ]:

        if "Dist_Texture" in Statistics or "Dist_Complexity":
            dT, dC = _distance_to_axis_function(PS_range)
        else:
            dT, dC = None, None

        results_path = os.path.join(
            sDir, f'UFG_Decoding_Log_{Modality}_Using_{"_".join(Statistics)}_Statistics'
        )

        ## Generalisation Across Objects
        score_MAE = np.zeros([nObjects, nObjects])

        if not os.path.isfile(results_path + "_Across_Objects" + "_Score_MAE.npy"):

            for i in range(nObjects):
                objTrain = Objects[i]

                MAE_InterCV = {j: [] for j in range(nObjects)}
                for v_idx in Versions:

                    X_train, y_train = _load_stimuli(
                        PS_range,
                        Modality,
                        Statistics,
                        v_idx,
                        object_name=objTrain,
                        bg_name=None,
                        dT=dT,
                        dC=dC,
                    )

                    ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                    clf.fit(X_train, y_train)

                    ## Compute Outer Loop Ridge Regression MAE score
                    for j in range(nObjects):

                        objTest = Objects[j]
                        print(objTrain, objTest)

                        ## Diagonal Generalises across the Version of the SAME Object Stimuli to see influence of Position
                        ## Off-Diagonal already takes the Generelasation across Objects into Account (also across Version even though it does not affect the Finer Grain Generalisation)
                        X_test, y_test = _load_stimuli(
                            PS_range,
                            Modality,
                            Statistics,
                            3 - v_idx,
                            object_name=objTest,
                            bg_name=None,
                            dT=dT,
                            dC=dC,
                        )  ## y=3-x i.e. (x=1,y=2 & x=2,y=1)

                        MAE_InterCV[j].append(MAE(clf.predict(X_test), y_test))

                for j in range(nObjects):
                    score_MAE[i, j] = np.mean(MAE_InterCV[j])

            np.save(results_path + "_Across_Objects" + "_Score_MAE.npy", score_MAE)

        ## Generalisation Across Backgrounds (collapsed Objects Dimension)

        score_MAE = np.zeros([nBackgrounds, nBackgrounds])

        if not os.path.isfile(results_path + "_Across_Backgrounds" + "_Score_MAE.npy"):

            for i in range(nBackgrounds):
                bg_idx, bg_alpha = Backgrounds[i]
                bgTrain = f"bg-{bg_idx}_alpha{bg_alpha}"

                MAE_InterCV = {j: [] for j in range(nBackgrounds)}
                for v_idx in Versions:

                    X_train, y_train = _load_stimuli(
                        PS_range,
                        Modality,
                        Statistics,
                        v_idx,
                        object_name=None,
                        bg_name=bgTrain,
                        dT=dT,
                        dC=dC,
                    )

                    ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                    clf.fit(X_train, y_train)

                    ## Compute Outer Loop Ridge Regression MAE score
                    for j in range(nBackgrounds):

                        bg_idx, bg_alpha = Backgrounds[j]
                        bgTest = f"bg-{bg_idx}_alpha{bg_alpha}"
                        print(bgTrain, bgTest)

                        X_test, y_test = _load_stimuli(
                            PS_range,
                            Modality,
                            Statistics,
                            3 - v_idx,
                            object_name=None,
                            bg_name=bgTest,
                            dT=dT,
                            dC=dC,
                        )

                        MAE_InterCV[j].append(MAE(clf.predict(X_test), y_test))

                for j in range(nBackgrounds):
                    score_MAE[i, j] = np.mean(MAE_InterCV[j])

            np.save(results_path + "_Across_Backgrounds" + "_Score_MAE.npy", score_MAE)
