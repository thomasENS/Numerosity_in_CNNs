# %% Imports & Constants
import numpy as np
from sklearn.linear_model import RidgeCV
import os
from utils import _read_param_space_log
from args import (
    mDir,
    dDir,
    Naturals,
    Animals,
    PS_Ranges,
    nSampled,
    nCVloop,
    Alphas,
    SubSpace,
    subSpaceTrain,
    subSpaceTest,
    nFeatures,
    Percentages,
    nPercentages,
    nImg,
)

rDir = os.path.join(dDir, "CNN_Representations", "Photorealistic_Dataset")
sDir = os.path.join(dDir, "Decoding_Results", "Dot_Patterns_Dataset")

nBackgrounds, nObjects = len(Naturals), len(Animals)

## Information about the networks used for this analysis
Models = ["AlexNet", "Random_AlexNet"]
Layer = "Conv5"

## Information about which stimuli are used in this decoding and which units are considered number selective
UseSegMask = ""  # '_Mask' for the binary mask or '' for photorealistic stimuli
Correction = "_Uncorrected"  # '_Uncorrected' for the uncorrected (without bonferonni correction) units

## Shared numerosities between our photorealistic dataset and the dot-arrays dataset (targets of this decoding scheme)
Numerosity = [1, 2, 4, 6, 10, 24]

## Information useful for the decoding with various fraction of the whole unit population
nBoostraps = np.round(1 / Percentages).astype(int)
nNeurons = np.round(nFeatures * Percentages).astype(int)
Neurons_Idx = np.array([i for i in range(nFeatures)])

clf = RidgeCV(alphas=Alphas)


# %% Useful Methods to perform the coarse-grained generalization decoding
def _load_stimuli_dataset(
    model,
    layer,
    load_space_idx,
    number_selective_units,
    sampling_matrix,
    numerosity,
    v_idx=1,
    _mask="",
):
    """
    Load all the representations extracted from AlexNet's [layer] of the stimuli
    pasted on the given [background] grouped by Object used to create the stimuli.
    """

    assert layer in ["Conv5"], "AlexNet layers available for this analysis is Conv5"

    Backgrounds, Objects = SubSpace[load_space_idx]

    X, y = [], []
    for ps_range in PS_Ranges:

        uDir = os.path.join(rDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")

        PS_path = os.path.join(
            mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv"
        )
        Param_Space_Description = _read_param_space_log(PS_path)

        for i in range(nObjects):
            for j in range(nBackgrounds):

                if sampling_matrix[i, j]:

                    object_name = Objects[i]
                    bg_idx, bg_alpha = Backgrounds[j]

                    for N, ID, FD in Param_Space_Description:
                        if N in numerosity:
                            if model != "RawPixels":
                                features_path = os.path.join(
                                    uDir,
                                    model,
                                    layer,
                                    object_name,
                                    f"Bg-{bg_idx}_Alpha{bg_alpha}",
                                    f"{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                                )
                            else:
                                features_path = os.path.join(
                                    uDir,
                                    "RawPixels",
                                    object_name,
                                    f"Bg-{bg_idx}_Alpha{bg_alpha}",
                                    f"RawPixels{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                                )

                            if os.path.isfile(features_path):
                                if model != "RawPixels":
                                    X.append(
                                        np.load(features_path)[number_selective_units]
                                    )
                                else:
                                    X.append(np.load(features_path))
                                y.append(np.log(N))  # Predictions of Log-Numerosity

    return np.array(X), np.array(y) - np.array(y).mean()  # Centered Targets


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


Full_Sampling_Matrix = np.ones((nObjects, nBackgrounds)).astype(bool)

MAE = lambda y_pred, y_true: np.mean(np.abs(y_true - y_pred))

# %% Numerosity Coarse-Grained Generalization Decoding - Using Selected Features Number Units (follwing Nasr's 2019 analysis)
for Model in Models:

    Number_Selective_Features_Idx = np.load(
        os.path.join(
            sDir,
            f"{Model}_{Layer}_Dots_Number_Selective_Units{Correction}_Reproduction_nImgs-{nImg}.npy",
        )
    )
    Non_Selective_Population_Idx = np.array(
        [idx for idx in range(nFeatures) if idx not in Number_Selective_Features_Idx]
    )
    Non_Selective_Features_Idx = np.load(
        os.path.join(
            sDir,
            f"{Model}_{Layer}_Dots_Number_Non_Selective_Units{Correction}_Reproduction_nImgs-{nImg}.npy",
        )
    )

    results_path = os.path.join(
        sDir,
        f"Full_Generalisation{UseSegMask}_{Model}_{Layer}_Decoding_Log_N_from_Number_Selective_Units{Correction}_nImgs-{nImg}",
    )
    # 0: Number Selective Units / 1: Non-Selective Units / 2: Non-Selective Population / 3: RawPixels
    score_MAE, score_Std = np.zeros(4), np.zeros(4)

    ## Compute Predictions for the Generalisation Pattern - Based on Number Selective Units
    MAE_InterCV = []
    for sTrain in subSpaceTrain:

        ## Outer CV Loop - Just multiple loops to explore most of the [Objects x Backgrounds] space
        for k in range(nCVloop):

            np.random.shuffle(Number_Selective_Features_Idx)

            Sampling_Matrix = _create_sampling_matrix(nSampled)

            X_train, y_train = _load_stimuli_dataset(
                Model,
                Layer,
                sTrain,
                Number_Selective_Features_Idx,
                Sampling_Matrix,
                Numerosity,
                _mask=UseSegMask,
            )

            ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
            clf.fit(X_train, y_train)

            ## Compute Outer Loop Ridge Regression MAE score
            for sTest in subSpaceTest[sTrain]:
                X_test, y_test = _load_stimuli_dataset(
                    Model,
                    Layer,
                    sTest,
                    Number_Selective_Features_Idx,
                    Sampling_Matrix,
                    Numerosity,
                    _mask=UseSegMask,
                )
                MAE_InterCV.append(MAE(clf.predict(X_test), y_test))

    score_MAE[0] = np.mean(MAE_InterCV)
    score_Std[0] = np.std(MAE_InterCV)

    ## Compute Predictions for the Generalisation Pattern - Based on Non Selective Units
    MAE_InterCV = []
    for sTrain in subSpaceTrain:

        ## Outer CV Loop - Just multiple loops to explore most of the [Objects x Backgrounds] space
        for k in range(nCVloop):

            np.random.shuffle(Non_Selective_Features_Idx)

            Sampling_Matrix = _create_sampling_matrix(nSampled)

            X_train, y_train = _load_stimuli_dataset(
                Model,
                Layer,
                sTrain,
                Non_Selective_Features_Idx,
                Sampling_Matrix,
                Numerosity,
                _mask=UseSegMask,
            )

            ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
            clf.fit(X_train, y_train)

            ## Compute Outer Loop Ridge Regression MAE score
            for sTest in subSpaceTest[sTrain]:
                X_test, y_test = _load_stimuli_dataset(
                    Model,
                    Layer,
                    sTest,
                    Non_Selective_Features_Idx,
                    Sampling_Matrix,
                    Numerosity,
                    _mask=UseSegMask,
                )
                MAE_InterCV.append(MAE(clf.predict(X_test), y_test))

    score_MAE[1] = np.mean(MAE_InterCV)
    score_Std[1] = np.std(MAE_InterCV)

    ## Compute Predictions for the Generalisation Pattern - Based on Non Selective Units with constraint (#features = #selectiveUnits)

    MAE_InterCV = []
    for sTrain in subSpaceTrain:

        ## Outer CV Loop - Just multiple loops to explore most of the [Objects x Backgrounds] space
        for k in range(nCVloop):

            Sampling_Matrix = _create_sampling_matrix(nSampled)

            X_train, y_train = _load_stimuli_dataset(
                Model,
                Layer,
                sTrain,
                Non_Selective_Population_Idx,
                Sampling_Matrix,
                Numerosity,
                _mask=UseSegMask,
            )

            ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
            clf.fit(X_train, y_train)

            ## Compute Outer Loop Ridge Regression MAE score
            for sTest in subSpaceTest[sTrain]:
                X_test, y_test = _load_stimuli_dataset(
                    Model,
                    Layer,
                    sTest,
                    Non_Selective_Population_Idx,
                    Sampling_Matrix,
                    Numerosity,
                    _mask=UseSegMask,
                )
                MAE_InterCV.append(MAE(clf.predict(X_test), y_test))

    score_MAE[2] = np.mean(MAE_InterCV)
    score_Std[2] = np.std(MAE_InterCV)

    ## Compute Predictions for the Generalisation Pattern - Based on RawPixels
    MAE_InterCV = []
    for sTrain in subSpaceTrain:

        ## Outer CV Loop - Just multiple loops to explore most of the [Objects x Backgrounds] space
        for k in range(nCVloop):

            Sampling_Matrix = _create_sampling_matrix(nSampled)

            X_train, y_train = _load_stimuli_dataset(
                "RawPixels",
                Layer,
                sTrain,
                Non_Selective_Features_Idx,
                Sampling_Matrix,
                Numerosity,
                _mask=UseSegMask,
            )

            ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
            clf.fit(X_train, y_train)

            ## Compute Outer Loop Ridge Regression MAE score
            for sTest in subSpaceTest[sTrain]:
                X_test, y_test = _load_stimuli_dataset(
                    "RawPixels",
                    Layer,
                    sTest,
                    Non_Selective_Features_Idx,
                    Sampling_Matrix,
                    Numerosity,
                    _mask=UseSegMask,
                )
                MAE_InterCV.append(MAE(clf.predict(X_test), y_test))

    score_MAE[3] = np.mean(MAE_InterCV)
    score_Std[3] = np.std(MAE_InterCV)

    np.save(results_path + "_Score_MAE.npy", score_MAE)
    np.save(results_path + "_Score_Std.npy", score_Std)

# %% Decoding for Various number of Units
for Model in Models:

    results_path = os.path.join(
        sDir,
        f"Coarse_Grained_Generalisation_{Model}_{Layer}_Decoding_Log_N_Redundancy_Assessement_Photorealistic_Stimuli",
    )
    score_MAE_RandomUnits_CV, score_Std_RandomUnits_CV = np.zeros(
        nPercentages
    ), np.zeros(nPercentages)

    ## 4-fold Cross-Validation Decoding Scheme : Coarse Grained Generalisation Scheme + Congruent/Incongruent Stimuli
    MAE_RandomUnits_CV = {n: [] for n in range(nPercentages)}
    for sTrain in subSpaceTrain:

        ## Compute Predictions for the Generalisation Pattern - Either Full either Partial and Average the Errors across the cases.
        X_train, y_train = _load_stimuli_dataset(
            Model, Layer, sTrain, Neurons_Idx, Full_Sampling_Matrix, Numerosity
        )
        X_test, y_test = _load_stimuli_dataset(
            Model,
            Layer,
            subSpaceTest[sTrain],
            Neurons_Idx,
            Full_Sampling_Matrix,
            Numerosity,
        )

        ## Compute Predictions for nNeuron Units
        for n in range(nPercentages):

            for _ in range(nBoostraps[n] * 2):

                np.random.shuffle(Neurons_Idx)

                ## Find the optimal hyperparameter - Inner CV Loop - Leave One [Sample] Out / Negative MSE is used as the score.
                clf.fit(X_train[:, Neurons_Idx[: nNeurons[n]]], y_train)

                ## Compute Generalization Score on the left-out subspace
                score = MAE(clf.predict(X_test[:, Neurons_Idx[: nNeurons[n]]]), y_test)
                MAE_RandomUnits_CV[n].append(score)

    for n in range(nPercentages):
        score_MAE_RandomUnits_CV[n] = np.mean(MAE_RandomUnits_CV[n])
        score_Std_RandomUnits_CV[n] = np.std(MAE_RandomUnits_CV[n])

    np.save(
        results_path + "_Score_MAE_Randomly_Chosen_Units.npy", score_MAE_RandomUnits_CV
    )
    np.save(
        results_path + "_Score_Std_Randomly_Chosen_Units.npy", score_Std_RandomUnits_CV
    )
