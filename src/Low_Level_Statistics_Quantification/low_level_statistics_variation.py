# %% Imports and Constants
import numpy as np
from numpy import fft
import skimage.io as io
import matplotlib.pyplot as plt
import os
from utils import _read_param_space_log, mask, _avg_weibull_params
from args import mDir, PS_Ranges, Backgrounds, Objects, Versions, SegMask, Numerosity

iDir = os.path.join(mDir, "data", "Stimuli", "Photorealistic_Dataset")
wDir = os.path.join(
    mDir, "derivatives", "Low_Level_Statistics", "Local_Contrast_Measures"
)
fDir = os.path.join(mDir, "derivatives", "Low_Level_Statistics", "Frequency_Measures")
lDir = os.path.join(mDir, "derivatives", "Low_Level_Statistics", "Luminance_Measures")

nBackgrounds, nObjects = len(Backgrounds), len(Objects)

## Definition of the small vs. high numerosity per range
Numerosity_Range = {"subitizing": [1, 4], "estimation": [6, 24]}

# %% Mean & Std of Luminance : Compute the Low-level Statistics induced variation when changing stimulus identities (numerosity, object or background)
for ps_range in PS_Ranges:

    uDir = os.path.join(lDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
    sDir = os.path.join(iDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)

    Mean_Lum, Std_Lum = np.zeros((nBackgrounds * nObjects)), np.zeros(
        (nBackgrounds * nObjects)
    )
    for i in range(nBackgrounds):
        bg_idx, bg_alpha = Backgrounds[i]
        for j in range(nObjects):
            object_name = Objects[j]

            IgM_mean, IgM_std = [], []
            for N, ID, FD in Param_Space_Description:
                for v_idx in Versions:

                    base_path = os.path.join(
                        object_name,
                        f"Bg-{bg_idx}_Alpha{bg_alpha}",
                        f"{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}",
                    )

                    if SegMask == "":

                        ## PhotoRealistic Stimuli
                        img_path = os.path.join(sDir, "Images", f"{base_path}.png")

                        if os.path.isfile(img_path):
                            I = io.imread(img_path)[
                                :, :, :3
                            ]  # load image and discard the alpha channal I[:,:,3] containing only 255s !
                            Ig = np.mean(
                                I, axis=2
                            )  # convert to gray scale : (height, width, depth)
                            IgM = Ig[
                                mask != 0
                            ]  # mask the outer circle of the edges/contrasts map which are identically zero

                    else:

                        ## Segmentation Mask
                        seg_path = os.path.join(sDir, "Masks", f"{base_path}.npy")

                        if os.path.isfile(seg_path):

                            Ig = np.load(seg_path)
                            IgM = Ig[mask != 0]

                    ## Computing the mean and std of grayscale values
                    IgM_mean.append(np.mean(IgM))
                    IgM_std.append(np.std(IgM))

            Mean_Lum[i * nBackgrounds + j] = np.mean(IgM_mean)
            Std_Lum[i * nBackgrounds + j] = np.mean(IgM_std)

    np.save(os.path.join(uDir, f"UFG{SegMask}_Mean_Luminance.npy"), Mean_Lum)
    np.save(os.path.join(uDir, f"UFG{SegMask}_Std_Luminance.npy"), Std_Lum)

# %% Mean & Std of Luminance : Computing the Difference due to Numerosity i.e. High minus Low Numerosity
for ps_range in PS_Ranges:

    uDir = os.path.join(lDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
    sDir = os.path.join(iDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)

    Mean_Lum = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): {
            N: [] for N in Numerosity_Range[ps_range]
        }
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }
    Std_Lum = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): {
            N: [] for N in Numerosity_Range[ps_range]
        }
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }

    for i in range(nBackgrounds):
        bg_idx, bg_alpha = Backgrounds[i]
        for j in range(nObjects):
            object_name = Objects[j]

            for N, ID, FD in Param_Space_Description:
                if N in Numerosity_Range[ps_range]:
                    for v_idx in Versions:

                        img_path = os.path.join(
                            sDir,
                            object_name,
                            f"Bg-{bg_idx}_Alpha{bg_alpha}",
                            f"{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}",
                        )

                        if SegMask == "":

                            ## PhotoRealistic Stimuli
                            img_path = os.path.join(sDir, "Images", f"{base_path}.png")

                            if os.path.isfile(img_path):
                                I = io.imread(img_path)[
                                    :, :, :3
                                ]  # load image and discard the alpha channal I[:,:,3] containing only 255s !
                                Ig = np.mean(
                                    I, axis=2
                                )  # convert to gray scale : (height, width, depth)
                                IgM = Ig[
                                    mask != 0
                                ]  # mask the outer circle of the edges/contrasts map which are identically zero

                                ## Computing the mean and std of grayscale values
                                Mean_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                                    N
                                ].append(np.mean(IgM))
                                Std_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                                    N
                                ].append(np.std(IgM))

                        else:

                            ## Segmentation Mask
                            seg_path = os.path.join(sDir, "Masks", f"{base_path}.npy")

                            if os.path.isfile(seg_path):

                                Ig = np.load(seg_path)
                                IgM = Ig[mask != 0]

                                ## Computing the mean and std of grayscale values
                                Mean_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                                    N
                                ].append(np.mean(IgM))
                                Std_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                                    N
                                ].append(np.std(IgM))

    Difference_Mean_Lum_by_N, Difference_Std_Lum_by_N = [], []
    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:

            Mean_Lum_small_N = np.array(
                Mean_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[ps_range][0]
                ]
            )
            Mean_Lum_high_N = np.array(
                Mean_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[ps_range][-1]
                ]
            )
            Std_Lum_small_N = np.array(
                Std_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[ps_range][0]
                ]
            )
            Std_Lum_high_N = np.array(
                Std_Lum[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[ps_range][-1]
                ]
            )

            Difference_Mean_Lum_by_N.append(np.mean(Mean_Lum_high_N - Mean_Lum_small_N))
            Difference_Std_Lum_by_N.append(np.mean(Std_Lum_high_N - Std_Lum_small_N))

    np.save(
        os.path.join(
            uDir, "UFG_Difference_Mean_Luminance_High_minus_Small_Numerosity.npy"
        ),
        np.array(Difference_Mean_Lum_by_N),
    )
    np.save(
        os.path.join(
            uDir, "UFG_Difference_Std_Luminance_High_minus_Small_Numerosity.npy"
        ),
        np.array(Difference_Std_Lum_by_N),
    )

# %% Computing Total Energy in the Low SF & High SF
for PS_range in PS_Ranges:

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{PS_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)

    uDir = os.path.join(fDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    Energy_Low_SF = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): []
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }
    Energy_High_SF = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): []
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }

    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:
            for v_idx in Versions:

                NRJ_High_SF_path = os.path.join(
                    uDir,
                    f"Energy_High_SF{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )
                NRJ_Low_SF_path = os.path.join(
                    uDir,
                    f"Energy_Low_SF{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                NRJ_Low_SF, NRJ_High_SF = np.load(NRJ_Low_SF_path), np.load(
                    NRJ_High_SF_path
                )

                Energy_Low_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"].append(
                    NRJ_Low_SF.flatten()
                )
                Energy_High_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"].append(
                    NRJ_High_SF.flatten()
                )

    ## Computing total Energy in low-SF & high-SF band (defined relative to the size of Objects)
    rAvg_lowSF, rAvg_highSF = np.zeros((nBackgrounds * nObjects)), np.zeros(
        (nBackgrounds * nObjects)
    )
    i = 0
    for bg_idx, bg_alpha in Backgrounds:
        j = 0
        for object_name in Objects:

            rAvg_lowSF[i * nBackgrounds + j] = np.mean(
                Energy_Low_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"]
            )
            rAvg_highSF[i * nBackgrounds + j] = np.mean(
                Energy_High_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"]
            )

            j += 1
        i += 1

    np.save(
        os.path.join(
            uDir,
            f"UFG{SegMask}_Total_Rotational_Average_Energy_in_Low_Spatial_Frequency.npy",
        ),
        rAvg_lowSF,
    )
    np.save(
        os.path.join(
            uDir,
            f"UFG{SegMask}_Total_Rotational_Average_Energy_in_High_Spatial_Frequency.npy",
        ),
        rAvg_highSF,
    )

# %% Computing Total Energy in the Low SF & High SF per Numerosity
for PS_range in PS_Ranges:

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{PS_range}_range.csv")
    Param_Space_Description = _read_param_space_log(PS_path)

    uDir = os.path.join(fDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    Energy_Low_SF = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): {
            N: [] for N in Numerosity_Range[PS_range]
        }
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }
    Energy_High_SF = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): {
            N: [] for N in Numerosity_Range[PS_range]
        }
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }

    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:
            for v_idx in Versions:

                NRJ_High_SF_path = os.path.join(
                    uDir,
                    f"Energy_High_SF{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )
                NRJ_Low_SF_path = os.path.join(
                    uDir,
                    f"Energy_Low_SF{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                NRJ_Low_SF, NRJ_High_SF = np.load(NRJ_Low_SF_path), np.load(
                    NRJ_High_SF_path
                )

                for N in Numerosity_Range[PS_range]:
                    idx_N = Numerosity[PS_range].index(N)
                    Energy_Low_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                        N
                    ].append(NRJ_Low_SF[idx_N].flatten().copy())
                    Energy_High_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                        N
                    ].append(NRJ_High_SF[idx_N].flatten().copy())

    Difference_rAVG_by_N_low_SF, Difference_rAVG_by_N_high_SF = [], []
    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:

            AVG_rAVG_small_N_low_SF = np.mean(
                Energy_Low_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[PS_range][0]
                ]
            )
            AVG_rAVG_high_N_low_SF = np.mean(
                Energy_Low_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[PS_range][-1]
                ]
            )
            Difference_rAVG_by_N_low_SF.append(
                AVG_rAVG_high_N_low_SF - AVG_rAVG_small_N_low_SF
            )

            AVG_rAVG_small_N_high_SF = np.mean(
                Energy_High_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[PS_range][0]
                ]
            )
            AVG_rAVG_high_N_high_SF = np.mean(
                Energy_High_SF[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity_Range[PS_range][-1]
                ]
            )
            Difference_rAVG_by_N_high_SF.append(
                AVG_rAVG_high_N_high_SF - AVG_rAVG_small_N_high_SF
            )

    np.save(
        os.path.join(
            uDir,
            f"UFG_Difference_Total_Rotational_Average_Energy_High_minus_Small_Numerosity_low_SF{SegMask}.npy",
        ),
        np.array(Difference_rAVG_by_N_low_SF),
    )
    np.save(
        os.path.join(
            uDir,
            f"UFG_Difference_Total_Rotational_Average_Energy_High_minus_Small_Numerosity_high_SF{SegMask}.npy",
        ),
        np.array(Difference_rAVG_by_N_high_SF),
    )

# %% Compute Influence of Aggregate Fourier Magnitude measure on our PhotoRealistic Dataset
for PS_range in PS_Ranges:

    uDir = os.path.join(fDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    Aggregate_Magnitude = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): {
            N: [] for N in Numerosity[PS_range]
        }
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }

    for object_name in Objects:
        for bg_idx, bg_alpha in Backgrounds:
            for v_idx in Versions:

                AggMag = np.load(
                    os.path.join(
                        uDir,
                        f"Aggregate_rAvg_Fourier_Magnitude{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                    )
                )

                for N in Numerosity[PS_range]:
                    idx_N = Numerosity[PS_range].index(N)
                    Aggregate_Magnitude[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                        N
                    ].append(AggMag[idx_N].flatten().copy())

    ## Computing Influence of Numerosity (Statistic Value for High N - Value for Low N for a given Obj/Bg condition)
    Difference_AggMag_by_N = []
    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:

            AVG_AggMag_small_N = np.mean(
                Aggregate_Magnitude[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity[PS_range][0]
                ]
            )
            AVG_AggMag_high_N = np.mean(
                Aggregate_Magnitude[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][
                    Numerosity[PS_range][-1]
                ]
            )
            Difference_AggMag_by_N.append(AVG_AggMag_high_N - AVG_AggMag_small_N)

    np.save(
        os.path.join(
            uDir,
            f"UFG{SegMask}_Difference_Aggregate_Fourier_Magnitude_High_minus_Small_Numerosity.npy",
        ),
        np.array(Difference_AggMag_by_N),
    )

    ## Computing Influence of Backgrounds & Objects (Statistic average across N for a given Obj/Bg condition)
    rAvg_AggMag = np.zeros((nBackgrounds * nObjects))
    i = 0
    for bg_idx, bg_alpha in Backgrounds:
        j = 0
        for object_name in Objects:

            AggMag = []
            for N in Numerosity[PS_range]:
                AggMag.append(
                    Aggregate_Magnitude[object_name, f"bg-{bg_idx}_alpha{bg_alpha}"][N]
                )
            rAvg_AggMag[i * nBackgrounds + j] = np.mean(AggMag)

            j += 1
        i += 1

    np.save(
        os.path.join(uDir, f"UFG{SegMask}_Aggregate_Fourier_Magnitude.npy"), rAvg_AggMag
    )

## Texture Similarity & Image Complexity Distance between (object x background) pairs and difference induced by numerosity change
if SegMask == "":

    for ps_range in PS_Ranges:

        uDir = os.path.join(wDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")

        ## Load Weibull Representation of the Stimuli Dataset
        Beta, Gamma = [], []
        for bg_idx, bg_alpha in Backgrounds:
            bg_name = f"bg-{bg_idx}_alpha{bg_alpha}"
            for obj_name in Objects:
                weibull_params = _avg_weibull_params(uDir, obj_name, bg_name)
                Beta.append(weibull_params["b" + "N"])
                Gamma.append(weibull_params["g" + "N"])
        Beta = np.concatenate(Beta)
        Gamma = np.concatenate(Gamma)

        ## Compute Distance from Texture & Complexity Axes
        M = np.zeros([len(Beta), 2])
        M[:, 0] = Gamma.copy()
        M[:, 1] = Beta.copy()
        mB, mG = np.mean(Beta), np.mean(Gamma)
        c = np.array([mG, mB])
        minG, maxG = np.min(Gamma), np.max(Gamma)

        u, s, vh = np.linalg.svd(M, full_matrices=True)
        wC, wT = np.array(vh[0, :]), np.array(vh[1, :])
        norm_wC, norm_wT = np.sqrt(np.sum(wC**2)), np.sqrt(np.sum(wT**2))
        slope_wC, slope_wT = vh[1, 1] / vh[1, 0], vh[0, 1] / vh[0, 0]
        intercept_wC, intercept_wT = mB - slope_wC * mG, mB - slope_wT * mG
        fC = lambda x: x * slope_wC + intercept_wC
        dC = lambda p: np.dot(wC, p - c) / norm_wC
        DistC = [dC(M[i, :]) for i in range(len(M))]
        fT = lambda x: x * slope_wT + intercept_wT
        dT = lambda p: np.dot(wT, p - c) / norm_wT
        DistT = [dT(M[i, :]) for i in range(len(M))]

        ## Difference of Distances to Axes between high vs. low Numerosity
        diff_DistT, diff_DistC = np.zeros(nBackgrounds * nObjects), np.zeros(
            nBackgrounds * nObjects
        )
        for k in range(nBackgrounds * nObjects):
            diff_DistC[k] = DistC[4 * (k + 1) - 1] - DistC[4 * k]
            diff_DistT[k] = DistT[4 * (k + 1) - 1] - DistT[4 * k]

        np.save(
            os.path.join(
                uDir,
                "UFG_Difference_Dist_Along_Complexity_Axis_High_minus_Small_Numerosity.npy",
            ),
            diff_DistC,
        )
        np.save(
            os.path.join(
                uDir,
                "UFG_Difference_Dist_Along_Texture_Axis_High_minus_Small_Numerosity.npy",
            ),
            diff_DistT,
        )

        ## Mean Parameters & Distances along Axes per {Background x Object} condition
        mean_DistT, mean_DistC = np.zeros(nBackgrounds * nObjects), np.zeros(
            nBackgrounds * nObjects
        )
        for k in range(nBackgrounds * nObjects):
            mean_DistC[k] = np.mean(DistC[4 * k : 4 * (k + 1)])
            mean_DistT[k] = np.mean(DistT[4 * k : 4 * (k + 1)])

        np.save(
            os.path.join(uDir, "UFG_Distance_Along_Complexity_Axis.npy"), mean_DistC
        )
        np.save(os.path.join(uDir, "UFG_Distance_Along_Texture_Axis.npy"), mean_DistT)
