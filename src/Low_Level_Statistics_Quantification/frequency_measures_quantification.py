# %% Imports and Constants
import numpy as np
from numpy import fft
import skimage.io as io
import matplotlib.pyplot as plt
import os
from utils import _read_parkspace_log, _compute_park_space_point
from args import (
    mDir,
    PS_Ranges,
    Backgrounds,
    Objects,
    Versions,
    SegMask,
    Numerosity,
    Spacing,
    Size_Area,
    ImgSize,
    cutoff_frequency,
)

fDir = os.path.join(mDir, "data", "Stimuli", "Photorealistic_Dataset")
rDir = os.path.join(mDir, "derivatives", "Low_Level_Statistics", "Frequency_Measures")

nBackgrounds, nObjects = len(Backgrounds), len(Objects)


# %% Useful Methods to compute Energy in High Spatial Frequencies
def rotational_average_power_spectrum(If):
    """
    Fct that computes the rotational average power spectrum of a given 2D fft
    """

    ImgSize = len(If)
    rAvg = np.zeros(int(ImgSize / 2))

    ## Zero-Frequency Value of the Power Spectrum (Magnitude would be without **2)
    rAvg[0] = np.abs(If[int(ImgSize / 2), int(ImgSize / 2)]) ** 2

    # Create meshgrid for x and y
    x, y = np.meshgrid(
        np.arange(-ImgSize / 2, ImgSize / 2), np.arange(-ImgSize / 2, ImgSize / 2)
    )

    # Convert to polar coordinates and round to integers
    _, radius = np.arctan2(y, x), np.round(np.sqrt(x**2 + y**2))

    ## All Remaining Frequency Rotational Average Values
    for r in range(1, int(ImgSize / 2)):
        rAvg[r] = np.mean(np.abs(If[radius == r]) ** 2)

    ## Remark : np.mean(np.abs(If[radius == r]))**2 or np.mean(np.abs(If[radius == r])**2)
    ## since Energy is Magnitude**2, it seems more logical to have mean(Mag**2)
    ## indeed, we take the Avg RotEnergy by squaring each Mag directly rather than
    ## computing the Avg RotMag and then squaring it to obtain Energy.

    return rAvg


# %% Computing Energy Spectrum
for PS_range in PS_Ranges:

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{PS_range}_range.csv")
    ParkSpace_Description = _read_parkspace_log(PS_path)

    sDir = os.path.join(fDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    sDir = (
        os.path.join(sDir, "Images") if SegMask == "" else os.path.join(sDir, "Masks")
    )
    uDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:

            for N, ID, FD in ParkSpace_Description:
                for v_idx in Versions:

                    img_path = os.path.join(
                        sDir,
                        object_name,
                        f"Bg-{bg_idx}_Alpha{bg_alpha}",
                        f"{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}",
                    )
                    img_path = img_path + ".png" if SegMask == "" else img_path + ".npy"

                    if os.path.isfile(img_path):

                        result_path = os.path.join(
                            uDir,
                            f"rAvg_Fourier_Power{SegMask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                        )

                        if not os.path.isfile(result_path):

                            if SegMask == "":
                                I = io.imread(img_path)[
                                    :, :, :3
                                ]  # load image and discard the alpha channal I[:,:,3] containing only 255s !
                                Ig = np.mean(
                                    I, axis=2
                                )  # convert to gray scale : (height, width, depth)
                            else:
                                Ig = np.load(img_path)

                            If = fft.fftshift(
                                fft.fft2(Ig)
                            )  # compute fft2 & shift the zero-frequency component to the center of the spectrum.
                            rotAvg = rotational_average_power_spectrum(If)

                            np.save(result_path, rotAvg)

# %% Finding Visually the Cut-Off Frequency between the Low and High Spatial frequencies for our Stimuli i.e. Difference between Energy Spectrum N=1 (resp. N=6) & N=4 (resp. N=24)
Numerosity_Range = {"subitizing": [1, 4], "estimation": [6, 24]}

for PS_range in PS_Ranges:

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{PS_range}_range.csv")
    ParkSpace_Description = _read_parkspace_log(PS_path)

    uDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    Rotational_Averages = {
        (obj_name, f"bg-{bg_idx}_alpha{bg_alpha}"): {
            N: [] for N in Numerosity_Range[PS_range]
        }
        for bg_idx, bg_alpha in Backgrounds
        for obj_name in Objects
    }

    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:

            for N, ID, FD in ParkSpace_Description:
                if N in Numerosity_Range[PS_range]:
                    for v_idx in Versions:

                        result_path = os.path.join(
                            uDir,
                            f"rAvg_Fourier_Power{SegMask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                        )

                        if os.path.isfile(result_path):

                            rAvg = np.load(result_path)
                            Rotational_Averages[
                                object_name, f"bg-{bg_idx}_alpha{bg_alpha}"
                            ][N].append(rAvg.copy())

    Difference_rAVG_by_N, Std_Diff_rAvg_by_N = [], []
    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:

            nMax = np.min(
                (
                    len(
                        Rotational_Averages[
                            object_name, f"bg-{bg_idx}_alpha{bg_alpha}"
                        ][Numerosity_Range[PS_range][0]]
                    ),
                    len(
                        Rotational_Averages[
                            object_name, f"bg-{bg_idx}_alpha{bg_alpha}"
                        ][Numerosity_Range[PS_range][-1]]
                    ),
                )
            )
            Diff_rAVG_by_N = []
            for i in range(nMax):

                AVG_rAVG_small_N = Rotational_Averages[
                    object_name, f"bg-{bg_idx}_alpha{bg_alpha}"
                ][Numerosity_Range[PS_range][0]][i]
                AVG_rAVG_high_N = Rotational_Averages[
                    object_name, f"bg-{bg_idx}_alpha{bg_alpha}"
                ][Numerosity_Range[PS_range][-1]][i]
                Diff_rAVG_by_N.append(AVG_rAVG_high_N - AVG_rAVG_small_N)

            if nMax > 0:
                Difference_rAVG_by_N.append(np.mean(Diff_rAVG_by_N, axis=0))
                Std_Diff_rAvg_by_N.append(np.std(Diff_rAVG_by_N, axis=0))

    np.save(
        os.path.join(
            uDir, f"Difference_rAvg_Fourier_Power{SegMask}_Small_Vs_High_Numerosity.npy"
        ),
        np.vstack(Difference_rAVG_by_N),
    )
    np.save(
        os.path.join(
            uDir, f"Std_Diff_rAvg_Fourier_Power{SegMask}_Small_Vs_High_Numerosity.npy"
        ),
        np.vstack(Std_Diff_rAvg_by_N),
    )

fcycle_min_IA, fcycle_max_IA = {
    "subitizing": ImgSize / 130,
    "estimation": ImgSize / 53,
}, {"subitizing": ImgSize / 260, "estimation": ImgSize / 106}
dcycle_min_IA, dcycle_max_IA = {
    "subitizing": ImgSize / (2 * 130),
    "estimation": ImgSize / (2 * 53),
}, {"subitizing": ImgSize / (2 * 260), "estimation": ImgSize / (2 * 106)}

## Plotting the Difference between Energy Spectrum N=1 (resp. N=6) & N=4 (resp. N=24)
for PS_range in PS_Ranges:

    uDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    Difference_rAVG = np.load(
        os.path.join(
            uDir, f"Difference_rAvg_Fourier_Power{SegMask}_Small_Vs_High_Numerosity.npy"
        )
    )
    Std_Diff_rAvg = np.load(
        os.path.join(
            uDir, f"Std_Diff_rAvg_Fourier_Power{SegMask}_Small_Vs_High_Numerosity.npy"
        )
    )

    plt.figure()
    plt.title(
        "["
        + PS_range[0].upper()
        + PS_range[1:]
        + " Range]\n"
        + "Difference of Energy between high vs. low Numerosity"
    )
    plt.xlabel("Spatial Frequency [cyles/image]", fontsize=12)
    plt.ylabel("Rotational Average Difference [a.u]", fontsize=12)
    for i in range(len(Difference_rAVG)):
        plt.loglog(range(1, ImgSize + 1), Difference_rAVG[i], alpha=0.1, color="gray")
    plt.loglog(range(1, ImgSize + 1), np.mean(Difference_rAVG, axis=0), "k")
    plt.loglog(
        [fcycle_min_IA[PS_range], fcycle_min_IA[PS_range]],
        [np.min(Difference_rAVG) * 0.9, np.max(Difference_rAVG) * 1.1],
        "g--",
        label="1 full cycle = smallest obj size",
    )
    plt.loglog(
        [fcycle_max_IA[PS_range], fcycle_max_IA[PS_range]],
        [np.min(Difference_rAVG) * 0.9, np.max(Difference_rAVG) * 1.1],
        "r--",
        label="1 full cycle = largest obj size",
    )
    plt.loglog(
        [dcycle_min_IA[PS_range], dcycle_min_IA[PS_range]],
        [np.min(Difference_rAVG) * 0.9, np.max(Difference_rAVG) * 1.1],
        "b--",
        label="1 demi cycle = smallest obj size",
    )
    plt.loglog(
        [dcycle_max_IA[PS_range], dcycle_max_IA[PS_range]],
        [np.min(Difference_rAVG) * 0.9, np.max(Difference_rAVG) * 1.1],
        "m--",
        label="1 demi cycle = largest obj size",
    )
    plt.legend()

## Remark : We found that the cut-off frequency between the low and high SF was :
## This cut-off frequency is hardcoded in the args.py file and is used in the next sections
## to compute the total energy in low or high spatial frequencies.

# %% Re-saving Low/High SF Energy in a Memory Efficient Way (as Aggregate Fourier Magnitude)
for PS_range in PS_Ranges:

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{PS_range}_range.csv")
    ParkSpace_Description = _read_parkspace_log(PS_path)

    size_area, spacing, numerosity, nPoints = (
        Size_Area[PS_range],
        Spacing[PS_range],
        Numerosity[PS_range],
        len(Numerosity[PS_range]),
    )

    uDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for object_name in Objects:
        for bg_idx, bg_alpha in Backgrounds:
            for v_idx in Versions:

                NRJ_High_SF_path = os.path.join(
                    uDir,
                    f"Energy_High_SF{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )
                NRJ_Low_SF_path = os.path.join(
                    uDir,
                    f"Energy_Low_SF{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                if not os.path.isfile(NRJ_High_SF_path):

                    try:
                        Energy_Low_SF, Energy_High_SF = np.zeros(
                            [nPoints, nPoints, nPoints]
                        ), np.zeros([nPoints, nPoints, nPoints])
                        for N, ID, FD in ParkSpace_Description:

                            base_stimulus_path = f"{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}"

                            Sp, SzA = _compute_park_space_point(N, ID, FD)
                            idx_N, idx_Sp, idx_SzA = (
                                numerosity.index(N),
                                spacing.index(Sp),
                                size_area.index(SzA),
                            )

                            rAvg = np.load(
                                os.path.join(
                                    uDir,
                                    f"rAvg_Fourier_Power{SegMask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                                )
                            )

                            Energy_Low_SF[idx_N, idx_Sp, idx_SzA] = np.sum(
                                rAvg[: cutoff_frequency[SegMask][PS_range]]
                            )
                            Energy_High_SF[idx_N, idx_Sp, idx_SzA] = np.sum(
                                rAvg[cutoff_frequency[SegMask][PS_range] :]
                            )

                        np.save(NRJ_High_SF_path, Energy_High_SF)
                        np.save(NRJ_Low_SF_path, Energy_Low_SF)

                    except:
                        print(object_name, bg_idx, bg_alpha, v_idx)

    ## Then remove the memory-costly useless file (if wanted, just uncomment this section)
    # for object_name in Objects:
    #     for bg_idx, bg_alpha in Backgrounds:
    #         for v_idx in Versions:
    #             for N, ID, FD in ParkSpace_Description:

    #                 old_path = uDir + f'rAvg_Fourier_Power{SegMask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy'
    #                 os.system(f'rm {old_path}')


# %% Use Ben Harvey's like Frequency Statistics i.e. Aggregate (over 1st Harmonic & Orientation) Fourier Magnitude
def rotational_aggregate_magnitude_spectrum(If):
    """
    Fct that computes the rotational aggregate magnitude spectrum from a 2D fft
    """

    ImgSize = len(If)
    rAvg = np.zeros(int(ImgSize / 2))

    ## Zero-Frequency Value of the Magnitude Spectrum (Power would be with **2)
    rAvg[0] = np.abs(If[int(ImgSize / 2), int(ImgSize / 2)])

    # Create meshgrid for x and y
    x, y = np.meshgrid(
        np.arange(-ImgSize / 2, ImgSize / 2), np.arange(-ImgSize / 2, ImgSize / 2)
    )

    # Convert to polar coordinates and round to integers
    _, radius = np.arctan2(y, x), np.round(np.sqrt(x**2 + y**2))

    ## All Remaining Frequency Rotational Aggregate Values
    for r in range(1, int(ImgSize / 2)):
        rAvg[r] = np.sum(np.abs(If[radius == r]))

    return rAvg


def _get_theoritical_first_harmonic(segmask_path):
    """
    Fct that computes the equivalent theoritical first harmonic frequency of
    a binarized stimulus, if all the white pixels (value equals 1) belonged to
    a set of equal-sized dots.
    """

    Stimulus_SegMask = np.load(
        segmask_path
    )  # values either 0 (background) or 1 (object)
    ImgSize = len(Stimulus_SegMask)

    ## Total area (in pixels) covered by all objects
    nPixels = Stimulus_SegMask.sum()
    ## Equivalent Radius of the average size covered by all objects
    Re = np.sqrt(nPixels / (N * np.pi))
    ## Theoritical first harmonic cut-off frequency for equivalent dot sets of radius Re (First order Bessel is FFT of dot)
    f1 = np.round(1.22 * (ImgSize / 2) / Re).astype(int)

    ## Remark: # ImgSize/2 correspond to maximum frequency, therefore f1 is the idx of the cut-off frequency

    return f1


## Computing Aggregate Fourier Magnitude for every Stimulus (PhotoRealistic & Binary Segmentation Mask)
for PS_range in PS_Ranges:

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{PS_range}_range.csv")
    ParkSpace_Description = _read_parkspace_log(PS_path)

    size_area, spacing, numerosity, nPoints = (
        Size_Area[PS_range],
        Spacing[PS_range],
        Numerosity[PS_range],
        len(Numerosity[PS_range]),
    )

    uDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    sDir = os.path.join(fDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    segDir = os.path.join(sDir, "Masks")
    imgDir = os.path.join(sDir, "Images")

    for object_name in Objects:
        for bg_idx, bg_alpha in Backgrounds:

            base_stimulus_dir = os.path.join(
                object_name, f"Bg-{bg_idx}_Alpha{bg_alpha}"
            )

            for v_idx in Versions:

                agg_mag_path = os.path.join(
                    uDir,
                    f"Aggregate_rAvg_Fourier_Magnitude{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                if not os.path.isfile(agg_mag_path):

                    Agg_Fourier_Mag = np.zeros([nPoints, nPoints, nPoints])
                    for N, ID, FD in ParkSpace_Description:

                        base_stimulus_path = f"{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}"

                        Sp, SzA = _compute_park_space_point(N, ID, FD)
                        idx_N, idx_Sp, idx_SzA = (
                            numerosity.index(N),
                            spacing.index(Sp),
                            size_area.index(SzA),
                        )

                        ## Finding first harmonic based on the Segmentation Mask Covered Area
                        segmask_path = os.path.join(
                            segDir, base_stimulus_dir, base_stimulus_path + ".npy"
                        )
                        if os.path.isfile(segmask_path):
                            f1 = _get_theoritical_first_harmonic(segmask_path)

                            ## Computing the Rotational Fourier Magnitude Average
                            if SegMask == "":

                                ## PhotoRealistic Stimulus
                                img_path = os.path.join(
                                    imgDir,
                                    base_stimulus_dir,
                                    base_stimulus_path + ".png",
                                )
                                I = io.imread(img_path)[
                                    :, :, :3
                                ]  # load image and discard the alpha channal I[:,:,3] containing only 255s !
                                Ig = np.mean(
                                    I, axis=2
                                )  # convert to gray scale : (height, width, depth)
                                If = fft.fftshift(fft.fft2(Ig))
                                rAvg = rotational_aggregate_magnitude_spectrum(If)
                                Agg_Fourier_Mag[idx_N, idx_Sp, idx_SzA] = np.sum(
                                    rAvg[:f1]
                                )

                            else:

                                ## Segmentation Mask
                                Ig = np.load(segmask_path)
                                If = fft.fftshift(fft.fft2(Ig))

                            rAvg = rotational_aggregate_magnitude_spectrum(If)
                            Agg_Fourier_Mag[idx_N, idx_Sp, idx_SzA] = np.sum(rAvg[:f1])

                    np.save(agg_mag_path, Agg_Fourier_Mag)
