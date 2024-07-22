# %% Imports & Constants
import numpy as np
import skimage.io as io
import os
from utils import _read_parkspace_log, _compute_park_space_point, mask
from args import (
    mDir,
    Backgrounds,
    Objects,
    Versions,
    PS_Ranges,
    Numerosity,
    Spacing,
    Size_Area,
    SegMask,
)

rDir = os.path.join(mDir, "derivatives", "Low_Level_Statistics", "Luminance_Measures")
iDir = os.path.join(mDir, "data", "Stimuli", "Photorealistic_Dataset")

nBackgrounds, nObjects = len(Backgrounds), len(Objects)

# %% Compute the Mean and Standard Deviation of (grayscale) Luminance of our Stimuli
for ps_range in PS_Ranges:

    uDir = os.path.join(rDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")
    sDir = os.path.join(iDir, f"PS_{ps_range[0].upper() + ps_range[1:]}_Range")

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{ps_range}_range.csv")
    ParkSpace_Description = _read_parkspace_log(PS_path)

    size_area, spacing, numerosity, nPoints = (
        Size_Area[ps_range],
        Spacing[ps_range],
        Numerosity[ps_range],
        len(Numerosity[ps_range]),
    )

    for bg_idx, bg_alpha in Backgrounds:
        for object_name in Objects:
            for v_idx in Versions:

                Mean_Lum_path = os.path.join(
                    uDir,
                    f"Mean_Lum{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )
                Std_Lum_path = os.path.join(
                    uDir,
                    f"Std_Lum{SegMask}_{object_name}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                IgM_mean, IgM_std = np.zeros((nPoints, nPoints, nPoints)), np.zeros(
                    (nPoints, nPoints, nPoints)
                )
                for N, ID, FD in ParkSpace_Description:

                    Sp, SzA = _compute_park_space_point(N, ID, FD)
                    idx_N, idx_Sp, idx_SzA = (
                        numerosity.index(N),
                        spacing.index(Sp),
                        size_area.index(SzA),
                    )

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
                    IgM_mean[idx_N, idx_Sp, idx_SzA] = np.mean(IgM)
                    IgM_std[idx_N, idx_Sp, idx_SzA] = np.std(IgM)

                ## Saving Mean & Std Luminance in a memory effective manner
                np.save(Mean_Lum_path, IgM_mean)
                np.save(Std_Lum_path, IgM_std)
