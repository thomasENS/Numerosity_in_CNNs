# %% Imports & Constants
import os
import numpy as np
import skimage.io as io
from scipy.stats import weibull_min
from scipy.ndimage import gaussian_gradient_magnitude
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
    nPoints,
    Sigma,
)

wDir = os.path.join(
    mDir, "derivatives", "Low_Level_Statistics", "Local_Contrast_Measures"
)
sDir = os.path.join(mDir, "data", "Stimuli", "Photorealistic_Dataset")


# %% Useful Methods to Estimate Weibull Parameters
def _estimate_weibull_params(img_path, sigma):
    """
    Fct that MLE estimate of the weibull parameters (beta, gamma) of the given img
    """

    gamma_opt, beta_opt, area_error, bad_fit = 1, 0, 0, True

    try:
        I = io.imread(img_path)[
            :, :, :3
        ]  # load image and discard the alpha channal I[:,:,3] containing only 255s !
        Ig = np.mean(I, axis=2)  # convert to gray scale : (height, width, depth)

        edges = gaussian_gradient_magnitude(Ig, sigma=sigma)  # compute edges
        edges = edges[
            mask != 0
        ]  # mask the outer circle of the edges/contrasts map which are identically zero
        y, x = np.histogram(
            edges, bins=256, density=True
        )  # compute density of contrasts
        dx = x[1:] - x[:-1]

        try:
            gamma_opt, mu, beta_opt = weibull_min.fit(
                edges
            )  # Careful : Do not force floc=np.min(edges) otherwise it will fail to fit !
            fitted_weibull = weibull_min(c=gamma_opt, loc=mu, scale=beta_opt)
            area_error = np.sum(np.abs(y - fitted_weibull.pdf(x[1:])) * dx)
        except ValueError:
            pass

    except Exception:
        # Catches all the exceptions (dangerous behaviour can happen. Do not do that.)
        pass

    if gamma_opt != 1:
        bad_fit = False

    return gamma_opt, beta_opt, area_error, bad_fit


# %% Compute the Weibull parameters
for PS_range in PS_Ranges:

    iDir = os.path.join(sDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    iDir = os.path.join(iDir, "Images")
    wDir = os.path.join(wDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    PS_path = os.path.join(mDir, "src", "Stimulus_Creation", f"PS_{PS_range}_range.csv")
    ParkSpace_Description = _read_parkspace_log(PS_path)

    for object_name in Objects:
        for bg_idx, bg_alpha in Backgrounds:

            bg_name = f"bg-{bg_idx}_alpha{bg_alpha}"

            Gamma_path = os.path.join(
                wDir, f"Sigma-{Sigma}_{object_name}_{bg_name}_Gamma.npy"
            )
            Beta_path = os.path.join(
                wDir, f"Sigma-{Sigma}_{object_name}_{bg_name}_Beta.npy"
            )
            Area_path = os.path.join(
                wDir, f"Sigma-{Sigma}_{object_name}_{bg_name}_Area.npy"
            )

            if not os.path.isfile(
                Gamma_path
            ):  # Do not Estimate Params on Imgs that were already processed !

                Gamma_AVG, Beta_AVG, Area_AVG = (
                    np.zeros((nPoints, nPoints, nPoints)),
                    np.zeros((nPoints, nPoints, nPoints)),
                    np.zeros((nPoints, nPoints, nPoints)),
                )
                for N, ID, FD in ParkSpace_Description:

                    Sp, SzA = _compute_park_space_point(N, ID, FD)
                    idx_N, idx_Sp, idx_SzA = (
                        Numerosity[PS_range].index(N),
                        Spacing[PS_range].index(Sp),
                        Size_Area[PS_range].index(SzA),
                    )

                    Gamma, Beta, Area = [], [], []
                    for v_idx in Versions:

                        img_path = os.path.join(
                            iDir,
                            object_name,
                            f"Bg-{bg_idx}_Alpha{bg_alpha}",
                            f"{object_name}-{N}_ID-{ID}_FD-{FD}_{bg_name}_version-{v_idx}.png",
                        )

                        gamma_opt, beta_opt, area_error, bad_fit = (
                            _estimate_weibull_params(img_path, Sigma)
                        )

                        if not bad_fit:
                            Gamma.append(gamma_opt)
                            Beta.append(beta_opt)
                            Area.append(area_error)

                    if (
                        len(Gamma) > 0
                    ):  # If only bad_fit happened, zero will remain in the AVG arrays at the given coordinate.
                        Gamma_AVG[idx_N, idx_Sp, idx_SzA] = np.mean(Gamma)
                        Beta_AVG[idx_N, idx_Sp, idx_SzA] = np.mean(Beta)
                        Area_AVG[idx_N, idx_Sp, idx_SzA] = np.mean(Area)

                np.save(Gamma_path, Gamma_AVG)
                np.save(Beta_path, Beta_AVG)
                np.save(Area_path, Area_AVG)
