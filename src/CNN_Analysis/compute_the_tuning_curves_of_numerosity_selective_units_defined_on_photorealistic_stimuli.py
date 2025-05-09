# %% Imports & Constant
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import colorsys
import matplotlib.colors
from utils import _read_param_space_log
from args import (
    mDir,
    dDir,
    Versions,
    nFeatures,
    PS_Ranges,
    Objects,
    Backgrounds,
)


rDir = os.path.join(dDir, "Representations")
iDir = os.path.join(dDir, "Decoding_Results", "Revision")
uDir = os.path.join(mDir, "Figures")

Numerosities = {"subitizing": [1, 2, 3, 4], "estimation": [6, 10, 15, 24]}
nNumerosity = 4


Models = ["AlexNet", "Random_AlexNet"]  # 'Random_AlexNet', 'AlexNet',
Layers = ["Conv5"]
SegMasks = ["", "_Mask"]
Label_Models = {"AlexNet": "AlexNet", "Random_AlexNet": "Random AlexNet"}
Label_Stimuli = {"": "Photorealistic Stimuli", "_Mask": "Binarised Stimuli"}

Correction = "_Uncorrected"
Mode = "3Way"


## Useful Methods
def _load_representations_stimuli_dataset_per_Numerosity(
    model, layer, ps_range, v_idx=1, _mask=""
):
    """
    Load all the representations extracted from AlexNet's [layer] of the stimuli pasted on the given [background] grouped by Object used to create the stimuli.
    """

    assert ps_range in [
        "subitizing",
        "estimation",
    ], 'range should be either "subitizing" or "estimation"'
    assert layer in ["Conv5"], "AlexNet layers are ConvX with X in {1 ... 5}"

    Numerosity = Numerosities[ps_range]

    PS_path = os.path.join(mDir, "Stimulus_Creation", f"new_PS_{ps_range}_range.csv")
    ParkSpace_Description = _read_param_space_log(PS_path)

    Features = [[] for _ in range(nNumerosity)]
    for object_name, category in Objects:
        for bg_idx, bg_alpha in Backgrounds:

            for N, ID, FD in ParkSpace_Description:

                features_path = os.path.join(
                    fDir,
                    f"{model}/{layer}/{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                if os.path.isfile(features_path):

                    idx_N = Numerosity.index(N)
                    Features[idx_N].append(np.load(features_path))

    return np.array(Features)


def _finding_preferred_numerosity(Number_Selective_Units_Idx, Features):

    Prefered_Numerosity, Averaged_Response = {N: [] for N in Numerosity}, np.zeros(
        (nNumerosity, nFeatures)
    )

    ## Computing averaged activations per Numerosity for every Number Selective Unit.
    avg_activations_by_numerosity = np.mean(Features[:, :, :], axis=1)
    prefered_numerosity = np.argmax(avg_activations_by_numerosity, axis=0)
    activity_max = np.max(avg_activations_by_numerosity, axis=0)
    activity_min = np.min(avg_activations_by_numerosity, axis=0)

    for idx in range(nFeatures):
        if (activity_max[idx] - activity_min[idx]) != 0:
            a, b = 1 / (activity_max[idx] - activity_min[idx]), activity_min[idx] / (
                activity_min[idx] - activity_max[idx]
            )
            Averaged_Response[:, idx] = (
                a * avg_activations_by_numerosity[:, idx] + b
            ).copy()
        if idx in Number_Selective_Units_Idx:
            Prefered_Numerosity[Numerosity[prefered_numerosity[idx]]].append(idx)

    return Prefered_Numerosity, Averaged_Response


def _compute_tuning_curves(Prefered_Numerosity, Averaged_Response):

    Tuning_Curves = {N: np.zeros(nNumerosity) for N in Numerosity}
    n_Prefered_Numerosity = {N: len(Prefered_Numerosity[N]) for N in Numerosity}

    for N in Numerosity:
        for idx in Prefered_Numerosity[N]:
            Tuning_Curves[N] += Averaged_Response[:, idx]
        if n_Prefered_Numerosity[N] > 0:
            Tuning_Curves[N] /= n_Prefered_Numerosity[N]

    return Tuning_Curves


def _plot_tuning_curves(
    Tuning_Curves, idx_preferred_numerosity, idx_avg_response, saveplot=False
):

    tuned_curves = np.zeros((nNumerosity, nNumerosity))
    for i in range(nNumerosity):
        tuned_curves[i] = Tuning_Curves[Numerosity[i]].copy()
    np.save(
        os.path.join(
            sDir,
            f"{Model}_{Layer}_{Mode}_Tuning_Curves{Correction}{SegMask}_Normalised_{idx_preferred_numerosity}{idx_avg_response}.npy",
        ),
        tuned_curves,
    )

    plt.figure()
    plt.title(
        f"Approximate Tuning Curve for Number Selective Units\n"
        + f"[{Label_Models[Model]}/{Layer}] - {PS_range[0].upper() + PS_range[1:]} Range\n"
        + f"{Label_Stimuli[SegMask]}"
    )
    plt.ylabel("Normalised Response [a.u.]", fontsize=12)
    plt.xlabel("Numerosity", fontsize=12)
    c = 0
    for N in Numerosity:
        plt.plot(
            Numerosity, Tuning_Curves[N], "o--", label=str(N), color=Colors_Gradient[c]
        )
        c += 1
    plt.legend(title="Preferred Numerosity", loc="upper left", bbox_to_anchor=(1, 1))
    if saveplot:
        plt.savefig(
            os.path.join(
                uDir,
                f"{Model}_{Layer}_{Mode}_Approximate_Tuning_Curve_for_Number_Selective_Units{Correction}{SegMask}_{PS_range[0].upper() + PS_range[1:]}_Range_{idx_preferred_numerosity}{idx_avg_response}.pdf",
            ),
            dpi=150,
            bbox_inches="tight",
        )


def colors_gradient(Colors, luminance_levels):
    """
    Transform a list of RGB colors (defined in Hex)
    to a list of list of RGB Colors with every sublist containing
    the gradient of previous base-color for different luminance level (in hls space)
    """
    Colors_Gradients = []
    for color in Colors:
        r, g, b = matplotlib.colors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        gradient = []
        for luminance_level in luminance_levels:
            gradient.append(colorsys.hls_to_rgb(h, luminance_level, s))
        Colors_Gradients.append(gradient.copy())
    return Colors_Gradients


Colors = ["#0094FF"]
luminance_levels = np.linspace(0.9, 0.1, nNumerosity)
Colors_Gradient = colors_gradient(Colors, luminance_levels)[0]

# %% Compute the Tuning Curves associated with the Number Selective Units
for PS_range in PS_Ranges[::-1]:

    fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    sDir = os.path.join(iDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    Numerosity = Numerosities[PS_range]

    for SegMask in SegMasks:
        for Model in Models:
            for Layer in Layers:

                if not os.path.isfile(
                    os.path.join(
                        sDir,
                        f"{Model}_{Layer}_{Mode}_Distribution_Preferred_Numerosities{Correction}{SegMask}.npy",
                    )
                ):

                    Features_1 = _load_representations_stimuli_dataset_per_Numerosity(
                        Model, Layer, PS_range, v_idx=1, _mask=SegMask
                    )
                    Features_2 = _load_representations_stimuli_dataset_per_Numerosity(
                        Model, Layer, PS_range, v_idx=2, _mask=SegMask
                    )

                    Number_Selective_Units_Idx_1 = np.load(
                        os.path.join(
                            sDir,
                            f"{Model}_{Layer}_{Mode}_Number_Selective_Units{Correction}{SegMask}_Stimuli_v1.npy",
                        )
                    )
                    Number_Selective_Units_Idx_2 = np.load(
                        os.path.join(
                            sDir,
                            f"{Model}_{Layer}_{Mode}_Number_Selective_Units{Correction}{SegMask}_Stimuli_v2.npy",
                        )
                    )

                    ## Finding preferred numerosity on the version of the stimuli the units where selected on
                    Prefered_Numerosity_1, Averaged_Response_1 = (
                        _finding_preferred_numerosity(
                            Number_Selective_Units_Idx_1, Features_1
                        )
                    )
                    Prefered_Numerosity_2, Averaged_Response_2 = (
                        _finding_preferred_numerosity(
                            Number_Selective_Units_Idx_2, Features_2
                        )
                    )

                    # Saving the Distribution of Preferred Numerosity
                    nPreferredN = [
                        [len(Prefered_Numerosity_1[N]) for N in Numerosity],
                        [len(Prefered_Numerosity_2[N]) for N in Numerosity],
                    ]
                    np.save(
                        os.path.join(
                            sDir,
                            f"{Model}_{Layer}_{Mode}_Distribution_Preferred_Numerosities{Correction}{SegMask}.npy",
                        ),
                        np.array(nPreferredN),
                    )

                    # Computing Tuning Curves (in a double-dipping totally circular fashion 11/22 or in the other version of the stimuli 12/21)
                    Tuning_Curves_11 = _compute_tuning_curves(
                        Prefered_Numerosity_1, Averaged_Response_1
                    )
                    Tuning_Curves_12 = _compute_tuning_curves(
                        Prefered_Numerosity_1, Averaged_Response_2
                    )
                    Tuning_Curves_21 = _compute_tuning_curves(
                        Prefered_Numerosity_2, Averaged_Response_1
                    )
                    Tuning_Curves_22 = _compute_tuning_curves(
                        Prefered_Numerosity_2, Averaged_Response_2
                    )

                    # Saving the Tuning Curves as an array (to prevent their time consuming computation each time)
                    _plot_tuning_curves(
                        Tuning_Curves_11,
                        idx_preferred_numerosity=1,
                        idx_avg_response=1,
                        saveplot=False,
                    )
                    _plot_tuning_curves(
                        Tuning_Curves_12,
                        idx_preferred_numerosity=1,
                        idx_avg_response=2,
                        saveplot=False,
                    )
                    _plot_tuning_curves(
                        Tuning_Curves_21,
                        idx_preferred_numerosity=2,
                        idx_avg_response=1,
                        saveplot=False,
                    )
                    _plot_tuning_curves(
                        Tuning_Curves_22,
                        idx_preferred_numerosity=2,
                        idx_avg_response=2,
                        saveplot=False,
                    )

# %% Evaluate whether the numerosity selective units are shared between stimulus datasets
for Model in Models:

    for idx_range, PS_range in enumerate(PS_Ranges):

        sDir = os.path.join(iDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

        Idx_Mask_1 = np.load(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Mask_Stimuli_v1.npy",
            )
        )
        Idx_Photorealistic_1 = np.load(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Stimuli_v1.npy",
            )
        )

        Idx_Mask_2 = np.load(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Mask_Stimuli_v2.npy",
            )
        )
        Idx_Photorealistic_2 = np.load(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Stimuli_v2.npy",
            )
        )

        ## Saving the np.array indices of numerosity-selective units as .mat for Evelyn
        scipy.io.savemat(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Mask_Stimuli_v1.mat",
            ),
            {"array": Idx_Mask_1},
        )
        scipy.io.savemat(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Mask_Stimuli_v2.mat",
            ),
            {"array": Idx_Mask_2},
        )
        scipy.io.savemat(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Stimuli_v1.mat",
            ),
            {"array": Idx_Photorealistic_1},
        )
        scipy.io.savemat(
            os.path.join(
                sDir,
                f"{Model}_Conv5_{Mode}_Number_Selective_Units{Correction}_Stimuli_v2.mat",
            ),
            {"array": Idx_Photorealistic_2},
        )

        cpt_1 = 0
        for idx in Idx_Photorealistic_1:
            if idx in Idx_Mask_1:
                cpt_1 += 1

        cpt_2 = 0
        for idx in Idx_Photorealistic_2:
            if idx in Idx_Mask_2:
                cpt_2 += 1

        cpt_sp = 0
        for idx in Idx_Photorealistic_2:
            if idx in Idx_Photorealistic_1:
                cpt_sp += 1

        cpt_sb = 0
        for idx in Idx_Mask_2:
            if idx in Idx_Mask_1:
                cpt_sb += 1

        print(
            f"For the numerosity selective units of {Model}, we found:\n"
            + "For the 1st version of the stimuli:\n"
            + f"Ns = {len(Idx_Photorealistic_1)} ({100*len(Idx_Photorealistic_1)/nFeatures:.2f}%) for the Photorealistic Stimuli\n"
            + f"Ns = {len(Idx_Mask_1)} ({100*len(Idx_Mask_1)/nFeatures:.2f}%) for the Binarised Stimuli\n"
            + f"{cpt_1} units were shared between the two stimulus datasets.\n"
            + "For the 2nd version of the stimuli:\n"
            + f"Ns = {len(Idx_Photorealistic_2)} ({100*len(Idx_Photorealistic_2)/nFeatures:.2f}%) for the Photorealistic Stimuli\n"
            + f"Ns = {len(Idx_Mask_2)} ({100*len(Idx_Mask_2)/nFeatures:.2f}%) for the Binarised Stimuli\n"
            + f"{cpt_2} units were shared between the two stimulus datasets.\n"
            + "In addition:\n"
            + f"{cpt_sp} ({100*cpt_sp/min(len(Idx_Photorealistic_1), len(Idx_Photorealistic_2)):.2f}%) units were shared between the two versions of the photorealistic stimuli\n"
            + f"{cpt_sb} ({100*cpt_sb/min(len(Idx_Mask_1), len(Idx_Mask_2)):.2f}%) units were shared between the two versions of the binarised stimuli\n"
        )
# %% Characterisation of the selected units
threshold = 0.01

Label_SegMask = {"": "Photorealistic", "_Mask": "Binary"}

for Model in Models:

    for idx_range, PS_range in enumerate(PS_Ranges):

        sDir = os.path.join(iDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

        for SegMask in SegMasks:

            for v_idx in Versions:

                number_pvalues = np.load(
                    os.path.join(
                        sDir,
                        f"{Model}_Conv5_3Way_Pvalues_Number{SegMask}_Stimuli_v{v_idx}.npy",
                    )
                )
                size_pvalues = np.load(
                    os.path.join(
                        sDir,
                        f"{Model}_Conv5_3Way_Pvalues_Size{SegMask}_Stimuli_v{v_idx}.npy",
                    )
                )
                spacing_pvalues = np.load(
                    os.path.join(
                        sDir,
                        f"{Model}_Conv5_3Way_Pvalues_Spacing{SegMask}_Stimuli_v{v_idx}.npy",
                    )
                )
                nHasConverged = np.load(
                    os.path.join(
                        sDir,
                        f"{Model}_Conv5_3Way_HasConverged{SegMask}_Stimuli_v{v_idx}.npy",
                    )
                ).sum()
                nNonZeros = len(number_pvalues)

                nNs = len(
                    np.load(
                        os.path.join(
                            sDir,
                            f"{Model}_Conv5_3Way_Number_Selective_Units{Correction}{SegMask}_Stimuli_v{v_idx}.npy",
                        )
                    )
                )
                nNum = len(number_pvalues[number_pvalues < threshold])
                nSzA = len(size_pvalues[size_pvalues < threshold])
                nSp = len(spacing_pvalues[spacing_pvalues < threshold])

                print(
                    f"For {Model}, we found:\n"
                    + f"For the version {v_idx} of the {Label_SegMask[SegMask]} Stimuli - {PS_range[0].upper() + PS_range[1:]} Range:\n"
                    + f"{nNs} ({100*nNs/nFeatures:.2f}% of all units) numerosity-selective units\n"
                    + f"{nNum} ({100*nNum/nNonZeros:.2f}% of non-zero units) showed a main effect of Numerosity\n"
                    + f"{nSzA} ({100*nSzA/nNonZeros:.2f}% of non-zero units) showed a main effect of Size\n"
                    + f"{nSp} ({100*nSp/nNonZeros:.2f}% of non-zero units) showed a main effect of Spacing\n"
                )
