# %% Imports & Constants
import numpy as np
import os
from args import dDir, Nasr_Numerosity, nFeatures, Models, Layer, nImg

nNumerosity = len(Nasr_Numerosity)

fDir = os.path.join(dDir, "CNN_Representations", "Dot_Patterns_Dataset")
sDir = os.path.join(dDir, "Decoding_Results", "Dot_Patterns_Dataset")

# %% Compute the Tuning Curves associated with the Number Selective Units
for Model in Models:

    Number_Selective_Units_Idx = np.load(
        os.path.join(
            sDir,
            f"{Model}_{Layer}_Nars_Number_Selective_Units_Reproduction_nImgs-{nImg}.npy",
        )
    )

    Features = np.zeros((nNumerosity, 3, nImg, nFeatures))
    for i in range(nImg):
        for n in range(nNumerosity):

            standard_set_Path = os.path.join(
                fDir,
                f"{Model}_{Layer}_Standard_Set_N-{Nasr_Numerosity[n]}_Stim-{1 + i}.npy",
            )
            control_set1_Path = os.path.join(
                fDir,
                f"{Model}_{Layer}_Control-1_Set_N-{Nasr_Numerosity[n]}_Stim-{1 + i}.npy",
            )
            control_set2_Path = os.path.join(
                fDir,
                f"{Model}_{Layer}_Control-2_Set_N-{Nasr_Numerosity[n]}_Stim-{1 + i}.npy",
            )

            Features[n, 0, i, :] = np.load(standard_set_Path).copy()
            Features[n, 1, i, :] = np.load(control_set1_Path).copy()
            Features[n, 2, i, :] = np.load(control_set2_Path).copy()

    Tuning_Curves = {N: np.zeros(nNumerosity) for N in Nasr_Numerosity}
    n_Prefered_Numerosity = {}

    Prefered_Numerosity, Averaged_Response = {N: [] for N in Nasr_Numerosity}, {
        idx: [] for idx in Number_Selective_Units_Idx
    }

    for idx in Number_Selective_Units_Idx:
        ## Computing averaged activations per numerosity for every Number Selective Unit.
        avg_activations_by_numerosity = np.mean(Features[:, :, :, idx], axis=(1, 2))
        prefered_numerosity_idx, activity_max, activity_min = (
            np.argmax(avg_activations_by_numerosity),
            np.max(avg_activations_by_numerosity),
            np.min(avg_activations_by_numerosity),
        )
        if activity_max != 0:
            Averaged_Response[idx] = (
                (avg_activations_by_numerosity - activity_min) / activity_max
            ).copy()
            Prefered_Numerosity[Nasr_Numerosity[prefered_numerosity_idx]].append(idx)

    # Directly on the Nars Dataset Stimuli
    n_Prefered_Numerosity = {N: len(Prefered_Numerosity[N]) for N in Nasr_Numerosity}
    for N in Nasr_Numerosity:
        for idx in Prefered_Numerosity[N]:
            Tuning_Curves[N] += Averaged_Response[idx]
        if n_Prefered_Numerosity[N] > 0:
            Tuning_Curves[N] /= n_Prefered_Numerosity[N]

    # Saving the Tuning Curves as an array (to prevent their time consuming computation each time)
    tuned_curves = np.zeros((nNumerosity, nNumerosity))
    for i in range(nNumerosity):
        tuned_curves[i] = Tuning_Curves[Nasr_Numerosity[i]].copy()

    np.save(
        os.path.join(
            sDir,
            f"{Model}_{Layer}_Nars_Number_Selective_Tuning_Curves_nImgs-{nImg}_normalised_between_0_1.npy",
        ),
        tuned_curves,
    )
