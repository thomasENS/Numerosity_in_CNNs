# %% Define method to catch functions that take too long to run
import concurrent.futures
import time
from joblib import Parallel, delayed


def _evaluate_unit(k):

    print(time.asctime(time.localtime()), Model, SegMask, k)

    ## Creating the dataframe for statsmodels for each artificial unit
    data = {"a": a, "b": b, "c": c, "response": activated_Features[:, k]}
    df = pd.DataFrame(data)

    # Fit the 3-way ANOVA model
    try:
        model = ols(
            "response ~ C(a) + C(b) + C(c) + C(a):C(b) + C(a):C(c) + C(b):C(c) + C(a):C(b):C(c)",
            data=df,
        ).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        sigA = anova_table["PR(>F)"]["C(a)"] < sigThreshold
        sigB = anova_table["PR(>F)"]["C(b)"] < sigThreshold
        sigC = anova_table["PR(>F)"]["C(c)"] < sigThreshold
        sigAB = anova_table["PR(>F)"]["C(a):C(b)"] < sigThreshold
        sigAC = anova_table["PR(>F)"]["C(a):C(c)"] < sigThreshold
        sigABC = anova_table["PR(>F)"]["C(a):C(b):C(c)"] < sigThreshold

        ## Assess if given unit is numerosity-selective
        significance_unit = sigA and ~sigB and ~sigC and ~sigAB and ~sigAC and ~sigABC
        number_pvalue = anova_table["PR(>F)"]["C(a)"]
        size_pvalue = anova_table["PR(>F)"]["C(b)"]
        spacing_pvalue = anova_table["PR(>F)"]["C(c)"]
        hasConverged = True
    except:
        hasConverged = False
        significance_unit = False
        number_pvalue = 1
        spacing_pvalue = 1
        size_pvalue = 1

    return significance_unit, number_pvalue, size_pvalue, spacing_pvalue, hasConverged


def run_with_timeout(func, arg, timeout):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, arg)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print(
                f"Function with arg {arg} took too long to complete and was terminated."
            )
            return False, 1, 1, 1, False
        except Exception as e:
            print(f"An error occurred with args {arg}: {e}")
            return False, 1, 1, 1, False


def parallel_run_with_timeout(func, args, timeout):
    results = Parallel(n_jobs=24)(
        delayed(run_with_timeout)(func, arg, timeout) for arg in args
    )
    return results


# %% Imports & Constant
import numpy as np
import os, time
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from utils import _read_param_space_log
from args import mDir, dDir, Versions, Objects, Backgrounds, PS_Ranges, nFeatures

rDir = os.path.join(dDir, "Representations")
iDir = os.path.join(dDir, "Decoding_Results", "Revision")

Numerosities = {"subitizing": [1, 2, 3, 4], "estimation": [6, 10, 15, 24]}
nNumerosity = 4

Models = ["AlexNet", "Random_AlexNet"]  # 'Random_AlexNet'
Layers = ["Conv5"]
SegMasks = ["", "_Mask"]
Label_Models = {"AlexNet": "AlexNet", "Random_AlexNet": "Random AlexNet"}
Label_Stimuli = {"": "Photorealistic Stimuli", "_Mask": "Binarised Stimuli"}

Label_Variables = {"a": "Numerosity", "b": "SzA", "c": "Sp"}

sigThreshold = 0.01  # Same threshold as Nasr and Kim


## Useful Methods
def _compute_park_space_point(N, ID, FD):
    Sp = np.round(np.log10(np.pi**2 * FD**4 / N) * 10) / 10
    SzA = np.round(np.log10(ID**4 * N) * 10) / 10
    return Sp, SzA


def _load_representations_stimuli_dataset(model, layer, ps_range, v_idx=1, _mask=""):
    """
    Load all the representations extracted from AlexNet's [layer] of the stimuli pasted on the given [background] grouped by Object used to create the stimuli.
    """

    assert ps_range in [
        "subitizing",
        "estimation",
    ], 'range should be either "subitizing" or "estimation"'
    assert layer in ["Conv5"], "AlexNet layers are ConvX with X in {1 ... 5}"

    PS_path = os.path.join(mDir, "Stimulus_Creation", f"new_PS_{ps_range}_range.csv")
    ParkSpace_Description = _read_param_space_log(PS_path)

    Features, A, B, C = [], [], [], []
    for object_name, _ in Objects:
        for bg_idx, bg_alpha in Backgrounds:

            for N, ID, FD in ParkSpace_Description:

                features_path = os.path.join(
                    fDir,
                    f"{model}/{layer}/{model}_{layer}{_mask}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.npy",
                )

                if os.path.isfile(features_path):

                    Sp, SzA = _compute_park_space_point(N, ID, FD)

                    A.append(N)
                    B.append(SzA)
                    C.append(Sp)
                    Features.append(np.load(features_path))

    return np.array(Features), np.array(A), np.array(B), np.array(C)


# %% Execute the 3-way ANOVA on 3 independant categorical variables !
for PS_range in PS_Ranges:

    fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")
    sDir = os.path.join(iDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Model in Models:
        for Layer in Layers:
            for SegMask in SegMasks:
                for v_idx in Versions:

                    if not os.path.isfile(
                        os.path.join(
                            sDir,
                            f"{Model}_{Layer}_3Way_Number_Selective_Units_Uncorrected{SegMask}_Stimuli_v{v_idx}.npy",
                        )
                    ):

                        Features, a, b, c = _load_representations_stimuli_dataset(
                            Model, Layer, PS_range, v_idx=v_idx, _mask=SegMask
                        )

                        ## Removing Features that are non-activated across all stimuli
                        overall_activity = np.sum(Features, axis=0)
                        activated_Features = Features[:, ~(overall_activity == 0)]
                        activated_units_cumsum = np.cumsum(
                            overall_activity.astype(bool)
                        )
                        nActivatedUnits = overall_activity.astype(bool).sum()
                        activated_units_idx = np.array(
                            [
                                np.where(activated_units_cumsum == idx)[0][0]
                                for idx in range(1, nActivatedUnits + 1)
                            ]
                        )

                        results = parallel_run_with_timeout(
                            func=_evaluate_unit,
                            args=list(range(nActivatedUnits)),
                            timeout=None,
                        )

                        (
                            significance_units,
                            number_pvalues,
                            size_pvalues,
                            spacing_pvalues,
                            hasConverged,
                        ) = ([], [], [], [], [])
                        for result in results:
                            significance_units.append(result[0])
                            number_pvalues.append(result[1])
                            size_pvalues.append(result[2])
                            spacing_pvalues.append(result[3])
                            hasConverged.append(result[4])

                        significance_units, number_pvalues = np.array(
                            significance_units
                        ), np.array(number_pvalues)
                        significant_units_cumsum = np.cumsum(significance_units)
                        nSignificantUnits = significance_units.sum()
                        sigificant_units_idx = np.array(
                            [
                                np.where(significant_units_cumsum == idx)[0][0]
                                for idx in range(1, nSignificantUnits + 1)
                            ]
                        )

                        ## Saving uncorrected units in their original indexation that are significant according to the ANOVA test
                        np.save(
                            os.path.join(
                                sDir,
                                f"{Model}_{Layer}_3Way_Number_Selective_Units_Uncorrected{SegMask}_Stimuli_v{v_idx}.npy",
                            ),
                            activated_units_idx[sigificant_units_idx],
                        )

                        ## Saving pvalues
                        np.save(
                            os.path.join(
                                sDir,
                                f"{Model}_{Layer}_3Way_Pvalues_Number{SegMask}_Stimuli_v{v_idx}.npy",
                            ),
                            np.array(number_pvalues),
                        )
                        np.save(
                            os.path.join(
                                sDir,
                                f"{Model}_{Layer}_3Way_Pvalues_Size{SegMask}_Stimuli_v{v_idx}.npy",
                            ),
                            np.array(size_pvalues),
                        )
                        np.save(
                            os.path.join(
                                sDir,
                                f"{Model}_{Layer}_3Way_Pvalues_Spacing{SegMask}_Stimuli_v{v_idx}.npy",
                            ),
                            np.array(spacing_pvalues),
                        )
                        np.save(
                            os.path.join(
                                sDir,
                                f"{Model}_{Layer}_3Way_HasConverged{SegMask}_Stimuli_v{v_idx}.npy",
                            ),
                            np.array(hasConverged),
                        )

                        ## Keeping only units whose corrected p-value (Bonferroni correction) is lower than 0.05
                        corrected_units_idx = np.where(
                            number_pvalues[significance_units] * nFeatures < 0.05
                        )[0]

                        ## Going Back to the original Units indexation & saving them
                        kept_units_idx = activated_units_idx[
                            sigificant_units_idx[corrected_units_idx]
                        ]
                        np.save(
                            os.path.join(
                                sDir,
                                f"{Model}_{Layer}_3Way_Number_Selective_Units{SegMask}_Stimuli_v{v_idx}.npy",
                            ),
                            kept_units_idx,
                        )
