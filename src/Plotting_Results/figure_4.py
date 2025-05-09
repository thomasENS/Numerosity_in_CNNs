# %% Imports, Constants
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Line2D
from args import mDir, saving_dir

dDir = os.path.join(mDir, "DNN_Analysis")
lDir = os.path.join(dDir, "Decoding_Results")

PS_Ranges = ["subitizing", "estimation"]
Layers = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]
UseSegMask = ""  # Corresponds to photorealistic stimuli

# %% Summary Quantity Bias Analysis
rModels = ["Random_AlexNet", "Random_ResNet50", "Random_VGG16"]
tModels = ["AlexNet", "ResNet50", "VGG16"]
Models = tModels + rModels
Cases = ["Trained", "Untrained"]
cModels = {"Trained": tModels, "Untrained": rModels}

Markers = {
    "AlexNet": "o",
    "ResNet50": "s",
    "VGG16": "v",
    "Random_AlexNet": "o",
    "Random_ResNet50": "s",
    "Random_VGG16": "v",
}
Colors = ["#0095EF", "#3C50B1", "#6A38B3", "#F31D64", "#FE433C"]
legend_layers = [
    Patch(facecolor=Colors[i], label=f"Conv{i+1}") for i in range(len(Colors))
]
legend_models = [
    Line2D(
        [0],
        [0],
        marker=Markers[Model],
        color="w",
        label=Model,
        markerfacecolor="k",
        markersize=9,
    )
    for Model in tModels
]

xlim, ylim = 0.6, 0.6
tx, ty = xlim, ylim
ft = 8
ms, ew = 4, 7 / 10

for PS_range in PS_Ranges:

    sDir = os.path.join(lDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    fig, axs = plt.subplots(
        nrows=2, ncols=3, sharey=True, sharex=False, figsize=(6.5, 4)
    )

    ## Manual Plot
    for i in range(2):
        for j in range(3):
            axs[i, j].spines[["right", "top"]].set_visible(False)
            axs[i, j].plot(
                [-xlim, xlim], [-ylim, ylim], color="#CCCCCC", linewidth=2 * ew
            )
            axs[i, j].plot(
                [-xlim, xlim], [ylim, -ylim], color="#CCCCCC", linewidth=2 * ew
            )
            axs[i, j].plot([-xlim, xlim], [0, 0], color="#CCCCCC", linewidth=2 * ew)
            axs[i, j].plot([0, 0], [-ylim, ylim], color="#CCCCCC", linewidth=2 * ew)
    axs[0, 0].plot(
        [-xlim / 3, xlim / 3], [-ylim, ylim], color="#CCCCCC", linewidth=2 * ew
    )
    axs[1, 0].plot(
        [-xlim / 3, xlim / 3], [-ylim, ylim], color="#CCCCCC", linewidth=2 * ew
    )

    ## Manual Legend
    plt.subplots_adjust(wspace=0.225, hspace=0.075)
    axs[0, 2].legend(
        handles=legend_models,
        loc="lower right",
        bbox_to_anchor=(0.45, 1.2),
        ncol=3,
        columnspacing=0.6,
    )
    axs[0, 0].legend(
        handles=legend_layers,
        loc="lower left",
        bbox_to_anchor=(-0.05, 1),
        ncol=5,
        columnspacing=0.8,
    )

    ## Manual labels
    for i in range(2):
        axs[i, 0].set_ylim((-ty, ty))
        for j in range(3):
            axs[i, j].set_xlim((-tx, tx))
        axs[i, 0].set_yticks([-0.5, 0, 0.5], [r"$- 0.5$", r"$0.0$", r"$0.5$"])
        axs[i, 0].set_ylabel(r"$\beta_{N}$")
        axs[i, 1].set_ylabel(r"$\beta_{N}$")
        axs[i, 2].set_ylabel(r"$\beta_{Sp}$")

    axs[1, 0].set_xlabel(r"$\beta_{SzA}$")
    axs[1, 1].set_xlabel(r"$\beta_{Sp}$")
    axs[1, 2].set_xlabel(r"$\beta_{SzA}$")

    for j in range(3):
        axs[0, j].set_xticks([], [])

    for k in range(2):

        for Model in cModels[Cases[k]]:

            for i, Layer in enumerate(Layers):

                results_path = os.path.join(
                    sDir,
                    f"FGG_Objects_{Model}_{Layer}_Quantity_Biases_Assessement_Train_Test_Same_Object_Beta_Weights{UseSegMask}.npy",
                )
                Beta_Weights_AVG = np.load(results_path).mean(axis=0)  # (N, SzA, Sp)
                Beta_Weights_STD = np.load(results_path).std(axis=0)

                ## (SzA, N)
                x, xerr, y, yerr = (
                    Beta_Weights_AVG[1],
                    Beta_Weights_STD[1],
                    Beta_Weights_AVG[0],
                    Beta_Weights_STD[0],
                )
                axs[k, 0].errorbar(
                    x=x,
                    y=y,
                    xerr=xerr,
                    yerr=yerr,
                    markeredgewidth=ew,
                    ms=ms,
                    marker=Markers[Model],
                    markeredgecolor="k",
                    color=Colors[i],
                    ecolor=Colors[i],
                    elinewidth=ew,
                )

                ## (Sp, N)
                x, xerr, y, yerr = (
                    Beta_Weights_AVG[2],
                    Beta_Weights_STD[2],
                    Beta_Weights_AVG[0],
                    Beta_Weights_STD[0],
                )
                axs[k, 1].errorbar(
                    x=x,
                    y=y,
                    xerr=xerr,
                    yerr=yerr,
                    markeredgewidth=ew,
                    ms=ms,
                    marker=Markers[Model],
                    markeredgecolor="k",
                    color=Colors[i],
                    ecolor=Colors[i],
                    elinewidth=ew,
                )

                ## (SzA, Sp)
                x, xerr, y, yerr = (
                    Beta_Weights_AVG[1],
                    Beta_Weights_STD[1],
                    Beta_Weights_AVG[2],
                    Beta_Weights_STD[2],
                )
                axs[k, 2].errorbar(
                    x=x,
                    y=y,
                    xerr=xerr,
                    yerr=yerr,
                    markeredgewidth=ew,
                    ms=ms,
                    marker=Markers[Model],
                    markeredgecolor="k",
                    color=Colors[i],
                    ecolor=Colors[i],
                    elinewidth=ew,
                )

    plt.savefig(
        os.path.join(
            saving_dir,
            f"Summary_Quantity_Analysis{UseSegMask}_{PS_range[0].upper() + PS_range[1:]}_Range.svg",
        ),
        dpi=600,
        bbox_inches="tight",
    )
