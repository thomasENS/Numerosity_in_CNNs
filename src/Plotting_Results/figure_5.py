# %% Imports, Constants
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import scipy.stats as st
from args import (
    result_dir,
    saving_dir,
    PS_Ranges,
    nRanges,
    Layers,
    nLayers,
    Models,
    nModels,
)
from utils import simple_beeswarm2

# %% Fig 5B - Fine Generalization Comparison between Photorealistic & Simplified Stimuli (2-way ANOVA RM)
UseSegMasks = ["", "_Mask"]

width_bp = 0.35  # width boxplot
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
    for Model in Models[:3]
]
legend_vs = [
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Subitizing",
        markerfacecolor="#999999",
        markeredgecolor="k",
        markersize=7,
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Estimation",
        markerfacecolor="w",
        markeredgecolor="k",
        markersize=7,
    ),
]

Height = 3
fig, axs = plt.subplots(figsize=(2 * Height, 1.2 * Height))

idx_mask, boxplot_list_of_points = 0, []
for UseSegMask in UseSegMasks:

    tMAE, rMAE = [], []
    for k in range(nRanges):
        PS_range = PS_Ranges[k]
        uDir = os.path.join(
            result_dir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range"
        )

        MAE, STD = {Model: {Layer: 0 for Layer in Layers} for Model in Models}, {
            Model: {Layer: 0 for Layer in Layers} for Model in Models
        }
        for Model in Models:
            for Layer in Layers:

                Acc = np.load(
                    os.path.join(
                        uDir,
                        f"Finer_Grain_Generalisation{UseSegMask}_{Model}_{Layer}_Decoding_Log_N_Across_Objects_Score_MAE.npy",
                    )
                )
                Acc_OOD = np.concatenate(
                    (
                        Acc[np.triu_indices_from(Acc, k=1)],
                        Acc[np.tril_indices_from(Acc, k=-1)],
                    )
                )
                MAE[Model][Layer] = Acc_OOD.mean()
                STD[Model][Layer] = Acc_OOD.std()

        tMAE.append([MAE[Model][Layer] for Layer in Layers for Model in Models[:3]])
        rMAE.append([MAE[Model][Layer] for Layer in Layers for Model in Models[3:]])

    tMAE, rMAE = np.concatenate(tMAE), np.concatenate(rMAE)
    stat_level, pvalue = st.wilcoxon(tMAE - rMAE)
    print("Paired Wilcoxon", stat_level, pvalue)

    boxplot_list_of_points.append(deepcopy(tMAE))
    boxplot_list_of_points.append(deepcopy(rMAE))

    ## plot average distribution for trained vs. untrained
    x_tMAE, x_rMAE = simple_beeswarm2(tMAE, width=width_bp), simple_beeswarm2(
        rMAE, width=width_bp
    )
    for k in range(nRanges):
        for j in range(nModels // 2):
            for i in range(nLayers):
                n = k * (nModels // 2 * nLayers) + j * nLayers + i
                if n > 15:  ## Estimation Range
                    axs.plot(
                        x_tMAE[n] + 1 + idx_mask,
                        tMAE[n],
                        color=Colors[i],
                        marker=Markers[Models[j]],
                        mec="k",
                        ls="",
                        alpha=0.8,
                    )
                    axs.plot(
                        x_rMAE[n] + 2 + idx_mask,
                        rMAE[n],
                        color=Colors[i],
                        marker=Markers[Models[j + nModels // 2]],
                        mec="k",
                        ls="",
                        alpha=0.8,
                    )
                else:
                    axs.plot(
                        x_tMAE[n] + 1 + idx_mask,
                        tMAE[n],
                        color="w",
                        marker=Markers[Models[j]],
                        mec=Colors[i],
                        ls="",
                        alpha=0.8,
                    )
                    axs.plot(
                        x_rMAE[n] + 2 + idx_mask,
                        rMAE[n],
                        color="w",
                        marker=Markers[Models[j + nModels // 2]],
                        mec=Colors[i],
                        ls="",
                        alpha=0.8,
                    )

    ## Manual significance levels of the wilcoxon test
    h_level = 0.42 if idx_mask == 0 else 0.35
    axs.hlines(y=h_level, xmin=1 + idx_mask, xmax=2 + idx_mask, color="k", lw=1)
    axs.vlines(
        x=1 + idx_mask, ymin=h_level - 0.01, ymax=h_level + 0.0007, color="k", lw=1
    )
    axs.vlines(
        x=2 + idx_mask, ymin=h_level - 0.01, ymax=h_level + 0.0007, color="k", lw=1
    )
    if idx_mask == 0:
        axs.annotate(r"$\star$", xy=(1.35 + idx_mask, h_level + 0.005), fontsize=19)
        axs.annotate(r"$\star$", xy=(1.45 + idx_mask, h_level + 0.005), fontsize=19)
    else:
        axs.annotate(r"$\star$", xy=(1.425 + idx_mask, h_level + 0.005), fontsize=19)

    idx_mask += 2

axs.boxplot(
    boxplot_list_of_points,
    widths=2 * width_bp,
    medianprops={"ls": ""},
    meanprops={"ls": "-", "color": "k"},
    meanline=True,
    showmeans=True,
)

## Manual Rectangle to depict Photorealistic vs. Simplified Stimuli
rect_width, rect_height, rect_y0 = 1.9, 0.35, 0.11
rectPhotoRealistic = Rectangle(
    xy=(0.55, rect_y0),
    width=rect_width,
    height=rect_height,
    ls="--",
    lw=1,
    ec="k",
    fc="w",
)
axs.add_artist(rectPhotoRealistic)
rectSegMask = Rectangle(
    xy=(2.55, rect_y0),
    width=rect_width,
    height=rect_height,
    ls="--",
    lw=1,
    ec="k",
    fc="w",
)
axs.add_artist(rectSegMask)

## Manual significance levels of the wilcoxon test
h_level, xmin, xmax = 0.48, 1.5, 3.5
axs.hlines(y=h_level, xmin=xmin, xmax=xmax, color="k", lw=1)
axs.vlines(x=xmin, ymin=h_level - 0.01, ymax=h_level + 0.0007, color="k", lw=1)
axs.vlines(x=xmax, ymin=h_level - 0.01, ymax=h_level + 0.0007, color="k", lw=1)
axs.annotate(r"$\star$", xy=(2.32, h_level + 0.005), fontsize=19)
axs.annotate(r"$\star$", xy=(2.42, h_level + 0.005), fontsize=19)
axs.annotate(r"$\star$", xy=(2.52, h_level + 0.005), fontsize=19)

## Manual Annotation for Photorealistic vs. SegMask
props = dict(boxstyle="round", facecolor="#C0C0C0", alpha=0.5)
# place a text box in upper left in axes coords
axs.text(
    1.5,
    0.15,
    "Photorealistic Stimuli",
    fontsize=11.5,
    va="center",
    ha="center",
    bbox=props,
)
axs.text(
    3.5,
    0.415,
    "Segmentation Masks",
    fontsize=11.5,
    va="center",
    ha="center",
    bbox=props,
)

axs.set_ylim([0.09, 0.52])
axs.set_xlim([0.4, 4.55])

## Labels
axs.set_ylabel(f"Mean Absolute Error [Log]", fontsize=12)

## Manual labels
axs.set_xticks(
    [1, 2, 3, 4], ["Trained\nModels", "Untrained\nCounterparts"] * 2, fontsize=12
)

## Manual legends
lgd = axs.legend(
    handles=legend_models + legend_vs,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=5,
    columnspacing=0.3,
)
axs.legend(
    handles=legend_layers,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.122),
    ncol=5,
    columnspacing=0.6,
)
axs.add_artist(lgd)

plt.savefig(
    os.path.join(
        saving_dir, "Fine_Grained_Decoding_Across_Objects_Stimuli_Comparison.svg"
    ),
    dpi=300,
    bbox_inches="tight",
)
