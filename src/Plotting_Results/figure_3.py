# %% Imports, Constants
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy.stats as st
from args import (
    result_dir,
    saving_dir,
    Chance_Level,
    PS_Ranges,
    Layers,
    Low_Level_Statistics,
    nRanges,
    nModels,
    nLayers,
)

## Decoded Modality in the coarse-grained generalization
Modality = "N"

# %% Fig 3B - Finer Grained Generalization for CNNs
rModels = ["Random_AlexNet", "Random_ResNet50", "Random_VGG16"]
tModels = ["AlexNet", "ResNet50", "VGG16"]
Models = tModels + rModels
trainingMode = ["Trained", "Untrained"]
cModels = {"Trained": tModels, "Untrained": rModels}
Modes = ["Objects", "Backgrounds"]
Cases = ["iid", "ood"]
lims = [np.log(0.075), np.log(4.2)]
Ticks = np.log(np.array([0.1, 0.2, Chance_Level[Modality], 1.25, 3]))

Markers = {
    "AlexNet": "o",
    "ResNet50": "s",
    "VGG16": "v",
    "Random_AlexNet": "o",
    "Random_ResNet50": "s",
    "Random_VGG16": "v",
}
Colors = {"Objects": "tab:green", "Backgrounds": "tab:orange"}
legend_vs = [Patch(facecolor=Colors[Mode], label=Mode) for Mode in Modes] + [
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Same",
        markerfacecolor="#999999",
        markeredgecolor="k",
        markersize=7,
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Generalization",
        markerfacecolor="w",
        markeredgecolor="k",
        markersize=7,
    ),
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
legend_chance = [
    Patch(
        facecolor="gray", alpha=0.7, edgecolor="k", label="MAE Better\n" + "than Chance"
    )
]

MAE, STD = {
    PS_range: {
        Mode: {Model: {Case: [] for Case in Cases} for Model in Models}
        for Mode in Modes
    }
    for PS_range in PS_Ranges
}, {
    PS_range: {
        Mode: {Model: {Case: [] for Case in Cases} for Model in Models}
        for Mode in Modes
    }
    for PS_range in PS_Ranges
}
for PS_range in PS_Ranges:
    uDir = os.path.join(result_dir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Mode in Modes:

        for Model in Models:

            data_ood, data_iid = [], []
            for Layer in Layers:
                Acc = np.load(
                    os.path.join(
                        uDir,
                        f"Finer_Grain_Generalisation_{Model}_{Layer}_Decoding_Log_N_Across_{Mode}_Score_MAE.npy",
                    )
                )
                data_ood.append(
                    np.concatenate(
                        (
                            Acc[np.triu_indices_from(Acc, k=1)],
                            Acc[np.tril_indices_from(Acc, k=-1)],
                        )
                    )
                )
                data_iid.append(np.diag(Acc))

            data = {"iid": data_iid, "ood": data_ood}
            for Case in Cases:

                MAE[PS_range][Mode][Model][Case] = [
                    np.mean(data[Case][i]) for i in range(len(data[Case]))
                ]
                STD[PS_range][Mode][Model][Case] = [
                    np.std(data[Case][i]) for i in range(len(data[Case]))
                ]

n = 0
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=False, figsize=(7, 3.6))
for tMode in trainingMode:
    n += 1

    plt.subplot(120 + n)
    ## Better than Chance Level Area
    axes[n - 1].fill_between(
        x=[np.log(0.07), np.log(Chance_Level[Modality])],
        y1=np.log(0.07),
        y2=np.log(Chance_Level[Modality]),
        color="gray",
        alpha=0.4,
    )
    axes[n - 1].fill_between(
        x=[np.log(0.07), np.log(Chance_Level[Modality])],
        y1=np.log(Chance_Level[Modality]),
        y2=np.log(4.3),
        color="gray",
        alpha=0.2,
    )
    axes[n - 1].fill_between(
        x=[np.log(Chance_Level[Modality]), np.log(4.3)],
        y1=np.log(0.07),
        y2=np.log(Chance_Level[Modality]),
        color="gray",
        alpha=0.2,
    )

    for Model in cModels[tMode]:
        for Mode in Modes:
            for Case in Cases:
                ix, iy = np.argmin(MAE["estimation"][Mode][Model][Case]), np.argmin(
                    MAE["subitizing"][Mode][Model][Case]
                )
                x, y = np.log(MAE["estimation"][Mode][Model][Case][ix]), np.log(
                    MAE["subitizing"][Mode][Model][Case][iy]
                )
                xerr = np.log(
                    (
                        MAE["estimation"][Mode][Model][Case][ix]
                        + STD["estimation"][Mode][Model][Case][ix] / 4
                    )
                    / (
                        MAE["estimation"][Mode][Model][Case][ix]
                        - STD["estimation"][Mode][Model][Case][ix] / 4
                    )
                )
                yerr = np.log(
                    (
                        MAE["subitizing"][Mode][Model][Case][iy]
                        + STD["subitizing"][Mode][Model][Case][iy] / 4
                    )
                    / (
                        MAE["subitizing"][Mode][Model][Case][iy]
                        - STD["subitizing"][Mode][Model][Case][iy] / 4
                    )
                )
                if Case == "iid":
                    plt.errorbar(
                        x=x,
                        y=y,
                        xerr=xerr,
                        yerr=yerr,
                        markeredgewidth=0.8,
                        ms=8,
                        marker=Markers[Model],
                        markeredgecolor="k",
                        color=Colors[Mode],
                        ecolor=Colors[Mode],
                        elinewidth=0.8,
                    )
                else:
                    plt.errorbar(
                        x=x,
                        y=y,
                        xerr=xerr,
                        yerr=yerr,
                        markeredgewidth=0.8,
                        ms=8,
                        marker=Markers[Model],
                        markeredgecolor=Colors[Mode],
                        color="w",
                        ecolor=Colors[Mode],
                        elinewidth=0.8,
                    )

## Manual Labels
plt.subplots_adjust(wspace=0.05)
axes[0].set_xticks(Ticks, [f"{xT:.2f}" for xT in np.exp(Ticks)])
axes[1].set_xticks(Ticks, [f"{xT:.2f}" for xT in np.exp(Ticks)])
axes[0].set_yticks(Ticks, [f"{yT:.2f}" for yT in np.exp(Ticks)])
axes[0].set_ylabel("MAE [Log] - Subitizing Range", fontsize=12)
axes[0].set_title("Trained Models", fontsize=12)
axes[1].set_title("Untrained Counterparts", fontsize=12)
fig.supxlabel("MAE [Log] - Estimation Range", fontsize=12)

## Separation between Better vs. Worse than Chance Level Region
for n in range(2):
    axes[n].plot(
        [np.log(0.07), np.log(Chance_Level[Modality])],
        [np.log(Chance_Level[Modality])] * 2,
        "k--",
        linewidth=1.0,
    )
    axes[n].plot(
        [np.log(Chance_Level[Modality])] * 2,
        [np.log(0.07), np.log(Chance_Level[Modality])],
        "k--",
        linewidth=1.0,
    )
    axes[n].plot(
        [np.log(Chance_Level[Modality]), np.log(4.3)],
        [np.log(Chance_Level[Modality])] * 2,
        "--",
        color="gray",
        linewidth=1.0,
    )
    axes[n].plot(
        [np.log(Chance_Level[Modality])] * 2,
        [np.log(Chance_Level[Modality]), np.log(4.3)],
        "--",
        color="gray",
        linewidth=1.0,
    )
    axes[n].set_xlim(lims)
    axes[n].set_ylim(lims)

# legend_chance = axes[]
legend_models = axes[1].legend(
    handles=legend_models, loc="upper left", bbox_to_anchor=(0, 1), fontsize=11
)
axes[1].legend(
    handles=legend_chance, loc="upper left", bbox_to_anchor=(0, 0.72), fontsize=9.5
)
axes[1].add_artist(legend_models)
axes[0].legend(handles=legend_vs, loc="upper left", bbox_to_anchor=(0, 1), fontsize=11)

plt.savefig(
    os.path.join(saving_dir, "Fine_Grained_Numerosity_Predictions_across_Models.svg"),
    dpi=300,
    bbox_inches="tight",
)

# %% Fig 3C - Fine Grained Generalisation using Low-Level Statistics as Inputs
Models = [
    ["Mean_Lum"],
    ["Std_Lum"],
    ["Agg_Mag_Fourier"],
    ["Energy_High_SF"],
    ["Dist_Texture"],
    ["Dist_Complexity"],
]
Label_Models = [
    "Mean Luminance",
    "Std Luminance",
    "Aggregate Fourier\nMagnitude",
    "Energy High SF",
    "Texture Similarity",
    "Image Complexity",
]
Modes = ["Objects", "Backgrounds"]
Cases = ["iid", "ood"]
lims = [np.log(0.075), np.log(4.2)]
Ticks = np.log(np.array([0.1, 0.2, Chance_Level[Modality], 1.25, 3]))
markersize = 7

Markers = {
    "Mean_Lum": "o",
    "Std_Lum": "H",
    "Energy_Low_SF": "v",
    "Agg_Mag_Fourier": "v",
    "Energy_High_SF": "^",
    "Dist_Texture": "s",
    "Dist_Complexity": "D",
    "_".join(Low_Level_Statistics): "*",
}
Colors = {"Objects": "tab:green", "Backgrounds": "tab:orange"}
legend_vs = [Patch(facecolor=Colors[Mode], label=Mode) for Mode in Modes] + [
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Same",
        markerfacecolor="#999999",
        markeredgecolor="k",
        markersize=7,
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Generalization",
        markerfacecolor="w",
        markeredgecolor="k",
        markersize=7,
    ),
]
legend_models_v1 = [
    Line2D(
        [0],
        [0],
        marker=Markers["_".join(Models[i])],
        color="w",
        label=Label_Models[i],
        markerfacecolor="k",
        markeredgecolor="k",
        markersize=9,
    )
    for i in range(nModels // 2)
]
legend_models_v2 = [
    Line2D(
        [0],
        [0],
        marker=Markers["_".join(Models[i])],
        color="w",
        label=Label_Models[i],
        markerfacecolor="k",
        markeredgecolor="k",
        markersize=9,
    )
    for i in range(nModels // 2, nModels)
]
legend_all_stats = [
    Line2D(
        [0],
        [0],
        marker=Markers["_".join(Low_Level_Statistics)],
        color="w",
        label="All Low-Level Statistics",
        markerfacecolor="k",
        markeredgecolor="k",
        markersize=9,
    )
]

MAE, STD = {
    PS_range: {
        Mode: {"_".join(Model): {Case: [] for Case in Cases} for Model in Models}
        for Mode in Modes
    }
    for PS_range in PS_Ranges
}, {
    PS_range: {
        Mode: {"_".join(Model): {Case: [] for Case in Cases} for Model in Models}
        for Mode in Modes
    }
    for PS_range in PS_Ranges
}
for PS_range in PS_Ranges:
    uDir = os.path.join(result_dir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Mode in Modes:

        for Model in Models:

            Acc = np.load(
                os.path.join(
                    uDir,
                    f'UFG_Decoding_Log_N_Using_{"_".join(Model)}_Statistics_v3_Across_{Mode}_Score_MAE.npy',
                )
            )
            data_ood = np.concatenate(
                (
                    Acc[np.triu_indices_from(Acc, k=1)],
                    Acc[np.tril_indices_from(Acc, k=-1)],
                )
            )
            data_iid = np.diag(Acc)

            data = {"iid": data_iid, "ood": data_ood}
            for Case in Cases:

                MAE[PS_range][Mode]["_".join(Model)][Case] = np.mean(data[Case])
                STD[PS_range][Mode]["_".join(Model)][Case] = np.std(data[Case])


fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=False, figsize=(7, 3.6))

## Better than Chance Level Area & Separation between Better vs. Worse than Chance Level Region
for ax in axes:
    ax.fill_between(
        x=[np.log(0.07), np.log(Chance_Level[Modality])],
        y1=np.log(0.07),
        y2=np.log(Chance_Level[Modality]),
        color="gray",
        alpha=0.4,
    )
    ax.fill_between(
        x=[np.log(0.07), np.log(Chance_Level[Modality])],
        y1=np.log(Chance_Level[Modality]),
        y2=np.log(5),
        color="gray",
        alpha=0.2,
    )
    ax.fill_between(
        x=[np.log(Chance_Level[Modality]), np.log(5)],
        y1=np.log(0.07),
        y2=np.log(Chance_Level[Modality]),
        color="gray",
        alpha=0.2,
    )
    ax.plot(
        [np.log(0.07), np.log(Chance_Level[Modality])],
        [np.log(Chance_Level[Modality])] * 2,
        "k--",
        linewidth=1.0,
    )
    ax.plot(
        [np.log(Chance_Level[Modality])] * 2,
        [np.log(0.07), np.log(Chance_Level[Modality])],
        "k--",
        linewidth=1.0,
    )
    ax.plot(
        [np.log(Chance_Level[Modality]), np.log(5)],
        [np.log(Chance_Level[Modality])] * 2,
        "--",
        color="gray",
        linewidth=1.0,
    )
    ax.plot(
        [np.log(Chance_Level[Modality])] * 2,
        [np.log(Chance_Level[Modality]), np.log(5)],
        "--",
        color="gray",
        linewidth=1.0,
    )

## axes[0] : Everything Link to the Individual Low-Level Statistics
for Model in Models:
    for Mode in Modes:
        for Case in Cases:
            x, y = np.log(MAE["estimation"][Mode]["_".join(Model)][Case]), np.log(
                MAE["subitizing"][Mode]["_".join(Model)][Case]
            )
            xerr = np.log(
                (
                    MAE["estimation"][Mode]["_".join(Model)][Case]
                    + STD["estimation"][Mode]["_".join(Model)][Case] / 4
                )
                / (
                    MAE["estimation"][Mode]["_".join(Model)][Case]
                    - STD["estimation"][Mode]["_".join(Model)][Case] / 4
                )
            )
            yerr = np.log(
                (
                    MAE["subitizing"][Mode]["_".join(Model)][Case]
                    + STD["subitizing"][Mode]["_".join(Model)][Case] / 4
                )
                / (
                    MAE["subitizing"][Mode]["_".join(Model)][Case]
                    - STD["subitizing"][Mode]["_".join(Model)][Case] / 4
                )
            )
            if Case == "iid":
                axes[0].errorbar(
                    x=x,
                    y=y,
                    xerr=xerr,
                    yerr=yerr,
                    markeredgewidth=0.8,
                    ms=markersize,
                    marker=Markers["_".join(Model)],
                    markeredgecolor="k",
                    color=Colors[Mode],
                    ecolor=Colors[Mode],
                    elinewidth=0.8,
                )
            else:
                axes[0].errorbar(
                    x=x,
                    y=y,
                    xerr=xerr / 1.2,
                    yerr=yerr / 1.2,
                    markeredgewidth=0.8,
                    ms=markersize,
                    marker=Markers["_".join(Model)],
                    markeredgecolor=Colors[Mode],
                    color="w",
                    ecolor=Colors[Mode],
                    elinewidth=0.8,
                )

## axes[1] : Generalisation of All the Summary Statistics concatenated
All_Stats_Models = [Low_Level_Statistics]

MAE, STD = {
    PS_range: {
        Mode: {
            "_".join(Model): {Case: [] for Case in Cases} for Model in All_Stats_Models
        }
        for Mode in Modes
    }
    for PS_range in PS_Ranges
}, {
    PS_range: {
        Mode: {
            "_".join(Model): {Case: [] for Case in Cases} for Model in All_Stats_Models
        }
        for Mode in Modes
    }
    for PS_range in PS_Ranges
}
for PS_range in PS_Ranges:
    uDir = os.path.join(result_dir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

    for Mode in Modes:

        for Model in All_Stats_Models:

            Acc = np.load(
                os.path.join(
                    uDir,
                    f'UFG_Decoding_Log_N_Using_{"_".join(Model)}_Statistics_v3_Across_{Mode}_Score_MAE.npy',
                )
            )
            data_ood = np.concatenate(
                (
                    Acc[np.triu_indices_from(Acc, k=1)],
                    Acc[np.tril_indices_from(Acc, k=-1)],
                )
            )
            data_iid = np.diag(Acc)

            data = {"iid": data_iid, "ood": data_ood}
            for Case in Cases:

                MAE[PS_range][Mode]["_".join(Model)][Case] = np.mean(data[Case])
                STD[PS_range][Mode]["_".join(Model)][Case] = np.std(data[Case])

for Model in All_Stats_Models:
    for Mode in Modes:
        for Case in Cases:
            x, y = np.log(MAE["estimation"][Mode]["_".join(Model)][Case]), np.log(
                MAE["subitizing"][Mode]["_".join(Model)][Case]
            )
            xerr = np.log(
                (
                    MAE["estimation"][Mode]["_".join(Model)][Case]
                    + STD["estimation"][Mode]["_".join(Model)][Case] / 4
                )
                / (
                    MAE["estimation"][Mode]["_".join(Model)][Case]
                    - STD["estimation"][Mode]["_".join(Model)][Case] / 4
                )
            )
            yerr = np.log(
                (
                    MAE["subitizing"][Mode]["_".join(Model)][Case]
                    + STD["subitizing"][Mode]["_".join(Model)][Case] / 4
                )
                / (
                    MAE["subitizing"][Mode]["_".join(Model)][Case]
                    - STD["subitizing"][Mode]["_".join(Model)][Case] / 4
                )
            )
            if Case == "iid":
                axes[1].errorbar(
                    x=x,
                    y=y,
                    xerr=xerr,
                    yerr=yerr,
                    markeredgewidth=0.8,
                    ms=11,
                    marker=Markers["_".join(Model)],
                    markeredgecolor="k",
                    color=Colors[Mode],
                    ecolor=Colors[Mode],
                    elinewidth=0.8,
                )
            else:
                axes[1].errorbar(
                    x=x,
                    y=y,
                    xerr=xerr / 2,
                    yerr=yerr / 2,
                    markeredgewidth=0.8,
                    ms=11,
                    marker=Markers["_".join(Model)],
                    markeredgecolor=Colors[Mode],
                    color="w",
                    ecolor=Colors[Mode],
                    elinewidth=0.8,
                )

## Manual Ticks & Axes Label
plt.subplots_adjust(wspace=0.05)
axes[0].set_xticks(Ticks, [f"{xT:.2f}" for xT in np.exp(Ticks)])
axes[1].set_xticks(Ticks, [f"{xT:.2f}" for xT in np.exp(Ticks)])
axes[0].set_yticks(Ticks, [f"{yT:.2f}" for yT in np.exp(Ticks)])
axes[0].set_ylabel("MAE [Log] - Subitizing Range", fontsize=12)
fig.supxlabel("MAE [Log] - Estimation Range", fontsize=12)
axes[0].set_xlim(lims)
axes[1].set_xlim(lims)
plt.ylim(lims)

## Manual Legend
legend_models = axes[0].legend(
    handles=legend_models_v2, loc="lower right", bbox_to_anchor=(1, 0), fontsize=9.5
)
axes[0].legend(
    handles=legend_models_v1, loc="upper left", bbox_to_anchor=(0, 1), fontsize=9.5
)
axes[0].add_artist(legend_models)
lgd_vs = axes[1].legend(
    handles=legend_vs, loc="upper left", bbox_to_anchor=(0, 1), fontsize=10.5
)
axes[1].legend(
    handles=legend_all_stats, loc="upper left", bbox_to_anchor=(0, 0.65), fontsize=9.5
)
axes[1].add_artist(lgd_vs)

plt.savefig(
    os.path.join(
        saving_dir, "Fine_Grained_Numerosity_Predictions_using_Low_Level_Statistics.svg"
    ),
    dpi=300,
    bbox_inches="tight",
)

# %% Fig 3E - Explained Variance of the Fine-Grained Numerosity Predictions by the Low-Level Statistics
UseSegMask = ""  # '_Mask'
Cases = ["Objects", "Backgrounds"]

rModels = ["Random_AlexNet", "Random_ResNet50", "Random_VGG16"]
tModels = ["AlexNet", "ResNet50", "VGG16"]
Models = tModels + rModels
nModels = len(Models)

Colors = ["#0095EF", "#3C50B1", "#6A38B3", "#F31D64", "#FE433C"]
width = 0.165
OffSet = [-2 * width, -width, 0, width, 2 * width]
legend_layers = [
    Patch(facecolor=Colors[i], label=f"Conv{i+1}") for i in range(len(Colors))
]
legend_vs = [
    Patch(facecolor="#DDDDDD", edgecolor="k", label="Trained Models"),
    Patch(facecolor="#848484", edgecolor="k", label="Untrained Counterparts"),
]

## Args for GridSpec Ratio
Height = 6
widths, heights = [4.5 * Height, 1 * Height], [1 * Height, 1 * Height]
gs_kw = dict(width_ratios=widths, height_ratios=heights)

for Case in Cases:

    fig, axs = plt.subplots(ncols=2, nrows=2, sharey=True, gridspec_kw=gs_kw)

    for k in range(nRanges):

        PS_range = PS_Ranges[k]
        sDir = os.path.join(
            result_dir + f"PS_{PS_range[0].upper() + PS_range[1:]}_Range"
        )

        Explained_Variance_Mean = {
            Model: {Layer: -1 for Layer in Layers} for Model in Models
        }
        Explained_Variance_Std = {
            Model: {Layer: -1 for Layer in Layers} for Model in Models
        }
        for Model in Models:
            for Layer in Layers:

                results_path = (
                    os.path.join(
                        sDir, f"FGG_OLS_{Model}_{Layer}_Decoding_Log_{Modality}"
                    )
                    if Model != "RawPixels"
                    else os.path.join(sDir, f"FGG_{Model}_Decoding_Log_{Modality}")
                )

                exp_var_path = (
                    results_path
                    + f"_Across_{Case}_LLS_Explained_Variance{UseSegMask}.npy"
                )
                Explained_Variance = np.load(exp_var_path)
                Explained_Variance_Mean[Model][Layer] = np.mean(
                    Explained_Variance, axis=0
                )
                Explained_Variance_Std[Model][Layer] = np.std(
                    Explained_Variance, axis=0
                )

        tMean, rMean, tStd, rStd = [], [], [], []
        ## Computing Average Explained Variance & Std : Trained Vs. Untrained Models
        for Layer in Layers:
            for tModel in tModels:
                tMean.append(np.sum(Explained_Variance_Mean[tModel][Layer]))
                tStd.append(np.mean(Explained_Variance_Std[tModel][Layer]))
            for rModel in rModels:
                rMean.append(np.sum(Explained_Variance_Mean[rModel][Layer]))
                rStd.append(np.mean(Explained_Variance_Std[rModel][Layer]))

        t_mean, t_std = np.mean(tMean), np.mean(tStd)
        r_mean, r_std = np.mean(rMean), np.mean(rStd)

        # statistic testing using paired version of Wilcoxon signed rank test
        stat_level, pvalue = st.wilcoxon(np.array(tMean) - np.array(rMean))
        print(Case, PS_range, "Paired Wilcoxon", stat_level, pvalue)

        ## Plot Explained Variance Per Model
        axs[k, 0].plot([-4 * width, nModels - 1 + 4 * width], [0, 0], "k--")
        axs[k, 0].set_ylim([-0.1, 1])

        i = 0
        for Model in Models:
            j = 0
            for Layer in Layers:
                Mean, Std = np.sum(Explained_Variance_Mean[Model][Layer]), np.mean(
                    Explained_Variance_Std[Model][Layer]
                )
                axs[k, 0].bar(
                    x=i + OffSet[j],
                    height=Mean,
                    color=Colors[j],
                    width=width,
                    edgecolor="k",
                )
                axs[k, 0].vlines(
                    i + OffSet[j], Mean - Std / 2, Mean + Std / 2, colors="k"
                )
                j += 1
            i += 1

        ## Plot Explained Variance Average across Trained vs. Untrained Models
        axs[k, 1].bar(
            x=OffSet[0], height=t_mean, color="#DDDDDD", width=2 * width, edgecolor="k"
        )
        axs[k, 1].vlines(OffSet[0], t_mean - t_std / 2, t_mean + t_std / 2, colors="k")
        axs[k, 1].bar(
            x=OffSet[-1], height=r_mean, color="#848484", width=2 * width, edgecolor="k"
        )
        axs[k, 1].vlines(OffSet[-1], r_mean - r_std / 2, r_mean + t_std / 2, colors="k")
        axs[k, 1].plot([OffSet[0] - 4 * width, OffSet[-1] + 4 * width], [0, 0], "k--")

        ## Manual Annotations of each subplot
        axs[k, 0].annotate(
            f"Generalization across {Case} - [{PS_range[0].upper() + PS_range[1:]} Range]",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            va="top",
            ha="center",
            fontsize=9,
        )

    ## Manual Ticks & Axes Labels
    for k in range(nRanges):
        axs[k, 0].set_xlim([-4 * width, nModels - 1 + 4 * width])
        axs[k, 1].set_xlim([OffSet[0] - 3 * width, OffSet[-1] + 3 * width])

    axs[0, 0].set_xticks([], [])
    axs[0, 1].set_xticks([], [])
    axs[1, 0].set_xticks(
        range(nModels),
        tModels + ["\n".join(model.split("_")) for model in rModels],
        fontsize=9,
    )
    axs[1, 1].set_xticks([0], ["Model\nAverages"], fontsize=9)
    fig.supylabel(r"Explained Variance [$r^2$]", fontsize=12)
    plt.subplots_adjust(hspace=0.08, wspace=0.04)

    ## Manual Legends
    axs[0, 0].legend(
        handles=legend_layers,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.22),
        ncol=nLayers,
        columnspacing=0.7,
    )
    axs[0, 1].legend(
        handles=legend_vs,
        loc="lower left",
        bbox_to_anchor=(-4.1, 1.0),
        ncol=nLayers,
        columnspacing=0.7,
    )

    ## Manual significance levels
    h_level, lw = 0.55, 1
    axs[0, 1].hlines(y=h_level, xmin=OffSet[0], xmax=OffSet[-1], color="k", lw=lw)
    axs[0, 1].vlines(
        x=OffSet[0], ymin=h_level - 0.05, ymax=h_level + 0.0055, color="k", lw=lw
    )
    axs[0, 1].vlines(
        x=OffSet[-1], ymin=h_level - 0.05, ymax=h_level + 0.0055, color="k", lw=lw
    )
    if Case == "Objects":
        axs[0, 1].annotate(r"$\star$", xy=(-0.12, h_level + 0.02), fontsize=14)
    else:
        axs[0, 1].annotate(r"$\star$", xy=(-0.05, h_level + 0.02), fontsize=14)
        axs[0, 1].annotate(r"$\star$", xy=(-0.21, h_level + 0.02), fontsize=14)

    h_level, lw = 0.75, 1
    axs[1, 1].hlines(y=h_level, xmin=OffSet[0], xmax=OffSet[-1], color="k", lw=lw)
    axs[1, 1].vlines(
        x=OffSet[0], ymin=h_level - 0.05, ymax=h_level + 0.0055, color="k", lw=lw
    )
    axs[1, 1].vlines(
        x=OffSet[-1], ymin=h_level - 0.05, ymax=h_level + 0.0055, color="k", lw=lw
    )
    if Case == "Backgrounds":
        axs[1, 1].annotate(r"$\star$", xy=(-0.12, h_level + 0.02), fontsize=14)
    else:
        axs[1, 1].annotate(r"$\star$", xy=(-0.05, h_level + 0.02), fontsize=14)
        axs[1, 1].annotate(r"$\star$", xy=(-0.21, h_level + 0.02), fontsize=14)

    plt.savefig(
        os.path.join(
            saving_dir,
            f"Variance_Decomposition_Low_Level_Stats_Generalisation_Across_{Case}.svg",
        ),
        dpi=300,
        bbox_inches="tight",
    )
