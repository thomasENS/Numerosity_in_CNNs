# %% Imports, Constants
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from args import (nDir, saving_dir, Percentages)

## Information about the networks used in this analysis
Models = ['AlexNet', 'Random_AlexNet']; nModels = len(Models)
Layer = 'Conv5'

## Decoding chance-level performance when using shared numerosity between
## our photo-realistic dataset and the dot-arrays dataset
Chance_Level = 0.8502993454155389

# %% Fig 4A - Tuning Curves on Dot-Patterns Stimuli from Number Selective Neurons

Tuning_Curves = np.load(os.path.join(nDir, 'AlexNet_Conv5_Nars_Number_Selective_Tuning_Curves_nImgs-900_normalised_between_0_1.npy'))
Tuning_Curves /= np.max(Tuning_Curves, axis=0)
Numerosity = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]; nNumerosity = len(Numerosity)

Height = 3
nCols, nRows = 2, 2

curves_idx = np.round(np.linspace(0, nNumerosity-1, nCols*nRows)).astype(int)

fig, axs = plt.subplots(ncols=nCols, nrows=nRows, sharey=True, figsize=(2.5*Height, 2*Height))

for i in range(nRows):
    for j in range(nCols):
        axs[i,j].spines[['right', 'top']].set_visible(False)
        axs[i,j].set_title(fr'$PN={Numerosity[curves_idx[i*nCols + j]]}$', color="#FF4D4D", y=.95, fontsize=12)
        for k in range(nNumerosity):
            if k == curves_idx[i*nCols + j]:
                axs[i,j].plot(Numerosity, Tuning_Curves[k], color="#FF4D4D", lw=2.4, zorder=10)
                axs[i,j].scatter(Numerosity, Tuning_Curves[k], fc="w", ec="#FF4D4D", s=60, lw=2.4, zorder=12)  
            else:
                axs[i,j].plot(Numerosity, Tuning_Curves[k], color="#BFBFBF", lw=1.5)

for j in range(nCols):
    axs[0,j].set_xticks([], [])

fig.supylabel('Normalized Response [a.u.]', x=.04, fontsize=15)
fig.supxlabel('Numerosity', y=.04, fontsize=15)
plt.subplots_adjust(hspace=0.1, wspace=.04)

plt.savefig(
    os.path.join(saving_dir, 'AlexNet_Conv5_Nars_Number_Selective_Tuning_Curves_nImgs-900.svg'),
    dpi=300,
    bbox_inches='tight'
)

# %% Fig 4C : Coarse-Grained Decoding on PhotoRealistic Stimuli based on Nasr Units + Population Code

Markers = ['o', 's']
Height = 5
alpha = .2
ft_text, ft_label, ft_legend = 14, 18, 15

width, dw = .2, 1.3
x = np.array([-dw*width, 0, dw*width])
Colors = ['#FF4D4D', '#69A7D3', '#999999']

## Create mosaic subplots
fig, axs = plt.subplot_mosaic('AAAB', sharey=True, figsize=(2.7*Height, Height))

## Plot of the Coarse-Grained Decoding Performances for AlexNet & Random AlexNet
for k in range(nModels):

    results_path = os.path.join(nDir, f'Full_Generalisation_{Models[k]}_{Layer}_Decoding_Log_N_from_Number_Selective_Units_Uncorrected_v3_nImgs-900')
    MAE, STD = np.load(results_path + '_Score_MAE.npy'), np.load(results_path + '_Score_Std.npy') 

    axs['A'].bar(x[0]+k, MAE[0], yerr=STD[0]/2, width=width, color=Colors[0])
    axs['A'].bar(x[1]+k, MAE[1], yerr=STD[1]/2, width=width, edgecolor=Colors[1], linewidth=2, color='w')
    axs['A'].bar(x[2]+k, MAE[2], yerr=STD[2]/2, width=width, color=Colors[1]) 

## Plot of the Coarse-Grained Decoding Performances for RawPixels (Baseline)
axs['A'].bar(x[0]+2, MAE[-1], yerr=STD[-1]/2, width=width, color=Colors[2])

## Manual legend
lgd = [
    Patch(facecolor=Colors[2], label='Image pixel values'),
    Patch(facecolor=Colors[0], label=r'Number neurons ($N_s$ units)'),
    Patch(facecolor='w', edgecolor=Colors[1],  label=r'Non-selective neurons ($N_{ns}$ units)'),
    Patch(facecolor=Colors[1], label=r'All Units except Number neurons'),
]
axs['A'].legend(handles=lgd, loc='lower left', bbox_to_anchor=(-.01, 1.01), ncol=2, columnspacing=.6, fontsize=ft_legend)

## Manual ticks & labels
axs['A'].plot([-1, 3], [Chance_Level]*2, '--', color=Colors[2])
axs['A'].set_xlim([x[0]-1.7*width, 2+x[0]+1.7*width])
axs['A'].set_ylim([0, 1.6])
axs['A'].set_ylabel('Mean Absolute Error [Log]', fontsize=ft_label)
axs['A'].set_xticks([0, 1, x[0]+2], ['AlexNet', 'Random\nAlexNet', 'Baseline'], fontsize=15)

## Population Code vs. Few Tuned Units ?
for i in range(nModels):

    results_path = os.path.join(nDir, f'Coarse_Grained_Generalisation_{Models[i]}_{Layer}_Decoding_Log_N_Redundancy_Assessement_Photorealistic_Stimuli')

    mae_randomly_units  = np.load(results_path + '_Score_MAE_Randomly_Chosen_Units.npy')
    std_randomly_units  = np.load(results_path + '_Score_Std_Randomly_Chosen_Units.npy')

    axs['B'].plot(Percentages*100, mae_randomly_units, '-', marker=Markers[i], markerfacecolor='w', markeredgecolor=Colors[1], color=Colors[1])
    axs['B'].fill_between(Percentages*100, mae_randomly_units-std_randomly_units/2, mae_randomly_units+std_randomly_units/2, alpha=alpha, color=Colors[1])

## Manual Ticks & Scale
for i in range(2):
    axs['B'].plot([0, 120], [Chance_Level]*2, '--', color=Colors[2])
    axs['B'].set_xlim([.2, 120])
    axs['B'].set_xscale('log')
    axs['B'].set_xticks([1, 10, 100], [1, 10, 100], fontsize=12)

plt.subplots_adjust(wspace=0.05)

## Manual Labels & Titles
axs['B'].set_xlabel('Proportion of Units [%]', fontsize=15)

## Manual legend
ms = 8
legend_vs = [Line2D([0], [0], marker=Markers[0], color='w', label='AlexNet', markerfacecolor='w', markeredgecolor='k',  markersize=ms),
            Line2D([0], [0], marker=Markers[1], color='w', label='Random AlexNet', markerfacecolor='w', markeredgecolor='k', markersize=ms)]

axs['B'].legend(handles=legend_vs, loc='lower left', bbox_to_anchor=(-.025, 1.01), fontsize=ft_legend)

plt.savefig(
    os.path.join(saving_dir, 'Coarse_Grained_Decoding_from_Nasr_Uncorrected_Units_nImgs-900_&_Population_Code_vs_Tuned_Units_Assessment.svg'),
    dpi=300,
    bbox_inches='tight'
)