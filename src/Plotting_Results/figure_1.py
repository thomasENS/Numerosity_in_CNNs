# %% Imports, Constants
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from args import (mDir, saving_dir)
from utils import (_load_data, beatiful_violin_plot)

wDir = os.path.join(mDir, 'derivatives', 'Low_Level_Statistics', 'Local_Contrast_Measures')
fDir = os.path.join(mDir, 'derivatives', 'Low_Level_Statistics', 'Frequency_Measures')
lDir = os.path.join(mDir, 'derivatives', 'Low_Level_Statistics', 'Luminance_Measures')

dirBase  = [lDir, lDir, fDir, fDir, wDir, wDir]
DistPath = [
    'UFG_Mean_Luminance.npy',
    'UFG_Std_Luminance.npy',
    'UFG_Aggregate_Fourier_Magnitude.npy',
    'UFG_Total_Rotational_Average_Energy_in_High_Spatial_Frequency.npy',
    'UFG_Distance_Along_Texture_Axis.npy',
    'UFG_Distance_Along_Complexity_Axis.npy'
]
DiffPath = [
    'UFG_Difference_Mean_Luminance_High_minus_Small_Numerosity.npy',
    'UFG_Difference_Std_Luminance_High_minus_Small_Numerosity.npy',
    'UFG_Difference_Aggregate_Fourier_Magnitude_High_minus_Small_Numerosity.npy',
    'UFG_Difference_Total_Rotational_Average_Energy_High_minus_Small_Numerosity_high_SF.npy',
    'UFG_Difference_Dist_Along_Texture_Axis_High_minus_Small_Numerosity.npy',
    'UFG_Difference_Dist_Along_Complexity_Axis_High_minus_Small_Numerosity.npy'
]

# %% Previous Figure v2 - For PhotoRealistic Stimuli
Labels = ['Numerosities', 'Objects', 'Backgrounds']
ylabel = "Statistic's induced Variation"

Title = [
    'Mean Luminance',
    'Std Luminance',
    'Aggregate Fourier Magnitude',
    'Total Energy in High SF', 
    'Texture Similarity',
    'Image Complexity',
]

ft_text, ft_legend = 20, 17

COLOR_SCALE_Legend = [Patch(facecolor="#3CAB88", edgecolor='k', label='Numerosity'),
                      Patch(facecolor="#DB6C17", edgecolor='k', label='Objects'),
                      Patch(facecolor="#827EB9", edgecolor='k', label='Backgrounds')]
COLOR_SCALE_S = ["#25CE9B", "#F46302", "#A09AF4"]
COLOR_SCALE_E = ["#167C5C",  "#AF4701", "#656199"]
HLINES = [0]

jitter = 0.02

legend = [Patch(facecolor='#CECECE', edgecolor='k', label='Subitizing Range'), Patch(facecolor='#707070', edgecolor='k', label='Estimation Range')]

Height = 6
nrows, ncols = 3, 2
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*Height, nrows*Height))

for i in range(nrows):
    for j in range(ncols):
        ax = axs[i, j]; k = ncols*i + j
        title = Title[k]
        Diff_S, Dist_Obj_S, Dist_Bg_S, Diff_E, Dist_Obj_E, Dist_Bg_E = _load_data(dirBase[k], DistPath[k], DiffPath[k])
        if k == 5:
            y_data_S = [-Diff_S, -Dist_Obj_S.flatten(),  -Dist_Bg_S.flatten()]
            y_data_E = [-Diff_E,  -Dist_Obj_E.flatten(),  -Dist_Bg_E.flatten()]
        else:
            y_data_S = [Diff_S,  Dist_Obj_S.flatten(),  Dist_Bg_S.flatten()]
            y_data_E = [Diff_E,  Dist_Obj_E.flatten(),  Dist_Bg_E.flatten()]

        y_S_max = np.max((np.concatenate(y_data_S)))
        y_E_max = np.max((np.concatenate(y_data_E)))

        for yS in y_data_S:
            yS /= y_S_max

        for yE in y_data_E:
            yE /= y_E_max

        beatiful_violin_plot(y_data_S, y_data_E, jitter, title, ft_text, COLOR_SCALE_S, COLOR_SCALE_E, HLINES=HLINES, ax=axs[i,j])
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_ylim([-1.1, 1.1])
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=ft_text)

plt.subplots_adjust(wspace=.04, hspace=.09)
axs[0,0].legend(handles=legend, loc='lower left', bbox_to_anchor=(-.03, 1.08), fontsize=ft_legend,) #title='Numerosity Range', title_fontsize=ft_legend, ncol=1, columnspacing=.8)
axs[0,1].legend(handles=COLOR_SCALE_Legend, loc='lower right', bbox_to_anchor=(1.03, 1.08), fontsize=ft_legend, title='Changes across ...', title_fontsize=ft_legend, ncol=3, columnspacing=.8)
for i in range(nrows):
    axs[i, 0].set_yticks([-1, 0, 1], [-1, 0, 1])

plt.savefig(
    os.path.join(saving_dir, 'Low_Level_Statistics_Influence_by_Numerosity_Backgrounds_Objects.svg'), 
    bbox_inches='tight',
    dpi=300
) 