# %% Imports, Constants
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from args import (result_dir, saving_dir, Chance_Level, PS_Ranges)

## Decoded Modality in the coarse-grained generalization
Modality = 'N'

## Convinient Naming of the Models for the Legend
rModels = ['Random_AlexNet', 'Random_ResNet50', 'Random_VGG16']
tModels = ['AlexNet', 'ResNet50', 'VGG16']
Models = tModels + rModels
Cases = ['Trained', 'Untrained']; cModels = {'Trained':tModels, 'Untrained':rModels}

## Axes limits
lims = [-1.5060440607531775 + .2, 0.6683987855562563]
Ticks = np.log(np.array([.22, Chance_Level[Modality], .8, 1.5]))

## Manual Legend
Markers = {'AlexNet':'o', 'ResNet50':'s', 'VGG16':'v',
           'Random_AlexNet':'o', 'Random_ResNet50':'s', 'Random_VGG16':'v'}
Colors = ['#0095EF', '#3C50B1', '#6A38B3', '#F31D64', '#FE433C']
legend_layers = [Patch(facecolor=Colors[i], label=f'Conv{i+1}') for i in range(len(Colors))]
legend_models = [Line2D([0], [0], marker=Markers[Model], color='w', label=Model, markerfacecolor='k', markersize=9) for Model in tModels]
legend_chance = [Patch(facecolor='gray', alpha=.7, edgecolor='k', label='MAE Better\nthan Chance')]

# %% Fig 2C : Coarse Grained Generalisation
MAE, STD = {PS_range:{Model:[] for Model in Models} for PS_range in PS_Ranges}, {PS_range:{Model:[] for Model in Models} for PS_range in PS_Ranges}
for PS_range in PS_Ranges:
    uDir = os.path.join(result_dir, f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

    for Model in Models:
        MAE[PS_range][Model] = np.load(os.path.join(uDir, f'Full_Generalisation_{Model}_Decoding_Log_{Modality}_Across_Hierarchy_Score_MAE.npy'))
        STD[PS_range][Model] = np.load(os.path.join(uDir, f'Full_Generalisation_{Model}_Decoding_Log_{Modality}_Across_Hierarchy_Score_Std.npy'))

n = 0
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=False, figsize=(7, 3.6))
for case in Cases:
    n += 1

    plt.subplot(120 + n)
    ## Better than Chance Level Area
    axes[n-1].fill_between(x=[np.log(.1), np.log(Chance_Level[Modality])], y1=np.log(.1), y2=np.log(Chance_Level[Modality]), color='gray', alpha=.4)
    axes[n-1].fill_between(x=[np.log(.1), np.log(Chance_Level[Modality])], y1=np.log(Chance_Level[Modality]), y2=np.log(2.3), color='gray', alpha=.2)
    axes[n-1].fill_between(x=[np.log(Chance_Level[Modality]), np.log(2.3)], y1=np.log(.1), y2=np.log(Chance_Level[Modality]), color='gray', alpha=.2)

    ## Plot MAE of every Model
    for Model in cModels[case]:
        for i in range(5):
            x, y = np.log(MAE['estimation'][Model][i]), np.log(MAE['subitizing'][Model][i])
            xerr = np.log((MAE['estimation'][Model][i] + STD['estimation'][Model][i]/4)/(MAE['estimation'][Model][i] - STD['estimation'][Model][i]/4))
            yerr = np.log((MAE['subitizing'][Model][i] + STD['subitizing'][Model][i]/4)/(MAE['subitizing'][Model][i] - STD['subitizing'][Model][i]/4))
            plt.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, markeredgewidth=.8, ms=7,
                        marker=Markers[Model], markeredgecolor='k', color=Colors[i], ecolor=Colors[i],
                        elinewidth=.8)

    plt.xlim(lims); plt.ylim(lims)
    
## Manual Labels
plt.subplots_adjust(wspace=0.05)
axes[0].set_xticks(Ticks, [f'{xT:.2f}' for xT in np.exp(Ticks)])
axes[1].set_xticks(Ticks, [f'{xT:.2f}' for xT in np.exp(Ticks)])
axes[0].set_yticks(Ticks, [f'{yT:.2f}' for yT in np.exp(Ticks)])
axes[0].set_ylabel('MAE [Log] - Subitizing Range', fontsize=12)
axes[0].set_title('Trained Models', fontsize=12); axes[1].set_title('Untrained Counterparts', fontsize=12)
fig.supxlabel('MAE [Log] - Estimation Range', fontsize=12)
lgd_models = axes[0].legend(handles=legend_models, loc='upper right', bbox_to_anchor=(1, 1))
axes[0].legend(handles=legend_layers, loc='upper left', bbox_to_anchor=(0, 1))
axes[0].add_artist(lgd_models)
axes[1].legend(handles=legend_chance, loc='upper right', bbox_to_anchor=(1, 1))
## Separation between Better vs. Worse than Chance Level Region   
for n in range(2):
    axes[n].plot([np.log(.1), np.log(Chance_Level[Modality])], [np.log(Chance_Level[Modality])]*2, 'k--', linewidth=1.)
    axes[n].plot([np.log(Chance_Level[Modality])]*2,  [np.log(.1), np.log(Chance_Level[Modality])], 'k--', linewidth=1.)
    axes[n].plot([np.log(Chance_Level[Modality]), np.log(2)], [np.log(Chance_Level[Modality])]*2, '--', color='gray', linewidth=1.)
    axes[n].plot([np.log(Chance_Level[Modality])]*2,  [np.log(Chance_Level[Modality]), np.log(2)], '--', color='gray', linewidth=1.)

plt.savefig(
    os.path.join(saving_dir, 'Coarse_Grained_Numerosity_Decoding_Classical_Generalisation.svg'),
    dpi=300,
    bbox_inches='tight'
)

# %% Fig 2E : Coarse Grained Generalisation with Congruent / Incongruent Sets
MAE, STD = {PS_range:{Model:[] for Model in Models} for PS_range in PS_Ranges}, {PS_range:{Model:[] for Model in Models} for PS_range in PS_Ranges}
for PS_range in PS_Ranges:
    uDir = os.path.join(result_dir + f'PS_{PS_range[0].upper() + PS_range[1:]}_Range')

    for Model in Models:
        MAE[PS_range][Model] = np.load(os.path.join(uDir, f'Full_Generalisation_Controlled_Non_Numerical_Params_{Model}_across_Hierarchy_Decoding_Log_{Modality}_Score_MAE.npy'))
        STD[PS_range][Model] = np.load(os.path.join(uDir, f'Full_Generalisation_Controlled_Non_Numerical_Params_{Model}_across_Hierarchy_Decoding_Log_{Modality}_Score_Std.npy'))

n = 0
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=False, figsize=(7, 3.6))
for case in Cases:
    n += 1

    plt.subplot(120 + n)
    ## Better than Chance Level Area
    axes[n-1].fill_between(x=[np.log(.1), np.log(Chance_Level[Modality])], y1=np.log(.1), y2=np.log(Chance_Level[Modality]), color='gray', alpha=.4)
    axes[n-1].fill_between(x=[np.log(.1), np.log(Chance_Level[Modality])], y1=np.log(Chance_Level[Modality]), y2=np.log(2.3), color='gray', alpha=.2)
    axes[n-1].fill_between(x=[np.log(Chance_Level[Modality]), np.log(2.3)], y1=np.log(.1), y2=np.log(Chance_Level[Modality]), color='gray', alpha=.2)

    ## Plot MAE of every Model
    for Model in cModels[case]:
        for i in range(5):
            x, y = np.log(MAE['estimation'][Model][i]), np.log(MAE['subitizing'][Model][i])
            xerr = np.log((MAE['estimation'][Model][i] + STD['estimation'][Model][i]/4)/(MAE['estimation'][Model][i] - STD['estimation'][Model][i]/4))
            yerr = np.log((MAE['subitizing'][Model][i] + STD['subitizing'][Model][i]/4)/(MAE['subitizing'][Model][i] - STD['subitizing'][Model][i]/4))
            plt.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, markeredgewidth=.8, ms=7,
                        marker=Markers[Model], markeredgecolor='k', color=Colors[i], ecolor=Colors[i],
                        elinewidth=.8)

    plt.xlim(lims); plt.ylim(lims)
    
## Manual Labels
plt.subplots_adjust(wspace=0.05)
axes[0].set_xticks(Ticks, [f'{xT:.2f}' for xT in np.exp(Ticks)])
axes[1].set_xticks(Ticks, [f'{xT:.2f}' for xT in np.exp(Ticks)])
axes[0].set_yticks(Ticks, [f'{yT:.2f}' for yT in np.exp(Ticks)])
axes[0].set_ylabel('MAE [Log] - Subitizing Range', fontsize=12)
axes[0].set_title('Trained Models', fontsize=12); axes[1].set_title('Untrained Counterparts', fontsize=12)
fig.supxlabel('MAE [Log] - Estimation Range', fontsize=12)
legend_models = axes[0].legend(handles=legend_models, loc='upper right', bbox_to_anchor=(1, 1))
axes[0].legend(handles=legend_layers, loc='upper left', bbox_to_anchor=(0, 1))
axes[0].add_artist(legend_models)
axes[1].legend(handles=legend_chance, loc='upper right', bbox_to_anchor=(1, 1))
## Separation between Better vs. Worse than Chance Level Region   
for n in range(2):
    axes[n].plot([np.log(.1), np.log(Chance_Level[Modality])], [np.log(Chance_Level[Modality])]*2, 'k--', linewidth=1.)
    axes[n].plot([np.log(Chance_Level[Modality])]*2,  [np.log(.1), np.log(Chance_Level[Modality])], 'k--', linewidth=1.)
    axes[n].plot([np.log(Chance_Level[Modality]), np.log(2)], [np.log(Chance_Level[Modality])]*2, '--', color='gray', linewidth=1.)
    axes[n].plot([np.log(Chance_Level[Modality])]*2,  [np.log(Chance_Level[Modality]), np.log(2)], '--', color='gray', linewidth=1.)

plt.savefig(
    os.path.join(saving_dir, 'Coarse_Grained_Numerosity_Decoding_Congruent_Incongruent_Generalisation.svg'),
    dpi=300,
    bbox_inches='tight'
)