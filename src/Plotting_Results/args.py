import numpy as np
import os

## Main 'root' directory & others useful directories
mDir = ''
saving_dir = os.path.join(mDir, 'derivatives', 'Figures')
result_dir = os.path.join(mDir, 'CNN_Analysis', 'Decoding_Results', 'Photorealistic_Dataset')
nDir       = os.path.join(mDir, 'CNN_Analysis', 'Decoding_Results', 'Dot_Patterns_Dataset')

## Information about the networks used in the analysis (layers from which we extract representations, convinience name)
Layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5']; nLayers = len(Layers)
Models = ['AlexNet',  'ResNet50', 'VGG16', 'Random_AlexNet', 'Random_ResNet50', 'Random_VGG16']; nModels = len(Models)

## Information defining the different backgrounds available in our stimulus dataset
Naturals    = [(5, 165),  (9, 119), (9, 213), (13, 174), (17, -18),
               (21, 80), (24, 280), (32, 120), (33, 180), (35, -155)]
Artificials = [(1, 103), (1, -212), (3, -41),  (6, 68), (11, 336),
               (16, 30), (19, 26), (20, 164), (22, 20), (23, -65)]
Backgrounds = Naturals + Artificials; nBackgrounds = len(Backgrounds)

# Information defining the different objects available in our stimulus dataset
Animals = ['Lemur', 'Elephant', 'Toucan', 
            'Crane', 'Owl', 'Cow', 'Toad',
            'Turtle', 'Horse', 'Goldfish']
Tools   = ['Accordion', 'AlarmClock', 'Iron',
            'Blender', 'CameraLeica', 'Grater',
            'Lute', 'HairDryer', 'Teapot', 
            'Wheelbarrow']
Objects = Animals + Tools; nObjects = len(Objects)

## Indexes of the different version of each stimulus available 
## These versions only differ by the random locations of the object
Versions = [1, 2]; nVersions = len(Versions)

## Name of the two distinct numerosity ranges available & Parametric Space information
PS_Ranges = ['estimation', 'subitizing']; nRanges = len(PS_Ranges)
Numerosity = {'estimation':[6, 10, 15, 24], 'subitizing':[1, 2, 3, 4]}; nPoints = 4
Spacing    = {'estimation':[10.7, 11. , 11.1, 11.3], 'subitizing':[11.5, 11.8, 12. , 12.1]}
Size_Area  = {'estimation':[8.3, 8.5, 8.7, 8.9], 'subitizing':[9.1, 9.4, 9.5, 9.7]}

## Decoding Chance level for the Decoding Analysis
Chance_Level = {'N':.448, 'Sp':.448, 'SzA':.448, 'FA':.288, 'TA':.288, 'IA':.288, 'Spar':.288}

## Definition of the convinience names for the low-level statistics used in our analysis
Low_Level_Statistics = ['Mean_Lum', 'Std_Lum', 
                        'Agg_Mag_Fourier', 'Energy_High_SF',
                        'Dist_Texture', 'Dist_Complexity']
nStatistics = len(Low_Level_Statistics)

## Definition of the several fraction of the whole unit population
Percentages = np.array([.0025, .005, .01, .02, .04, .08, .16, .32, .64, 1])
nPercentages = len(Percentages)