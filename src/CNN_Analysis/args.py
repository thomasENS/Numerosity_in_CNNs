import numpy as np
import os

## Main 'root' directory
mDir = ""

## Directory for the CNN analysis
dDir = os.path.join(mDir, "derivatives", "CNN_Analysis")

## Indexes of the different version of each stimulus available
## These versions only differ by the random locations of the object
Versions = [1, 2]
nVersions = len(Versions)

## Name of the two distinct numerosity ranges available & Parametric Space information
PS_Ranges = ["estimation", "subitizing"]
Numerosity = {"estimation": [6, 10, 15, 24], "subitizing": [1, 2, 3, 4]}
Spacing = {
    "estimation": [10.7, 11.0, 11.1, 11.3],
    "subitizing": [11.5, 11.8, 12.0, 12.1],
}
Size_Area = {"estimation": [8.3, 8.5, 8.7, 8.9], "subitizing": [9.1, 9.4, 9.5, 9.7]}

## Information defining the different backgrounds available in our stimulus dataset
Naturals = [
    (1, 103),
    (1, -212),
    (3, -41),
    (6, 68),
    (11, 336),
    (16, 30),
    (19, 26),
    (20, 164),
    (22, 20),
    (23, -65),
]
Artificials = [
    (5, 165),
    (9, 119),
    (9, 213),
    (13, 174),
    (17, -18),
    (21, 80),
    (24, 280),
    (32, 120),
    (33, 180),
    (35, -155),
]
Backgrounds = Naturals + Artificials
nBackgrounds = len(Backgrounds)

# Information defining the different objects available in our stimulus dataset
Animals = [
    "Lemur",
    "Elephant",
    "Toucan",
    "Crane",
    "Owl",
    "Cow",
    "Toad",
    "Turtle",
    "Horse",
    "Goldfish",
]
Tools = [
    "Accordion",
    "AlarmClock",
    "Iron",
    "Blender",
    "CameraLeica",
    "Grater",
    "Lute",
    "HairDryer",
    "Teapot",
    "Wheelbarrow",
]
Objects = Animals + Tools
nObjects = len(Objects)

## Name of the two stimulus types available
## '' for the photorealistic stimulus dataset and '_Mask' for their associated binary masks
Stimulus_Types = ["", "_Mask"]

## Information about the networks used in the analysis (training mode, model name, layers from which features are extracted)
Training_Modes = ["pretrained", "untrained"]
Models = ["alexnet", "resnet50", "vgg16"]
Layers = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]

## Arbitrary user-friendly names used as a wrapper upon our models
Model_Names = {
    "pretrained": {"resnet50": "ResNet50", "vgg16": "VGG16", "alexnet": "AlexNet"},
    "untrained": {
        "resnet50": "Random_ResNet50",
        "vgg16": "Random_VGG16",
        "alexnet": "Random_AlexNet",
    },
}

## Definition of the subspaces for the train/test split of the coarse-grained generalization
SubSpace = {
    1: [Artificials, Tools],
    2: [Artificials, Animals],
    3: [Naturals, Animals],
    4: [Naturals, Tools],
}

subSpaceTrain = [1, 2, 3, 4]
subSpaceTest = {1: [3], 2: [4], 3: [1], 4: [2]}

## Definition of the RidgeCV penalization hyperparameters
Alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

## Definition of the fraction of 100 object x background pairs that we will sample at each iteration
nSampledPairs = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100]
nSampling = len(nSampledPairs)

## Number of repetitions of the train/test procedure that we will perform to statistically sample the whole subspace
nOuterCVloop = np.round(100 / np.array(nSampledPairs)).astype(int) + 1
sampling_idx = 2  # 1280 Samples for Training, prevent as well as possible overconfidence of the trained Ridge !
nSampled, nCVloop = nSampledPairs[sampling_idx], nOuterCVloop[sampling_idx]

## Modalities that we will decode from each stimulus when applying the coarse-grained generalization
Modalities = ["N"]  # + ['Spar', 'IA', 'TA', 'FA', 'SzA', 'Sp']

## Definition of the convinience names for the low-level statistics used in our analysis
Low_Level_Statistics = [
    "Mean_Lum",
    "Std_Lum",
    "Agg_Mag_Fourier",
    "Energy_High_SF",
    "Dist_Texture",
    "Dist_Complexity",
]

## Definition of the numerosities covered by the dot-arrays stimuli
Nasr_Numerosity = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

## Dimension of the features representation for AlexNet/Conv5
nFeatures = 43264

## Definition of the several fraction of the whole unit population
Percentages = np.array([0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1])
nPercentages = len(Percentages)

## Number of images per numerosity used to find the number-selective units with the two-way ANOVA
nImg = 900  # reproducing Kim's Paper for nImg=50 (nImg=900 corresponds to ANOVA sample size of 100%)
