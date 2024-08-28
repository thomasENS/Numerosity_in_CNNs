import os

## Main 'root' directory
mDir = ""

## Directory for the CNN analysis
dDir = os.path.join(mDir, "derivatives", "CNN_Analysis")

## Information about the networks used for this analysis
Models = ["AlexNet", "Random_AlexNet"]
Layer = "Conv5"

## Definition of the numerosities covered by the Dot stimuli
Dot_Numerosity = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

## Size of an Image Stimulus (ImgSize x ImgSize)
ImgSize = 224

## Dimension of the features representation for AlexNet/Conv5
nFeatures = 43264

## Number of images per numerosity used to find the number-selective units with the two-way ANOVA
nImg = 900  # reproducing Kim's Paper for nImg=50 (nImg=900 corresponds to ANOVA sample size of 100%)
