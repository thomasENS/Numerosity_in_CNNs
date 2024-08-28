## Installation of the required python libraries

```
git clone git@github.com:thomasENS/Numerosity_in_CNNs.git
pip install -r requirements.txt
```
## Folder structure

```
mDir
├── src
│   ├── Stimulus_Creation
│   |   └── PS_{subitizing/estimation}_range.csv
│   ├── Low_Level_Statistics_Quantification
│   |   └── *.py
│   ├── CNN_Analysis
│   |   └── *.py
│   ├── Network_Units_Number_Selectivity
│   |   └── *.py
│   └── Plotting_Results
│       └── *.py
├── derivatives
│   ├── Low_Level_Statistics
│   |   ├── Frequency_Measures
│   |   |   └── PS_{Subitizing/Estimation}_Range
│   |   ├── Luminance_Measures
│   |   |   └── PS_{Subitizing/Estimation}_Range
│   |   └── Local_Contrast_Measures
│   |       └── PS_{Subitizing/Estimation}_Range
│   ├── CNN_Analysis
│   |   ├── Decoding_Results
|   |   |   ├── Dot_Patterns_Dataset
|   |   |   └── Photorealistic_Dataset
│   |   |       └── PS_{Subitizing/Estimation}_Range
│   |   ├── Features_Selection
│   |   |   └── PS_{Subitizing/Estimation}_Range
|   |   └── CNN_Representations
|   |       ├── Dot_Patterns_Dataset
|   |       |   └── Network-nn
|   |       |       └── Layer-mm
|   |       └── Photorealistic_Dataset
|   |           └── PS_{Subitizing/Estimation}_Range
|   |               └── Network-nn
|   |                   └── Layer-mm
|   |                       └── Object-xx
|   |                           └── Background-yy
│   └── Figures
├── data
|   └── Stimuli
|       ├── Dot_Patterns_Dataset
|       └── Photorealistic_Dataset
|           └── PS_{Subitizing/Estimation}_Range
|               ├── Images
|               |   └── Object-xx
|               |       └── Background-yy
|               |           └── *.png
|               └── Masks
|                   └── Object-xx
|                       └── Background-yy
|                           └── *.npy
├── README.md
└── requirements.txt
```

## Instructions
To reproduce the analyses from our paper, please run _**one after the other**_ the following scripts within an adequate python environment.

**NOTE** : _Since our stimulus datasets and the feature dimensions are quite large, it can take a very long time to run the several analyses.
Therefore, do not hesitate to modify the numbers of ```objects```, ```backgrounds```, etc. that will be used in the analyses
by modifying directly the ```args.py``` that contains the input arguments of all the analyses !_

## Creating the stimulus datasets

To generate the dot-array stimulus dataset, run :

```shell
python3 src/Network_Units_Number_Selectivity/generating_dots_stimulus_dataset.py
```

## Feeding the stimulus datasets to the convolutional networks
#### Extract the networks' feature representations

To extract the feature representations from the trained and untrained counterparts of ```AlexNet```, ```ResNet50``` and ```VGG16``` from :
- Our photorealistic stimulus datasets or their associated binary masks
- Some examples of dot-arrays stimulus dataset _(```idx_start``` and ```n_imgs``` can be modified inside the script depending on the number of dot-array stimuli generated)_

run :
```shell
python3 src/CNN_Analysis/extract_network_features_representation.py
```

#### Perform the feature representation dimension reduction for each generalization scheme
To compute the dimension reduction to apply to each representation, run :

```shell
python3 src/CNN_Analysis/perform_features_representation_dimension_reduction.py
```

## Figure 1 - Stimulus dataset characterisation in terms of low-level statistics

To quantify all the following low-level statistics of our photo-realistic stimuli, run : 

* Luminance measures
```shell
python3 src/Low_Level_Statistics_Quantification/luminance_measures_quantification.py
```

* Frequency measures
```shell
python3 src/Low_Level_Statistics_Quantification/frequency_measures_quantification.py
```
_Remark_ : If you want to visually assess for yourself the cut-off frequency between the low and high spatial frequencies, you can open the previous scripts in a notebook and run the plotting sections of the _"difference between the energy spectrum for small and large numerosity"_.

* Local contrast measures
```shell
python3 src/Low_Level_Statistics_Quantification/local_contrast_measures_quantification.py
```

Then, to compute how these low-level statistics vary when one changes numerosity, object or background identity, run :

```shell
python3 src/Low_Level_Statistics_Quantification/low_level_statistics_variation.py
```

**Note** : If you want to perform the quantification of the low-level statistics on the binary mask of our photorealistic stimuli, you just have to change the ```SegMask``` arguments to ```'_Mask'``` in the _args.py_ file and then to relaunch the scripts for the luminance and frequency measures.

Finally, to plot the characterisation of our stimulus dataset as in Fig 1, run :

```shell
python3 src/Plotting_Results/figure_1.py
```

## Figure 2 - Decoding of Numerosity in CNNs across high-level visual changes

To perform numerosity decoding following the coarse-grained generalization _**as in Fig 2.C**_, run :


```shell
python3 src/CNN_Analysis/coarse_grained_numerosity_decoding.py
```

In addition, to ensure that the non-numerical parameters are incongruent between the train/test sets _**as in Fig 2.E**_, run :


```shell
python3 src/CNN_Analysis/coarse_grained_numerosity_decoding_controlling_non_numerical_parameters.py
```

Finally, to plot the decoding results of Fig 2, run :

```shell
python3 src/Plotting_Results/figure_2.py
```

## Figure 3 - Numerosity decoding across fine changes in visual properties

To perform numerosity decoding following the fine-grained generalization _**as in Fig 3.B**_, run :

```shell
python3 src/CNN_Analysis/fine_grained_numerosity_decoding.py
```

To perform the fine-grained generalization using the low-level statistics for baseline comparison _**as in Fig 3.C**_, run :

```shell
python3 src/CNN_Analysis/fine_grained_numerosity_decoding_from_low_level_statistics.py
```

To assess how much the low-level statistics explain the fine-grained numerosity predictions _**as in Fig 3.E**_, run :

```shell
python3 src/CNN_Analysis/fine_grained_numerosity_predictions_assessment_using_low_level_statistics.py
```

Finally, to plot the decoding results of Fig 3, run :

```shell
python3 src/Plotting_Results/figure_3.py
```

## Figure 4 - Analyses of dot-pattern selective units

To find the artificial units of trained AlexNet/Conv5 (as well as its untrained counterpart) that are number selective when using the dot-pattern stimulus dataset, run :

```shell
python3 src/Network_Units_Number_Selectivity/number_selective_units_selection.py
```

To compute the numerosity tuning curves of the previously found number selective units _**as in Fig 4.A**_, run :

```shell
python3 src/Network_Units_Number_Selectivity/compute_tuning_curves.py
```

To perform the coarse-grained generalization on our photo-realistic stimuli but using only feature representations from the numerosity-selective units _**as in Fig 4.C**_, run :

```shell
python3 src/CNN_Analysis/coarse_grained_numerosity_decoding_using_number_neurons.py
```

Finally, to plot the results of Fig 4, run :

```shell
python3 src/Plotting_Results/figure_4.py
```

## Figure 5 - Impact of photorealistic compared to simplified stimuli

Finally, to plot the results of Fig 5, run :

```shell
python3 src/Plotting_Results/figure_5.py
```
