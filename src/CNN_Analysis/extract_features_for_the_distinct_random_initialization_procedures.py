# %% Imports and Constants
import os
import numpy as np
from utils import (
    TorchHub_FeatureExtractor,
    ImageNet_Stimuli_PreProcessing,
    _read_param_space_log,
)
from args import (
    mDir,
    dDir,
    Versions,
    PS_Ranges,
    Model_Names,
)

Selected_Backgrounds_Info = [
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

Selected_Objects_Info = [
    ["Lemur", "Animal"],
    ["Elephant", "Animal"],
    ["Cow", "Animal"],
    ["Crane", "Animal"],
    ["Owl", "Animal"],
    ["Toucan", "Animal"],
    ["Turtle", "Animal"],
    ["Toad", "Animal"],
    ["Goldfish", "Animal"],
    ["Horse", "Animal"],
    ["Accordion", "Tool"],
    ["AlarmClock", "Tool"],
    ["Blender", "Tool"],
    ["CameraLeica", "Tool"],
    ["Grater", "Tool"],
    ["Lute", "Tool"],
    ["HairDryer", "Tool"],
    ["Teapot", "Tool"],
    ["Iron", "Tool"],
    ["Wheelbarrow", "Tool"],
]

sDir = os.path.join(dDir, "Stimuli")
rDir = os.path.join(dDir, "Representations", "Revision")


def _check_folder(pth):
    """
    Fct that checks if a path leads to an existing folder :
            - If not, creates recursively the path to the folder.
    """

    try:
        if not os.path.isdir(pth):
            _check_folder(os.path.dirname(pth))
            os.mkdir(pth)
    except:
        pass

    return None


## Useful Methods to initialize the untrained networks
import torch.nn as nn
import torch.nn.init as init


def variable_normal_initialize(model, gain=1.0):
    """
    Applies Normal initialization to all layers of the given PyTorch model.
    Allows specifying a gain factor that adjusts the standard deviation.

    Remarks :
      - Lecun procedure corresponds to gain = 1.0
      - He procedure corresponds to gain = 2.0
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
            std = gain / (fan_in**0.5)  # Adjusting std for normal distribution
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)

    return None


def variable_uniform_initialize(model, gain=1.0):
    """
    Applies Uniform initialization to all layers of the given PyTorch model.
    Allows specifying a gain factor that adjusts the standard deviation.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            fan_in, _ = init._calculate_fan_in_and_fan_out(module.weight)
            a = (
                np.sqrt(3 / fan_in) * gain
            )  # Adjusting std for uniform distribution to equal the std from normal distribution
            init.uniform_(module.weight, low=-a, high=a)
            if module.bias is not None:
                init.zeros_(module.bias)

    return None


# %% Extract Representation for Selected Objects & Backgrounds for Earlier Layers
Model = "alexnet"
Mode = "untrained"
Layers = ["Conv5"]

Gains = [1 / 500, 1 / 10, 2, 10]
Gains_Uniform = [1 / 500, 1 / 10, 1, 2, 10]

preprocess = ImageNet_Stimuli_PreProcessing()
FeaturesExtractor = TorchHub_FeatureExtractor(Model, Mode)

for idx_gain, gain in enumerate(Gains):

    variable_normal_initialize(FeaturesExtractor, gain=gain)

    for PS_range in PS_Ranges:

        iDir = os.path.join(
            sDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range", "Images"
        )
        fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

        PS_path = os.path.join(
            mDir, "Stimulus_Creation", f"new_PS_{PS_range}_range.csv"
        )
        ParkSpace_Description = _read_param_space_log(PS_path)

        for layer in Layers:
            _check_folder(os.path.join(fDir, Model_Names[Mode][Model], layer))

        for object_name, category in Selected_Objects_Info:
            for bg_idx, bg_alpha in Selected_Backgrounds_Info:
                for N, ID, FD in ParkSpace_Description:
                    for v_idx in Versions:

                        features_paths = [
                            os.path.join(
                                fDir,
                                Model_Names[Mode][Model],
                                layer,
                                f"{Model_Names[Mode][Model]}_{layer}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}_gain-{idx_gain}.npy",
                            )
                            for layer in Layers
                        ]
                        if not np.any(
                            [
                                not os.path.isfile(features_path)
                                for features_path in features_paths
                            ]
                        ):  # Do not compute a representation that was already created !

                            img_path = os.path.join(
                                iDir,
                                category,
                                object_name,
                                f"Bg-{bg_idx}_Alpha{bg_alpha}",
                                f"{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.png",
                            )

                            # if os.path.isfile(img_path): # Do not try to compute representation of an image that does not exist !
                            #     img_input = preprocess(img_path)
                            #     features  = FeaturesExtractor(img_input)

                            for layer in Layers:
                                features_path = os.path.join(
                                    fDir,
                                    Model_Names[Mode][Model],
                                    layer,
                                    f"{Model_Names[Mode][Model]}_{layer}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}_gain-{idx_gain}.npy",
                                )
                                # np.save(features_path, features[layer])
                                os.system(f"rm {features_path}")

for idx_gain, gain in enumerate(Gains_Uniform):

    variable_normal_initialize(FeaturesExtractor, gain=gain)

    for PS_range in PS_Ranges[:1]:

        iDir = os.path.join(
            sDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range", "Images"
        )
        fDir = os.path.join(rDir, f"PS_{PS_range[0].upper() + PS_range[1:]}_Range")

        PS_path = os.path.join(
            mDir, "Stimulus_Creation", f"new_PS_{PS_range}_range.csv"
        )
        ParkSpace_Description = _read_param_space_log(PS_path)

        for layer in Layers:
            _check_folder(os.path.join(fDir, Model_Names[Mode][Model], layer))

        for object_name, category in Selected_Objects_Info:
            for bg_idx, bg_alpha in Selected_Backgrounds_Info:
                for N, ID, FD in ParkSpace_Description:
                    for v_idx in Versions:

                        features_paths = [
                            os.path.join(
                                fDir,
                                Model_Names[Mode][Model],
                                layer,
                                f"{Model_Names[Mode][Model]}_{layer}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}_gain_uniform-{idx_gain}.npy",
                            )
                            for layer in Layers
                        ]
                        if not np.any(
                            [
                                not os.path.isfile(features_path)
                                for features_path in features_paths
                            ]
                        ):  # Do not compute a representation that was already created !

                            img_path = os.path.join(
                                iDir,
                                category,
                                object_name,
                                f"Bg-{bg_idx}_Alpha{bg_alpha}",
                                f"{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}.png",
                            )

                            # if os.path.isfile(img_path): # Do not try to compute representation of an image that does not exist !
                            #     img_input = preprocess(img_path)
                            #     features  = FeaturesExtractor(img_input)

                            for layer in Layers:
                                features_path = os.path.join(
                                    fDir,
                                    Model_Names[Mode][Model],
                                    layer,
                                    f"{Model_Names[Mode][Model]}_{layer}_{object_name}-{N}_ID-{ID}_FD-{FD}_bg-{bg_idx}_alpha{bg_alpha}_version-{v_idx}_gain_uniform-{idx_gain}.npy",
                                )
                                # np.save(features_path, features[layer])
                                os.system(f"rm {features_path}")
