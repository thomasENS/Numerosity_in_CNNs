# %% Imports
import csv
import numpy as np

# Outer Field Area Mask of our stimuli
from PIL import Image, ImageDraw

width, height = 900, 900
percent_max = 0.95
mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse(
    (
        (1 - percent_max) * width / 2,
        (1 - percent_max) * height / 2,
        (1 + percent_max) * width / 2,
        (1 + percent_max) * height / 2,
    ),
    fill=255,
)
mask = np.array(mask)


# %% Useful Methods for loading the stimulus dataset
def _read_log(log_path):
    """
    Read the .csv file associated with the given path and return each row as a
    dictionnary whose values are the different column content in regard to the row
    and whose keys are the content of each column of the first row.
    """

    with open(log_path, mode="r", newline="") as log_file:
        logs = list(csv.DictReader(log_file))

    return logs


def _read_param_space_log(PS_path):
    """
    Read a .csv file describing the Parametric Space and return the list of coordinates of each
    point in this space, formated as (N, ID, FD)
    """

    PS_Points = _read_log(PS_path)

    Param_Space_Description = []
    for point in PS_Points:
        Param_Space_Description.append(
            (
                int(point["numerosity"]),
                int(point["item_diameter"]),
                int(point["field_diameter"]),
            )
        )
    return Param_Space_Description


# %% Useful Method to quantify the low-level statistics in our stimui
def _compute_param_space_point(N, ID, FD):
    """
    Conversion from the non-numerical parameters space (N, ID, FD) to the parametric space (N, Sp, SzA)
    """

    Sp = np.round(np.log10(np.pi**2 * FD**4 / N) * 10) / 10
    SzA = np.round(np.log10(ID**4 * N) * 10) / 10
    return Sp, SzA


def _avg_weibull_params(uDir, object_name, bg_name):
    """
    Fct that computes the weibull parameters (beta, gamma) across all the parametric space parameters (N, Sp, SzA)
    """

    mBeta = np.load(os.path.join(uDir, f"Sigma-12_{object_name}_{bg_name}_Beta.npy"))
    mGamma = np.load(os.path.join(uDir, f"Sigma-12_{object_name}_{bg_name}_Gamma.npy"))
    return {
        "bN": np.mean(mBeta, axis=(1, 2)),
        "gN": np.mean(mGamma, axis=(1, 2)),
        "bSp": np.mean(mBeta, axis=(0, 2)),
        "gSp": np.mean(mGamma, axis=(0, 2)),
        "bSzA": np.mean(mBeta, axis=(1, 0)),
        "gSzA": np.mean(mGamma, axis=(1, 0)),
    }
