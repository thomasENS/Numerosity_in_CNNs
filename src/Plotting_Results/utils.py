# %% Imports
import numpy as np
import os, csv
import scipy.stats as st
from args import nObjects, nBackgrounds


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


def _read_parkspace_log(PS_path):
    """
    Read a .csv file describing a ParkSpace and return the list of coordinates of each
    point in this space, formated as (N, ID, FD)
    """

    PS_Points = _read_log(PS_path)

    ParkSpace_Description = []
    for point in PS_Points:
        ParkSpace_Description.append(
            (
                int(point["numerosity"]),
                int(point["item_diameter"]),
                int(point["field_diameter"]),
            )
        )
    return ParkSpace_Description


# %% Useful Methods for Plotting results
def beatiful_violin_plot(
    y_data_S,
    y_data_E,
    jitter,
    title,
    ft_text,
    COLOR_SCALE_S,
    COLOR_SCALE_E,
    ylabel=None,
    Labels=None,
    HLINES=None,
    width=0.3,
    ax=None,
):
    """
    Beatiful Violin Plot from the https://python-graph-gallery.com/
    """

    OFFSET = 1.5 * width / 2

    # Create jittered version of "x"
    x_data = [np.array([i] * len(d)) for i, d in enumerate(y_data_S)]
    x_jittered = [x + st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

    POSITIONS = np.array([i for i in range(len(y_data_S))])

    # Colors
    BG_WHITE = "#ffffff"  # "#fbf9f4"
    GREY50 = "#7F7F7F"
    BLACK = "#282724"
    GREY_DARK = "#747473"

    ax.set_title(title, fontsize=ft_text)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=ft_text)

    # Some layout stuff ----------------------------------------------
    # Background color
    ax.set_facecolor(BG_WHITE)

    # Horizontal lines that are used as scale reference
    if HLINES is not None:
        for h in HLINES:
            ax.axhline(h, color="k", ls=(0, (3, 5)), alpha=0.8, zorder=0)

    # Add violins ----------------------------------------------------
    # bw_method="silverman" means the bandwidth of the kernel density
    # estimator is computed via Silverman's rule of thumb.
    # More on this in the bonus track ;)

    # The output is stored in 'violins', used to customize their appearence
    violins = ax.violinplot(
        y_data_S,
        positions=POSITIONS - OFFSET,
        widths=width,
        bw_method="silverman",
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Customize violins (remove fill, customize line, etc.)
    for pc in violins["bodies"]:
        pc.set_facecolor("none")
        pc.set_edgecolor(BLACK)
        pc.set_linewidth(1.4)
        pc.set_alpha(1)

    violins = ax.violinplot(
        y_data_E,
        positions=POSITIONS + OFFSET,
        widths=width,
        bw_method="silverman",
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Customize violins (remove fill, customize line, etc.)
    for pc in violins["bodies"]:
        pc.set_facecolor("none")
        pc.set_edgecolor(BLACK)
        pc.set_linewidth(1.4)
        pc.set_alpha(1)

    # Add boxplots ---------------------------------------------------
    # Note that properties about the median and the box are passed
    # as dictionaries.

    medianprops = dict(linewidth=4, color=GREY_DARK, solid_capstyle="butt")
    boxprops = dict(linewidth=2, color=GREY_DARK)

    ax.boxplot(
        y_data_S,
        positions=POSITIONS - OFFSET,
        showfliers=False,  # Do not show the outliers beyond the caps.
        showcaps=False,  # Do not show the caps
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops,
    )

    ax.boxplot(
        y_data_E,
        positions=POSITIONS + OFFSET,
        showfliers=False,  # Do not show the outliers beyond the caps.
        showcaps=False,  # Do not show the caps
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops,
    )

    # Add jittered dots ----------------------------------------------
    for x, y, color in zip(x_jittered, y_data_S, COLOR_SCALE_S):
        ax.scatter(x - OFFSET, y, s=100, color=color, alpha=0.2, marker=".")

    for x, y, color in zip(x_jittered, y_data_E, COLOR_SCALE_E):
        ax.scatter(x + OFFSET, y, s=100, color=color, alpha=0.2, marker=".")

    if Labels is not None:
        ax.set_xticks(range(len(y_data_S)), Labels, fontsize=ft_text)

    return None


def averaging_diff_per_obj_bg(d):
    """
    Difference is computed as the average across all Objects (resp. Backgrounds) within a specific Backgrounds (resp. Objects)
    """

    avg_bg = [
        np.mean(d[i * nBackgrounds : (i + 1) * nBackgrounds])
        for i in range(nBackgrounds)
    ]
    avg_obj = []
    for j in range(nObjects):
        dist_obj = []
        for i in range(nBackgrounds):
            dist_obj.append(d[i * nBackgrounds + j])
        avg_obj.append(np.mean(dist_obj))
    return avg_obj, avg_bg


def averaging_dist_per_obj_bg(d):
    """
    Dist Across Objects : Within each Background, compute distance between Objects & then Average for the 20 Backgrounds.
    Dist Across Backgrounds : Within each Object, compute distance between Backgrounds & then Average for the 20 Objects.
    """
    
    avg_obj = np.zeros((nObjects, nObjects))
    for i in range(nBackgrounds):
        for j in range(nObjects):
            for k in range(nObjects):
                avg_obj[j, k] += d[i * nBackgrounds + j] - d[i * nBackgrounds + k]
    avg_obj /= nBackgrounds

    avg_bg = np.zeros((nBackgrounds, nBackgrounds))
    for i in range(nObjects):
        for j in range(nBackgrounds):
            for k in range(nBackgrounds):
                avg_bg[j, k] += d[j * nBackgrounds + i] - d[k * nBackgrounds + i]
    avg_bg /= nObjects

    return avg_obj, avg_bg


def _load_data(mDir, dist_path, diff_path):

    Dist_S = np.load(os.path.join(mDir, "PS_Subitizing_Range", dist_path))
    Diff_S = np.load(os.path.join(mDir, "PS_Subitizing_Range", diff_path))
    Dist_Obj_S, Dist_Bg_S = averaging_dist_per_obj_bg(Dist_S)

    Dist_E = np.load(os.path.join(mDir, "PS_Estimation_Range", dist_path))
    Diff_E = np.load(os.path.join(mDir, "PS_Estimation_Range", diff_path))
    Dist_Obj_E, Dist_Bg_E = averaging_dist_per_obj_bg(Dist_E)

    return Diff_S, Dist_Obj_S, Dist_Bg_S, Diff_E, Dist_Obj_E, Dist_Bg_E


def simple_beeswarm2(y, nbins=None, width=1.0):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """

    # Convert y to a numpy array to ensure it is compatible with numpy functions
    y = np.asarray(y)

    # If nbins is not provided, calculate a suitable number of bins based on data length
    if nbins is None:
        # nbins = len(y) // 6
        nbins = np.ceil(len(y) / 6).astype(int)

    # Get the histogram of y and the corresponding bin edges
    nn, ybins = np.histogram(y, bins=nbins)

    # Find the maximum count in any bin to be used in calculating the x positions
    nmax = nn.max()

    # Create an array of zeros with the same length as y, to store x-coordinates
    x = np.zeros(len(y))

    # Divide indices of y-values into corresponding bins
    ibs = []
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):

        # Find the indices where y falls within the current bin
        i = np.nonzero((y > ymin) * (y <= ymax))[0]
        ibs.append(i)

    # Assign x-coordinates to the points in each bin
    dx = width / (nmax // 2)

    for i in ibs:
        yy = y[i]
        if len(i) > 1:

            # Determine the starting index (j) based on the number of elements in the bin
            j = len(i) % 2

            # Sort the indices based on their corresponding y-values
            i = i[np.argsort(yy)]

            # Separate the indices into two halves (a and b) for arranging the points
            a = i[j::2]
            b = i[j + 1 :: 2]

            # Assign x-coordinates to points in each half of the bin
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x
