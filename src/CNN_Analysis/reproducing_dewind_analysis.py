# %% Imports and Constant
from sklearn.linear_model import LinearRegression
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from utils import _read_param_space_log
from args import (
    mDir,
    dDir,
    nVersions,
    nObjects,
    Alphas,
    PS_Ranges,
    Layers,
)

iDir = os.path.join(mDir, "Figures")

Stimulus_Features = ["N", "SzA", "Sp", "TA", "IA", "FA", "Spar", "TP", "Cov", "AC"]
nModalities = len(Stimulus_Features)

UseSegMask = ""  # Just Remove (or replace '_Mask' by '') this Arg for using the photorealistic Stimuli (rather than the Segmentation Masks)
Models = [
    "AlexNet",
    "Random_AlexNet",
    "ResNet50",
    "Random_ResNet50",
    "VGG16",
    "Random_VGG16",
]
nModels = len(Models)


## Useful Methods
def _load_regressors(N, ID, FD, modality):
    """
    Load the regressor associated to a specific modality : N, IA, TA, FA and Spar
    """

    if modality == "N":
        y = np.log(N)
    elif modality == "IA":
        y = np.log(ID**2)
    elif modality == "TA":
        y = np.log(N * (ID**2))
    elif modality == "FA":
        y = np.log(np.pi * (FD**2) / 4)
    elif modality == "Spar":
        y = np.log(np.pi * (FD**2) / (4 * N))
    elif modality == "SzA":
        y = 2 * np.log(ID**2) + np.log(N)  # log(SzA) = log(IA) + log(TA)
    else:
        y = 2 * np.log(np.pi * (FD**2) / 4) - np.log(N)  # log(Sp) = log(FA) + log(Spar)

    return y


def _stimulus_features_axis_vector(modality):

    if modality == "N":
        axis_direction = (1, 0, 0)
    elif modality == "IA":
        axis_direction = (-1 / 2, 1 / 2, 0)
    elif modality == "TA":
        axis_direction = (1 / 2, 1 / 2, 0)
    elif modality == "FA":
        axis_direction = (1 / 2, 0, 1 / 2)
    elif modality == "Spar":
        axis_direction = (-1 / 2, 0, 1 / 2)
    elif modality == "SzA":
        axis_direction = (0, 1, 0)
    elif modality == "Sp":
        axis_direction = (0, 0, 1)
    elif modality == "IP":
        axis_direction = (-1.4, 1 / 4, 0)
    elif modality == "TP":
        axis_direction = (3 / 4, 1 / 4, 0)
    elif modality == "Cov":
        axis_direction = (0, 1 / 2, -1 / 2)
    else:
        axis_direction = (0, 1 / 2, 1 / 2)

    return axis_direction


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# %% Execute - Linear regression of (N, SzA, Sp) on the fine-grained across objects numerosity predictions
clf = LinearRegression()

for PS_range in PS_Ranges:

    sDir = os.path.join(
        dDir, "Decoding_Results", f"PS_{PS_range[0].upper() + PS_range[1:]}_Range"
    )

    PS_path = os.path.join(mDir, "Stimulus_Creation", f"new_PS_{PS_range}_range.csv")
    ParkSpace_Description = _read_param_space_log(PS_path)

    X_test = []
    for N, ID, FD in ParkSpace_Description:

        x_N = _load_regressors(N, ID, FD, "N")
        x_SzA = _load_regressors(N, ID, FD, "SzA")
        x_Sp = _load_regressors(N, ID, FD, "Sp")
        X_test.append([x_N, x_SzA, x_Sp])

    X_test = np.array(X_test * nObjects * nObjects * nVersions)

    ## Standardization of the regressors log(N, SzA, Sp)
    X_test -= X_test.mean(axis=0)
    X_test /= X_test.std(axis=0)

    for Model in Models:
        for Layer in Layers:

            results_path = os.path.join(
                sDir, f"FGG_Objects_{Model}_{Layer}_DeWind_Biases_Assessement"
            )
            predictions_path = os.path.join(
                sDir,
                f"Finer_Grain_Generalisation{UseSegMask}_{Model}_{Layer}_Decoding_Log_N",
            )
            save_path = results_path + f"_Beta_Weights{UseSegMask}.npy"

            if not os.path.isfile(save_path):

                Beta_Weights = np.zeros([nObjects, 3])
                predictions = np.load(
                    predictions_path + "_Predictions_Across_Objects.npy"
                )  # shape : (20, 20, 2560) i.e. pred[i, j, :1280] correspond to the predictions of the 2nd version of the stimuli (since X_test on 3-v_idx !)

                ## Loading & Fitting the Predictions (y_test_pred) for each Training (X_train may use different confounders) independantly.
                for i in range(nObjects):

                    y_test_pred = predictions[
                        i, :, :
                    ].flatten()  # Careful: pred[i*1280:(i+1)*1280] correspond to the "3-v_idx" stimuli whereas pred[(i+1)*1280:(i+2)*1280] correspond to the "v_idx"
                    ## centering the target in order to verigy the hypothesis of the product measure (variance decompositon)
                    y_test_pred -= y_test_pred.mean()
                    y_test_pred = y_test_pred.reshape(-1, 1)

                    ## Fitting the BRR between the Predictions (y_test_pred) and the Statistics of the associated Stimuli (X_test).
                    clf.fit(X_test, y_test_pred)

                    ## Estimating the Explained Variance of each Low-Level Statistics accounting for the mapping from X_test to y_test_pred
                    Beta_Weights[i] = clf.coef_

                np.save(save_path, Beta_Weights)

# %% Compute angles for Train/Test on SAME object only
clf = LinearRegression()

for PS_range in PS_Ranges:

    sDir = os.path.join(
        dDir, "Decoding_Results", f"PS_{PS_range[0].upper() + PS_range[1:]}_Range"
    )

    PS_path = os.path.join(mDir, "Stimulus_Creation", f"new_PS_{PS_range}_range.csv")
    ParkSpace_Description = _read_param_space_log(PS_path)

    X_test = []
    for N, ID, FD in ParkSpace_Description:

        x_N = _load_regressors(N, ID, FD, "N")
        x_SzA = _load_regressors(N, ID, FD, "SzA")
        x_Sp = _load_regressors(N, ID, FD, "Sp")
        X_test.append([x_N, x_SzA, x_Sp])

    X_test = np.array(X_test * nObjects * nVersions)

    ## Standardization of the regressors log(N, SzA, Sp)
    X_test -= X_test.mean(axis=0)
    X_test /= X_test.std(axis=0)

    for Model in Models:
        for Layer in Layers:

            results_path = os.path.join(
                sDir,
                f"FGG_Objects_{Model}_{Layer}_DeWind_Biases_Assessement_Train_Test_Same_Object",
            )
            predictions_path = os.path.join(
                sDir,
                f"Finer_Grain_Generalisation{UseSegMask}_{Model}_{Layer}_Decoding_Log_N",
            )
            save_path = results_path + f"_Beta_Weights{UseSegMask}.npy"

            if not os.path.isfile(save_path):

                Beta_Weights = np.zeros([nObjects, 3])
                predictions = np.load(
                    predictions_path + "_Predictions_Across_Objects.npy"
                )  # shape : (20, 20, 2560) i.e. pred[i, j, :1280] correspond to the predictions of the 2nd version of the stimuli (since X_test on 3-v_idx !)

                ## Loading & Fitting the Predictions (y_test_pred) for each Training (X_train may use different confounders) independantly.
                for i in range(nObjects):

                    y_test_pred = predictions[
                        i, i, :
                    ]  # Careful: pred[i*1280:(i+1)*1280] correspond to the "3-v_idx" stimuli whereas pred[(i+1)*1280:(i+2)*1280] correspond to the "v_idx"
                    ## centering the target in order to verigy the hypothesis of the product measure (variance decompositon)
                    y_test_pred -= y_test_pred.mean()
                    y_test_pred = y_test_pred.reshape(-1, 1)

                    ## Fitting the BRR between the Predictions (y_test_pred) and the Statistics of the associated Stimuli (X_test).
                    clf.fit(X_test, y_test_pred)

                    ## Estimating the Explained Variance of each Low-Level Statistics accounting for the mapping from X_test to y_test_pred
                    Beta_Weights[i] = clf.coef_

                np.save(save_path, Beta_Weights)

# %% Shuffle the targets to see what is the "null" angles.
sDir = os.path.join(
    dDir, "Decoding_Results", f"PS_{PS_range[0].upper() + PS_range[1:]}_Range"
)

nShuffles = 10000
Beta_Weights = np.zeros((nShuffles, 3))
for i in range(nShuffles):

    np.random.shuffle(y_test_pred)

    ## Fitting the BRR between the Predictions (y_test_pred) and the Statistics of the associated Stimuli (X_test).
    clf.fit(X_test, y_test_pred)

    ## Estimating the Explained Variance of each Low-Level Statistics accounting for the mapping from X_test to y_test_pred
    Beta_Weights[i] = clf.coef_

Alphas = [[] for _ in range(nModalities)]
for i, modality in enumerate(Stimulus_Features):

    axis_direction = _stimulus_features_axis_vector(modality)

    for betas in Beta_Weights:
        alpha = angle_between(betas, axis_direction)
        Alphas[i].append(alpha)

plt.figure()
for i, modality in enumerate(Stimulus_Features):
    plt.hist(np.array(Alphas[i]) * 180 / np.pi, alpha=0.2, label=modality, bins=100)
plt.legend()

m_N, M_N = np.min(Beta_Weights[:, 0]), np.max(Beta_Weights[:, 0])
m_SzA, M_SzA = np.min(Beta_Weights[:, 1]), np.max(Beta_Weights[:, 1])
m_Sp, M_Sp = np.min(Beta_Weights[:, 2]), np.max(Beta_Weights[:, 2])

plt.figure()
plt.plot([0, 0], [m_SzA, M_SzA], "k")
plt.plot([m_N, M_N], [0, 0], "k")
plt.plot(Beta_Weights[:, 0], Beta_Weights[:, 1], ".")
plt.xlabel("N")
plt.ylabel("SzA")

plt.figure()
plt.plot([0, 0], [m_N, M_N], "k")
plt.plot([m_Sp, M_Sp], [0, 0], "k")
plt.plot(Beta_Weights[:, 2], Beta_Weights[:, 0], ".")
plt.ylabel("N")
plt.xlabel("Sp")

plt.figure()
plt.plot([0, 0], [m_Sp, M_Sp], "k")
plt.plot([m_SzA, M_SzA], [0, 0], "k")
plt.plot(Beta_Weights[:, 1], Beta_Weights[:, 2], ".")
plt.ylabel("Sp")
plt.xlabel("SzA")

for i, modality in enumerate(Stimulus_Features):
    np.save(
        os.path.join(
            sDir, f"{modality}_Angles_Distribution_{nShuffles}_Random_Shuffles.npy"
        ),
        np.array(Alphas[i]),
    )

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

for alpha in Alphas:
    ax.plot([0, alpha], [0, 1], label=modality)

ax.set_title(f"{Model}/{Layer}")
ax.set_rticks([])  # Less radial ticks
ax.set_rmax(1)
ax.set_thetamax(180)
# ax.set_ylabel('Mean vector CNN', x=.9)
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)
plt.legend(
    Stimulus_Features,
    loc="center left",
    bbox_to_anchor=(-0.05, 0.5),
    ncol=1,
    columnspacing=0.6,
)
plt.show()

# %% Evaluating the significance of the CNNs' bias angles
sDir = os.path.join(dDir, "Decoding_Results" + "PS_Subitizing_Range")
nShuffles = 10000
threshold = 0.05

D_alphas_Threshold, M_alphas = [], []
for modality in Stimulus_Features:
    alphas = np.load(
        os.path.join(
            sDir, f"{modality}_Angles_Distribution_{nShuffles}_Random_Shuffles.npy"
        )
    )

    ## finding the mean of each angle's distribution
    cf, x, _ = plt.hist(alphas, bins=100, density=True, cumulative=True)
    idx_mean = len(cf[cf < 0.5])
    alpha_mean = x[idx_mean]
    M_alphas.append(alpha_mean)
    d_alphas = np.zeros(nShuffles)
    for i, alpha in enumerate(alphas):
        d_alphas[i] = np.abs(alpha - alpha_mean)

    ## Computing the Confidence interval at 1-thresold % (if we are outside them, we can reject H0)
    cf, x, _ = plt.hist(d_alphas, bins=100, density=True, cumulative=True)
    idx_d_alpha_threshold = len(cf[cf > 1 - threshold])
    d_alpha_threshold = x[-idx_d_alpha_threshold]
    D_alphas_Threshold.append(d_alpha_threshold)


## Lower and Upper bound for the 1 - thresold % Confidence Interval
## For angles outside of this CI we can, reject the null hypothesis H0
alpha_m = np.mean(M_alphas) - np.mean(D_alphas_Threshold)
alpha_M = np.mean(M_alphas) + np.mean(D_alphas_Threshold)

## For thresold = .05
# alpha_m = 0.2920639275535317
# alpha_M = 2.8174341613441065

# %% Plot DeWind's Angles
## For thresold = .05
alpha_m = 0.2920639275535317
alpha_M = 2.8174341613441065

Height = 3

Models = ["AlexNet", "Random_AlexNet"]

colors = matplotlib.colormaps["tab10"]
lgd = [
    Line2D([0], [0], marker="", color=colors(i), label=Stimulus_Features[i])
    for i, features in enumerate(Stimulus_Features)
]


Case = "_Train_Test_Same_Object"  # "_Train_Test_Same_Object"

for PS_range in PS_Ranges:

    sDir = os.path.join(
        dDir, "Decoding_Results", f"PS_{PS_range[0].upper() + PS_range[1:]}_Range"
    )

    for idx_model, Model in enumerate(Models):

        for idx_layer, Layer in enumerate(Layers):

            results_path = os.path.join(
                sDir,
                f"FGG_Objects_{Model}_{Layer}_DeWind_Biases_Assessement{Case}_Beta_Weights{UseSegMask}.npy",
            )
            Beta_Weights = np.load(results_path)

            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

            Alphas, Std = [], []
            for modality in Stimulus_Features:

                axis_direction = _stimulus_features_axis_vector(modality)

                alphas = []
                for betas in Beta_Weights:
                    alpha = angle_between(betas, axis_direction)
                    alphas.append(alpha)

                Alphas.append(np.mean(alphas))
                Std.append(np.std(alphas))

            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            ## Plot the region where we cannot reject the null hypothesis (CI at 95%)
            ax.fill_between(
                x=[alpha_m, alpha_M], y1=0, y2=5, color="#C8C8C8", alpha=0.6
            )

            for k, alpha in enumerate(Alphas):
                ax.plot([0, alpha], [0, 1], label=modality)
                ax.fill_between(
                    x=[alpha - Std[k] / 2, alpha + Std[k] / 2], y1=0, y2=5, alpha=0.6
                )

            # ax.set_title(f'{Model}/{Layer}')
            ax.set_rticks([])  # Less radial ticks
            ax.set_rmax(1)
            ax.set_thetamax(180)
            # ax.set_ylabel('Mean vector CNN', x=.9)
            ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
            ax.grid(True)

            # plt.legend(handles=lgd, loc='lower left', bbox_to_anchor=(0, 1.1), ncol=nModalities, columnspacing=.6)
            plt.savefig(
                os.path.join(
                    iDir,
                    f"{Model}_{Layer}_DeWind_Biases_Assessement{Case}_Beta_Weights{UseSegMask}_{PS_range[0].upper() + PS_range[1:]}_Range.svg",
                ),
                dpi=300,
                bbox_inches="tight",
            )

    # plt.legend(handles=lgd, loc='center left', bbox_to_anchor=(-.05, .5), ncol=1, columnspacing=.6)
