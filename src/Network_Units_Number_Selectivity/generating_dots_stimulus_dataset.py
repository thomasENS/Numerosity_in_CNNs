# %% Imports & Constants
import os
import numpy as np
import random as rd
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from args import mDir, ImgSize, Dot_Numerosity, nImg

sDir = os.path.join(mDir, "data", "Stimuli", "Dot_Patterns_Dataset")


## Useful Methods
def inpolygon(xq, yq, xv, yv):
    """
    The Path class from the matplotlib.path module is used to define a polygon,
    and the contains_points method is used to check whether a set of points is
    inside the polygon. The result is returned as a array of booleans indicating
    whether each point is inside or outside the polygon.

    The set of points is defined as the set of coordinates : xq, yq.
    The edge of the polygon is defined as the set of coordinates : xv, yv.
    """

    points = np.column_stack((xq, yq))
    polygon = Path(np.column_stack((xv, yv)))
    return polygon.contains_points(points)


def dot_stimulus_generation(number_sets, image_iter):
    """
    Fct that generates stimuli similar to the ones from [K. Dots et al., Sci Adv 5, 2019]
    """

    xax, yax = np.meshgrid(np.arange(ImgSize), np.arange(ImgSize))
    rax = xax + 1j * yax

    image_sets_standard = np.zeros((ImgSize, ImgSize, image_iter, len(number_sets)))

    for ii in range(len(number_sets)):
        numtmp = number_sets[ii]

        for jj in range(image_iter):
            circle_radius = 7 + 0.7 * np.random.randn()
            circle_loc = (
                np.ceil(circle_radius)
                + rd.randint(0, ImgSize - 2 * np.ceil(circle_radius))
                + 1j
                * (
                    np.ceil(circle_radius)
                    + rd.randint(0, ImgSize - 2 * np.ceil(circle_radius))
                )
            )
            radtmp = [circle_radius]
            loctmp = [circle_loc]

            while len(radtmp) < numtmp:

                circle_radius = 7 + 0.7 * np.random.randn()
                circle_loc = (
                    np.ceil(circle_radius)
                    + rd.randint(0, ImgSize - 2 * np.ceil(circle_radius))
                    + 1j
                    * (
                        np.ceil(circle_radius)
                        + rd.randint(0, ImgSize - 2 * np.ceil(circle_radius))
                    )
                )

                distancestmp = np.abs(circle_loc - np.array(loctmp))
                radistmp = np.array(radtmp) + circle_radius
                okToAdd = np.all(distancestmp > radistmp)
                if circle_radius > 0:
                    if okToAdd:
                        radtmp.append(circle_radius)
                        loctmp.append(circle_loc)

            imgtmp = np.zeros((ImgSize, ImgSize))
            for kk in range(numtmp):
                rtmp = np.abs(rax - loctmp[kk])
                imgtmpp = rtmp <= radtmp[kk]
                imgtmp = imgtmp + imgtmpp

            image_sets_standard[:, :, jj, ii] = imgtmp

    image_sets_control1 = np.zeros((ImgSize, ImgSize, image_iter, len(number_sets)))

    for ii in range(len(number_sets)):
        numtmp = number_sets[ii]

        for jj in range(image_iter):
            epsil = np.random.randn(1, numtmp)
            circle_radius = 7 + 0.7 * epsil
            area_sum = np.sum(np.pi * np.power(circle_radius, 2))
            scalingtmp = np.sqrt(area_sum / 1200)
            circle_radius = circle_radius / scalingtmp

            average_dist = 0
            while not (average_dist > 90 and average_dist < 100):
                radtmp, loctmp = [], []
                average_dist = 0
                radind = 1
                while len(radtmp) < numtmp:
                    rad = circle_radius[0, radind - 1]
                    loc = (
                        np.ceil(rad) + rd.randint(0, ImgSize - 2 * np.ceil(rad))
                    ) + 1j * (np.ceil(rad) + rd.randint(0, ImgSize - 2 * np.ceil(rad)))

                    if len(loctmp) >= 1:
                        distancestmp = np.abs(loc - loctmp)
                        radistmp = rad + radtmp
                    else:
                        distancestmp = 1
                        radistmp = 0

                    ok_to_add = np.all(distancestmp > radistmp)
                    if rad > 0:
                        if ok_to_add:
                            radtmp.append(rad)
                            loctmp.append(loc)
                            radind += 1

                if numtmp > 1:
                    for avdind in range(len(loctmp)):
                        tmp = np.abs(loctmp[avdind] - loctmp)
                        tmp = np.delete(tmp, np.where(tmp == 0))
                        distmeantmp = np.mean(tmp)
                        average_dist += distmeantmp
                    average_dist = average_dist / len(loctmp)
                else:
                    average_dist = 95

            imgtmp = np.zeros((ImgSize, ImgSize))
            for kk in range(numtmp):
                rtmp = np.abs(rax - loctmp[kk])
                imgtmpp = np.less_equal(rtmp, radtmp[kk])
                imgtmp = imgtmp + imgtmpp

            image_sets_control1[:, :, jj, ii] = imgtmp

    image_sets_control2 = np.zeros((ImgSize, ImgSize, image_iter, len(number_sets)))

    for ii in range(len(number_sets)):
        numtmp = number_sets[ii]

        for jj in range(image_iter):

            ## Define the "BIG" Convex Hull within which all "Dots" have to be drawn, defined by xtmpp, ytmpp coordinates.
            theta = rd.uniform(0, 180) * np.pi / 180
            xtmpp, ytmpp = np.zeros(5), np.zeros(5)
            for thetaind in range(5):
                xtmpp[thetaind] = round(ImgSize / 2) + 110 * np.cos(
                    theta + (thetaind - 1) * 2 * np.pi / 5
                )
                ytmpp[thetaind] = round(ImgSize / 2) + 110 * np.sin(
                    theta + (thetaind - 1) * 2 * np.pi / 5
                )

            k = ConvexHull(np.c_[xtmpp, ytmpp]).vertices

            ## Create the remaining Dots & ensure they are within the "BIG" Convex Hull (as if they were all CIRCULAR Dots)
            radtmp, loctmp = [], []
            while len(radtmp) < numtmp:

                circle_radius = 7 + 0.7 * np.random.randn()
                circle_loc = (
                    np.ceil(circle_radius)
                    + rd.randint(0, ImgSize - 2 * np.ceil(circle_radius))
                    + 1j
                    * (
                        np.ceil(circle_radius)
                        + rd.randint(0, ImgSize - 2 * np.ceil(circle_radius))
                    )
                )

                distancestmp = np.abs(circle_loc - np.array(loctmp))
                radistmp = circle_radius + np.array(radtmp)
                ok_to_add = np.all(distancestmp > np.sqrt(2) * radistmp)
                IN = inpolygon(
                    np.real(circle_loc), np.imag(circle_loc), xtmpp[k], ytmpp[k]
                )
                if circle_radius > 0:
                    if ok_to_add and np.all(IN):
                        radtmp.append(circle_radius)
                        loctmp.append(circle_loc)

            ## Create Iteratively the Stimuli Image by Adding the Dot Shape one after another (imgtmpp is binary [False...True...False])
            imgtmp = np.zeros((ImgSize, ImgSize))

            for kk in range(numtmp):
                rtmp = rax - loctmp[kk]
                tmp = rd.random()

                ## Choosing at random from the four potential shapes (Rectangle, Circle, Ellipse, Triangle)
                if tmp > 0.75:  # rectangle
                    imgtmpp = np.logical_and(
                        np.abs(np.real(rtmp)) <= radtmp[kk],
                        np.abs(np.imag(rtmp)) <= radtmp[kk],
                    )

                elif tmp > 0.5 and tmp < 0.75:  # circle
                    imgtmpp = np.abs(rtmp) <= radtmp[kk]

                elif tmp < 0.5 and tmp > 0.25:  # ellipse
                    imgtmpp = (
                        np.real(rtmp) ** 2 / (radtmp[kk] ** 2)
                        + np.imag(rtmp) ** 2 / ((0.5 * radtmp[kk]) ** 2)
                    ) < 1

                else:  # triangle

                    ## Create the Triangle Shape at Random Orientation by its Convex Hull (its 3 Vertices)
                    theta = rd.randint(0, 180) * np.pi / 180
                    xtmp, ytmp = np.zeros(3), np.zeros(3)
                    for thetaind in range(3):
                        xtmp[thetaind] = np.real(loctmp[kk]) + radtmp[kk] * np.cos(
                            theta + (thetaind - 1) * 2 * np.pi / 3
                        )
                        ytmp[thetaind] = np.imag(loctmp[kk]) + radtmp[kk] * np.sin(
                            theta + (thetaind - 1) * 2 * np.pi / 3
                        )
                    k = ConvexHull(np.c_[xtmp, ytmp]).vertices

                    ## Create the temporaty Binary Boolean Image to add to imgtmp by looking at which Points of the Grid is contained within the Triangle Convex Hull.
                    IN = inpolygon(
                        xax.flatten(), yax.flatten(), xtmp[k], ytmp[k]
                    ).reshape(rax.shape)
                    imgtmpp = IN

                imgtmp += imgtmpp

            image_sets_control2[:, :, jj, ii] = imgtmp

    return image_sets_standard, image_sets_control1, image_sets_control2


# %% Generating images reproducing Dots stimulus datasets
image_sets_standard, image_sets_control1, image_sets_control2 = (
    dot_stimulus_generation(Dot_Numerosity, nImg)
)

for i in range(nImg):
    for n in range(len(Dot_Numerosity)):
        sPath = os.path.join(
            sDir, f"Standard_Set_N-{Dot_Numerosity[n]}_Stim-{1+ i}.npy"
        )
        np.save(sPath, image_sets_standard[:, :, i, n])
        sPath = os.path.join(
            sDir, f"Control-1_Set_N-{Dot_Numerosity[n]}_Stim-{1+ i}.npy"
        )
        np.save(sPath, image_sets_control1[:, :, i, n])
        sPath = os.path.join(
            sDir, f"Control-2_Set_N-{Dot_Numerosity[n]}_Stim-{1+ i}.npy"
        )
        np.save(sPath, image_sets_control2[:, :, i, n])
