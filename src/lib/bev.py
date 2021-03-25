# Added Script for Birds Eye Transformation
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec

import pandas as pd
from skimage import morphology
from scipy.ndimage import rotate


def autocrop(image, threshold=0):
    """
    Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    return image


def get_patch(segmented_image, latitude, longitude, theta, dst_x, dst_z):
    angle = np.deg2rad(theta)
    # Transforming Coordinates
    x, z = longitude, latitude 
    corner4 = np.array([[-dst_x, 0],
                        [ dst_x, 0], 
                        [ dst_x, dst_z], 
                        [-dst_x, dst_z]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]])
    corner4t = np.dot(R, corner4.T).T
    corner4t[:, 0] = corner4t[:, 0] + x
    corner4t[:, 1] = corner4t[:, 1] + z
    min_x, min_z = np.min(corner4t, axis=0)
    max_x, max_z = np.max(corner4t, axis=0)
    roi = segmented_image[int(min_z):int(max_z), int(min_x):int(max_x)]
    corner4t[:, 0] = corner4t[:, 0] - min_x
    corner4t[:, 1] = corner4t[:, 1] - min_z
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(mask, [corner4t.astype(int)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    roit = cv2.bitwise_and(roi, roi, mask=mask)
    roit = rotate(roit, theta)
    roit = autocrop(roit)
    return np.flip(roit, axis=1)


def slope_calculator(x, y):
    long_a, long_b = x[0], x[-1]
    lat_a, lat_b = y[0], y[-1]
    dL = long_b - long_a
    X = np.cos(np.deg2rad(lat_b)) * np.sin(np.deg2rad(dL))
    Y = np.cos(np.deg2rad(lat_a)) * np.sin(np.deg2rad(lat_b)) - np.sin(np.deg2rad(lat_a)) \
        * np.cos(np.deg2rad(lat_b)) * np.cos(np.deg2rad(dL))
    return np.rad2deg(np.arctan2(X,Y))


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def compute_birdviewbox(info_dict, shape, scale):
    h = info_dict['dim'][0] * scale
    w = info_dict['dim'][1] * scale
    l = info_dict['dim'][2] * scale
    x = info_dict['loc'][0] * scale
    y = info_dict['loc'][1] * scale
    z = info_dict['loc'][2] * scale
    rot_y = info_dict['rot_y']
    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T
    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2
    x_corners += -w / 2
    z_corners += -l / 2
    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape / 2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T
    return np.vstack((corners_2D, corners_2D[0, :]))


def draw_birdeyes(ax2, info_dict, shape, scale):
    pred_corners_2d = compute_birdviewbox(info_dict, shape=shape, scale=scale)
    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=True, color='black', label='prediction')
    ax2.add_patch(p)


def get_bev(res, opt, scale, birdimage):
    shape_h = birdimage.shape[0]
    shape_w = birdimage.shape[1]
    fig = plt.figure(figsize=(shape_w/100, shape_h/100))
    ax2 = fig.add_subplot()
    for index in range(len(res)):
        draw_birdeyes(ax2, res[index], shape=shape_w, scale=scale)
    # plot camera view range
    x1 = np.linspace(0, shape_w / 2, 100)
    x2 = np.linspace(shape_w / 2, shape_w, 100)
    y1 = np.linspace(shape_h / 2, 0, 100)
    y2 = np.linspace(0, shape_h / 2, 100)
    ax2.plot(x1, y1, ls='--', color='grey', linewidth=1, alpha=10)
    ax2.plot(x2, y2, ls='--', color='grey', linewidth=1, alpha=10)
    ax2.plot(shape_w / 2, 0, marker='+', markersize=16, markeredgecolor='black')
    # visualize bird eye view
    ax2.imshow(birdimage, origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])
    # redraw the canvas
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(shape_h, shape_w, 3)
    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.close()
    return img


def undistort_image(img):
    """
    A custom function to undistort image
    just for Negeley-Black Video.
    """
    h, w = img.shape[:2]
    mtx = np.array([
        [3389.14855, 0, 982.985434],
        [0, 3784.14471, 556.363307],
        [0, 0, 1]])
    dist = np.array([-1.83418584,  12.2930625, -0.00434882103,  0.0226389517, -85.1805652])
    # undistort
    img = cv2.undistort(img, mtx, dist)
    return img
