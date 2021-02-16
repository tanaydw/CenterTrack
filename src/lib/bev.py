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


def get_segmented_map(img):
    segmented_image = np.zeros(img.shape, dtype=np.uint8)
    
    # Defining background color = (254, 204, 165)
    segmented_image[:, :, 0] = 254
    segmented_image[:, :, 1] = 204
    segmented_image[:, :, 2] = 165

    # Segmenting Road (255, 255, 255)
    road_mask = (img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)
    segmented_image[road_mask, 0] = 219
    segmented_image[road_mask, 1] = 108
    segmented_image[road_mask, 2] = 115

    # Segmenting Buildings (241, 241, 241)
    build_mask = (img[:, :, 0] == 241) & (img[:, :, 1] == 241) & (img[:, :, 2] == 241)
    """n_opt = 3
    for _ in range(n_opt):
        build_mask = morphology.binary_erosion(build_mask)
    for _ in range(n_opt):
        build_mask = morphology.binary_dilation(build_mask)"""
    segmented_image[build_mask, 0] = 254
    segmented_image[build_mask, 1] = 137
    segmented_image[build_mask, 2] = 9
    
    return segmented_image


def get_map_vectors(p1, p2, img):
    u1, v1 = 0, 0
    u2, v2 = img.shape[1], img.shape[0]
    lat1, long1 = p1
    lat2, long2 = p2

    scale_u = (long1 - long2) / (u1 - u2)
    scale_v = (lat1 - lat2) / (v1 - v2)

    lat_org = lat1 - scale_v * v1
    long_org = long1 - scale_u * u1
    
    return lat_org, long_org, scale_u, scale_v


def get_u(x, long_org, scale_u):
    return int((x - long_org) / scale_u)


def get_v(x, lat_org, scale_v):
    return int((x - lat_org) / scale_v)


def get_patch(segmented_image, latitude, longitude, idx, theta):
    dst_x, dst_z = 1000, 2500
    angle = np.deg2rad(180 + theta)
    
    # Transforming Coordinates
    x, z = longitude, latitude 
    corner4 = np.array([[-dst_x, 0],
                        [ dst_x, 0], 
                        [ dst_x, dst_z], 
                        [-dst_x, dst_z]])

    R = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]])

    corner4t = np.floor(np.dot(R, corner4.T)).T
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

    # Removing extra gray Area around image
    gray = cv2.cvtColor(roit, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xt, yt, w, h = cv2.boundingRect(contours[0])
    roit = roit[:yt, :xt]

    roit = rotate(roit, 180)
    gray = cv2.cvtColor(roit, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xt, yt, w, h = cv2.boundingRect(contours[0])
    roit = roit[:yt, :xt]
    roit = rotate(roit, 180)

    return roit


def compute_birdviewbox(info_dict, shape, scale):
    h = info_dict['dim'][0] * scale
    w = info_dict['dim'][1] * scale
    l = info_dict['dim'][2] * scale
    x = info_dict['loc'][0] * scale * 2.0
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
    p = patches.PathPatch(pth, fill=True, color='orange', label='prediction')
    ax2.add_patch(p)


def get_bev(res, opt, segmented_image, latitude, longitude, cnt, theta):
    scale = 15
    birdimage = get_patch(segmented_image, latitude, longitude, cnt, theta)
    
    fig = plt.figure(figsize=birdimage.shape[:2] / 100)
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
    ax2.plot(shape_w / 2, 0, marker='+', markersize=16, markeredgecolor='red')

    # visualize bird eye view
    ax2.imshow(birdimage, origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # redraw the canvas
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(opt.video_h, opt.video_w, 3)

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
        [0, 0, 1]]
    )
    dist = np.array([-1.83418584,  12.2930625, -0.00434882103,  0.0226389517, -85.1805652])

    # undistort
    img = cv2.undistort(img, mtx, dist)
    return img
