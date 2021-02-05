# Added Script for Birds Eye Transformation
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec


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


def get_bev(res, opt):
    fig = plt.figure(figsize=(opt.video_w / 100, opt.video_h / 100), dpi=100)
    ax2 = fig.add_subplot()

    shape_w = opt.video_w
    shape_h = opt.video_h

    scale = 15
    birdimage = np.zeros((shape_h, shape_w, 3), np.uint8)

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
