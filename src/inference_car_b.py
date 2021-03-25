from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from detector import Detector

from bev import *
import warnings
warnings.simplefilter('ignore', np.RankWarning)

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


def demo(opt):
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Read Satellite Metadata
    sat_img = cv2.cvtColor(cv2.imread(opt.map), cv2.COLOR_BGR2RGB)
    segmented_image = get_segmented_map(sat_img)

    # Setting GPS coordinates
    csv_list = []
    p1 = (opt.map_lat_top, opt.map_lon_top)
    p2 = (opt.map_lat_bot, opt.map_lon_bot)
    lat_org, long_org, scale_u, scale_v = get_map_vectors(p1, p2, segmented_image)
    
    for i in range(opt.csv):
        csv_list.append(pd.read_csv(opt.base_dir + '1_' + str(i+1) + '_all.csv', encoding = "ISO-8859-1")[['latitude', 'longitude']])
    csv = pd.concat(csv_list)
    csv.reset_index(drop=True, inplace=True)

    # Transformation
    csvt = csv.copy()
    csvt['longitude'] = csvt['longitude'].apply(lambda x: get_u(x, long_org, scale_u))
    csvt['latitude'] = csvt['latitude'].apply(lambda x: get_v(x, lat_org, scale_v))

    # Loading metadata
    results = json.load('./metadata/{}_results.json'.format(opt.exp_id + '_' + opt.json_name))

    cnt = 0
    cnt_max = csv.shape[0]
    out2_init = False

    while True:
        if cnt >= cnt_max - opt.fpts:
            break
        
        cnt += 1

        # skip the first X frames of the video
        if cnt < opt.skip_first:
            continue

        # save debug image to video
        if opt.save_video:

            # Getting Slope
            scale = 30
            dst_x, dst_z = 1000, 2500
            lon = csv.loc[cnt-1:cnt-1+opt.fpts, 'longitude'].to_numpy()
            lat = csv.loc[cnt-1:cnt-1+opt.fpts, 'latitude'].to_numpy()
            theta = 180 + slope_calculator(lon, lat)
            
            # Getting BEV Image
            latitude = csvt.loc[cnt-1, 'latitude']
            longitude = csvt.loc[cnt-1, 'longitude']
            birdimage = get_patch(segmented_image, latitude, longitude, theta, dst_x, dst_z)
            img = get_bev(results[cnt], opt, scale, birdimage)

            if not out2_init:
                vw, vh = img.shape[0], img.shape[1]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out2 = cv2.VideoWriter('./results/{}_bev.avi'.format(
                    opt.exp_id + '_' + 'recreation'), fourcc, opt.save_framerate, (vh, vw))
                out2_init = True
            
            comb = cv2.resize(img, (vh, vw))
            out2.write(comb)

    out2.release()


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)