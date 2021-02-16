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


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    detector = Detector(opt)

    if not os.path.exists('./results'):
        os.makedirs('./results')

    if opt.demo == 'webcam' or \
        opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        is_video = True
        # demo on video stream
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    else:
        is_video = False
        # Demo on images sequences
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

    # Read Satellite Metadata
    sat_img = cv2.cvtColor(cv2.imread('satmap/negley_map_2.png'), cv2.COLOR_BGR2RGB)
    segmented_image = get_segmented_map(img)

    p1 = (40.469862, -79.930679)
    p2 = (40.466952, -79.926014)
    lat_org, long_org, scale_u, scale_v = get_map_vectors(p1, p2, img)

    csv_1 = pd.read_csv('satmap/1_1_all.csv', encoding = "ISO-8859-1")[['latitude', 'longitude']]
    csv_2 = pd.read_csv('satmap/1_2_all.csv', encoding = "ISO-8859-1")[['latitude', 'longitude']]
    csv_3 = pd.read_csv('satmap/1_3_all.csv', encoding = "ISO-8859-1")[['latitude', 'longitude']]

    # Transformation
    csv_1['longitude'] = csv_1['longitude'].apply(lambda x: get_u(x, long_org, scale_u))
    csv_1['latitude'] = csv_1['latitude'].apply(lambda x: get_v(x, lat_org, scale_v))

    csv_2['longitude'] = csv_2['longitude'].apply(lambda x: get_u(x, long_org, scale_u))
    csv_2['latitude'] = csv_2['latitude'].apply(lambda x: get_v(x, lat_org, scale_v))

    csv_3['longitude'] = csv_3['longitude'].apply(lambda x: get_u(x, long_org, scale_u))
    csv_3['latitude'] = csv_3['latitude'].apply(lambda x: get_v(x, lat_org, scale_v))

    # Initialize output video
    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:]
    print('out_name', out_name)
    if opt.save_video:
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./results/{}.avi'.format(
            opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
            opt.video_w, opt.video_h))

        # CHANGE: Added if statement for Bird's Eye Transformation
        if opt.bev:
            out2 = cv2.VideoWriter('./results/{}_bev.avi'.format(
                opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
                2 * opt.video_w, opt.video_h))

    if opt.debug < 5:
        detector.pause = False
    cnt = 0
    results = {}

    while True:
        if is_video:
            _, img = cam.read()
            if img is None:
                save_and_exit(opt, out, results, out_name)
            img = undistort_image(img)
        else:
            if cnt < len(image_names):
                img = cv2.imread(image_names[cnt])
            else:
                save_and_exit(opt, out, results, out_name)
        cnt += 1

        # resize the original video for saving video results
        if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))

        # skip the first X frames of the video
        if cnt < opt.skip_first:
            continue

        # CHANGE: Commented below line for Colab Compactibility
        # cv2.imshow('input', img)

        # track or detect the image.
        ret = detector.run(img)

        # log run time
        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)

        # results[cnt] is a list of dicts:
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        results[cnt] = ret['results']

        # save debug image to video
        if opt.save_video:
            out.write(ret['generic'])

            # CHANGE: Added 'if' statement for Bird's Eye Transformation
            if opt.bev:
                # Getting BEV

                theta = 115
                latitude, longitude = csv_1.loc[cnt, 'latitude'], csv_1.loc[cnt, 'longitude']
                img = get_bev(results[cnt], opt, segmented_image, latitude, longitude, cnt, theta)

                # Writing to Video
                rows_rgb, cols_rgb, channels = ret['generic'].shape
                rows_gray, cols_gray, _ = img.shape
                rows_comb = max(rows_rgb, rows_gray)
                cols_comb = cols_rgb + cols_gray
                comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)
                comb[:rows_rgb, :cols_rgb] = ret['generic']
                comb[:rows_gray, cols_rgb:] = img
                out2.write(comb)

            if not is_video:
                cv2.imwrite('./results/demo{}.jpg'.format(cnt), ret['generic'])

        # esc to quit and finish saving video
        if cv2.waitKey(1) == 27:
            save_and_exit(opt, out, results, out_name)
            return

    out2.release()
    save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
    if opt.save_results and (results is not None):
        save_dir = './results/{}_results.json'.format(opt.exp_id + '_' + out_name)
        print('saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)),
                  open(save_dir, 'w'))
    if opt.save_video and out is not None:
        out.release()
    sys.exit(0)


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
