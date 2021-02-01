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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


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
  corners_2D[0] += int(shape/2)
  corners_2D = (corners_2D).astype(np.int16)
  corners_2D = corners_2D.T

  return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, info_dict, shape, scale):
    pred_corners_2d = compute_birdviewbox(info_dict, shape=shape, scale=scale)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)

def get_bev(res, opt):
    fig = plt.figure(figsize=(opt.video_w / 100, opt.video_h / 100), dpi=100)
    ax2 = fig.add_subplot()
    
    shape_w = opt.video_w
    shape_h = opt.video_h
    
    scale = 10
    birdimage = np.zeros((shape_h, shape_w, 3), np.uint8)
    
    for index in range(len(res)):
        draw_birdeyes(ax2, res[index], shape=shape_w, scale=scale)
        
    # plot camera view range
    x1 = np.linspace(0, shape_w / 2, 100)
    x2 = np.linspace(shape_w / 2, shape_w, 100)
    y1 = np.linspace(shape_h / 2, 0, 100)
    y2 = np.linspace(0, shape_h / 2, 100)

    ax2.plot(x1, y1, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(x2, y2,  ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(shape_w / 2, 0, marker='+', markersize=16, markeredgecolor='red')
        
    # visualize bird eye view
    ax2.imshow(birdimage, origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # redraw the canvas
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(opt.video_h, opt.video_w, 3)

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.close()
    
    return img


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  if not os.path.exists('../results'):
        os.makedirs('../results')

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

  # Initialize output video
  out = None
  out_name = opt.demo[opt.demo.rfind('/') + 1:]
  print('out_name', out_name)
  if opt.save_video:
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('../results/{}.avi'.format(
      opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
        opt.video_w, opt.video_h))
    if opt.bev:
        out2 = cv2.VideoWriter('../results/{}_bev.avi'.format(
          opt.exp_id + '_' + out_name),fourcc, opt.save_framerate, (
            2 * opt.video_w, opt.video_h))
  
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}
  bev_boxes = {} 

  while True:
      if is_video:
        _, img = cam.read()
        if img is None:
          save_and_exit(opt, out, results, out_name)
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
        
        if opt.bev:
            # Getting BEV
            img = get_bev(results[cnt], opt)

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
          cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])
      
      # esc to quit and finish saving video
      if cv2.waitKey(1) == 27:
        save_and_exit(opt, out, results, out_name)
        return 

  out2.release()
  save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
  if opt.save_results and (results is not None):
    save_dir =  '../results/{}_results.json'.format(opt.exp_id + '_' + out_name)
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
