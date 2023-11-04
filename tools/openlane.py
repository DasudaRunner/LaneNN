import sys, os
sys.path.append('/Users/haibo/workspace/code/SmartLane')
from utils.misc import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pdb
import random

def gen_openlane_list():
    json_dir = '/Users/haibo/workspace/dataset/lane3d_300/training'
    all_fpath = []
    for fpath in os.listdir(json_dir):
        if fpath == '.DS_Store':
            continue
        single_folder = []
        for sfapth in os.listdir(os.path.join(json_dir, fpath)):
            if not sfapth.endswith('.json'):
                continue
            single_folder.append(os.path.join(fpath, sfapth))
        random.shuffle(single_folder)
        all_fpath += single_folder[:1]
    save_list(all_fpath, 'openlane/curve_case.list')


if __name__ == '__main__':
    gen_openlane_list()
    os._exit(0)
    
    json_dir = '/Users/haibo/workspace/dataset/lane3d_300/training'
    all_fpath = []
    for fpath in os.listdir(json_dir):
        if fpath == '.DS_Store':
            continue
        for sfapth in os.listdir(os.path.join(json_dir, fpath)):
            if not sfapth.endswith('.json'):
                continue
            all_fpath.append(os.path.join(json_dir, fpath, sfapth))
            break
    
    all_line_num = []
    for s_fpath in all_fpath:
        json_data = load_json(s_fpath)
        all_lines = json_data['lane_lines']
        vis_lines = []
        all_type = []
        for sline in all_lines:
            type = sline['category']
            coord = sline['uv']
            vis_lines.append(np.array(coord))
            all_type.append(type)
        if len(vis_lines) == 0:
            continue
        all_line_num.append(len(vis_lines))
        
        # white_image = np.ones((1280, 1920, 3)) * 255
        # for _coord, _type in zip(vis_lines, all_type):
        #     for i in range(_coord.shape[1]):
        #         if _type in [20, 21]:
        #             c = [0, 0, 255]
        #         else:
        #             c = [255, 0, 0]
        #         white_image[int(_coord[1][i]), int(_coord[0][i]), :] = c
        
        # cv2.imshow('out', white_image)
        # cv2.waitKey()   
        
        # plt.clf()
        # plt.xlim([0, 1920])
        # plt.ylim([0, 1280])
        # for _vis in vis_lines:
        #     plt.scatter(_vis[0, :], 1280-_vis[1, :],s=3)
        
        # plt.savefig(f'results/{os.path.basename(s_fpath)}.png', dpi=400)
    
    plt.hist(all_line_num, bins=10)
    plt.show()
