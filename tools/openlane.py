import sys, os
sys.path.append('/Users/wanghaibo/workspace/code/SmartLane')
from utils.misc import *
import matplotlib.pyplot as plt
import numpy as np
import cv2

def gen_openlane_list():
    json_dir = '/Users/wanghaibo/workspace/temp/openlane/lane3d_300/test/curve_case'
    all_fpath = []
    for fpath in os.listdir(json_dir):
        if fpath == '.DS_Store':
            continue
        for sfapth in os.listdir(os.path.join(json_dir, fpath)):
            if not sfapth.endswith('.json'):
                continue
            all_fpath.append(os.path.join(fpath, sfapth))
    save_list(all_fpath, 'openlane/curve_case.list')


if __name__ == '__main__':
    # json_dir = '/Users/wanghaibo/workspace/temp/openlane/lane3d_300/test/curve_case'
    # all_fpath = []
    # for fpath in os.listdir(json_dir):
    #     if fpath == '.DS_Store':
    #         continue
    #     for sfapth in os.listdir(os.path.join(json_dir, fpath)):
    #         if not sfapth.endswith('.json'):
    #             continue
    #         all_fpath.append(os.path.join(json_dir, fpath, sfapth))
    #         break

    # for s_fpath in all_fpath:
    #     json_data = load_json(s_fpath)
    #     all_lines = json_data['lane_lines']
    #     vis_lines = []
    #     for sline in all_lines:
    #         type = sline['category']
    #         coord = sline['uv']
    #         vis_lines.append(np.array(coord))
    #     if len(vis_lines) == 0:
    #         continue
        
    #     # white_image = np.zeros((1280, 1920, 3))
    #     # for _vis in vis_lines:
    #     #     for i in range(len(_vis.shape[1])):
    #     #         white_image[i[0], i[1]] = 
        
    #     plt.clf()
    #     plt.xlim([0, 1920])
    #     plt.ylim([0, 1280])
    #     for _vis in vis_lines:
    #         plt.scatter(_vis[0, :], 1280-_vis[1, :],s=3)
        
    #     plt.savefig(f'results/{os.path.basename(s_fpath)}.png', dpi=400)
    
    gen_openlane_list()
