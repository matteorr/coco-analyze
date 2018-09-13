## imports
import os, sys, time, json
import numpy as np
from colour import Color
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from scipy.misc import imresize
import skimage.io as io

"""
Utility functions
"""
num_kpts  = 17
oks       = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
sqrt_neg_log_oks = np.sqrt(-2*np.log(oks))
sigmas    = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
variances = (sigmas * 2)**2
skeleton  = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
colors    = {(0,1): '#cd87ff', (0,2): '#cd87ff', (1,2): '#cd87ff', (1,3): '#cd87ff', (2,4): '#cd87ff',
            (3,5): '#74c8f9', (4,6): '#74c8f9', (5,6): '#feff95', (5,7): '#74c8f9', (5,11): '#feff95',
            (6,8): '#74c8f9', (6,12): '#feff95',(7,9): '#74c8f9', (8,10): '#74c8f9',(11,12): '#feff95',
            (13,11): '#a2805b',(14,12): '#a2805b',(15,13): '#a2805b',(16,14): '#a2805b'}

def show_dets(coco_dts, coco_gts, img_info, save_path=None):
    if len(coco_dts) == 0 and len(coco_gts)==0:
        return 0

    I = io.imread(img_info['coco_url'])
    plt.figure(figsize=(10,10)); plt.axis('off')
    plt.imshow(I)
    ax = plt.gca(); ax.set_autoscale_on(False)
    polygons = []; color = []

    for ann in coco_gts:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        if 'keypoints' in ann and type(ann['keypoints']) == list:
            # turn skeleton into zero-based index
            sks = np.array(skeleton)-1
            kp = np.array(ann['keypoints'])
            x = kp[0::3]; y = kp[1::3]; v = kp[2::3]
            for sk in sks:
                if np.all(v[sk]>0):
                    plt.plot(x[sk],y[sk], linewidth=3, color='green')

            plt.plot(x[v>0], y[v>0],'o',markersize=2, markerfacecolor='green',
                                        markeredgecolor='k',markeredgewidth=3)
            plt.plot(x[v>1], y[v>1],'o',markersize=2, markerfacecolor='green',
                                        markeredgecolor='green', markeredgewidth=2)

            for x1, y1, sigma1 in zip(x[v>0], y[v>0], sigmas[v>0]):
                r = sigma1 * (np.sqrt(ann["area"])+np.spacing(1))
                circle = plt.Circle((x1,y1), sqrt_neg_log_oks[0]*r, fc=(1,0,0,0.4),ec='k')
                ax.add_patch(circle)
                for a1 in sqrt_neg_log_oks[1:]:
                    circle = plt.Circle((x1,y1), a1*r, fc=(0,0,0,0),ec='k')
                    ax.add_patch(circle)

        if len(coco_dts)==0 and len(coco_gts)==1:
            bbox  = ann['bbox']
            rect = plt.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],fill=False,edgecolor=[1, .6, 0],linewidth=3)
            ax.add_patch(rect)
            title = "[%d][%d][%d]"%(coco_gts[0]['image_id'],coco_gts[0]['id'],coco_gts[0]['num_keypoints'])
            plt.title(title,fontsize=20)

    for ann in coco_dts:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        sks = np.array(skeleton)-1
        kp = np.array(ann['keypoints'])
        x = kp[0::3]; y = kp[1::3]; v = kp[2::3]
        for sk in sks:
            plt.plot(x[sk],y[sk], linewidth=3, color=colors[sk[0],sk[1]])

        for kk in range(17):
            if kk in [1,3,5,7,9,11,13,15]:
                plt.plot(x[kk], y[kk],'o',markersize=5, markerfacecolor='r',
                                              markeredgecolor='r', markeredgewidth=3)
            elif kk in [2,4,6,8,10,12,14,16]:
                plt.plot(x[kk], y[kk],'o',markersize=5, markerfacecolor='g',
                                              markeredgecolor='g', markeredgewidth=3)
            else:
                plt.plot(x[kk], y[kk],'o',markersize=5, markerfacecolor='b',
                                              markeredgecolor='b', markeredgewidth=3)

        bbox  = ann['bbox']; score = ann['score']
        rect = plt.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],fill=False,edgecolor=[1, .6, 0],linewidth=3)

        if len(coco_dts)==1:
            if len(coco_gts)==0:
                title = "[%d][%d][%.3f]"%(coco_dts[0]['image_id'],coco_dts[0]['id'],coco_dts[0]['score'])
                plt.title(title,fontsize=20)

            if len(coco_gts)==1:
                oks = compute_kpts_oks(coco_dts[0]['keypoints'], coco_gts[0]['keypoints'],coco_gts[0]['area'])
                title = "[%.3f][%.3f][%d][%d][%d]"%(score,oks,coco_gts[0]['image_id'],coco_gts[0]['id'],coco_dts[0]['id'])
                plt.title(title,fontsize=20)

        else:
            ax.annotate("[%.3f][%.3f]"%(score,0), (bbox[0]+bbox[2]/2.0, bbox[1]-5),
                    color=[1, .6, 0], weight='bold', fontsize=12, ha='center', va='center')
        ax.add_patch(rect)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
    ax.add_collection(p)

    if save_path:
        plt.savefig(save_path,bbox_inches='tight',dpi=50)
    else:
        plt.show()
    plt.close()

def compute_kpts_oks(dt_kpts, gt_kpts, area):
    # this function only works for computing oks with keypoints
    g = np.array(gt_kpts); xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
    assert( np.count_nonzero(vg > 0) > 0)
    d = np.array(dt_kpts); xd = d[0::3]; yd = d[1::3]

    dx = xd - xg; dy = yd - yg
    e = (dx**2 + dy**2) / variances / (area+np.spacing(1)) / 2
    e=e[vg > 0]

    return np.sum(np.exp(-e)) / e.shape[0]

def compute_oks(dts, gts):
    if len(dts) * len(gts) == 0:
        return np.array([])
    oks_mat = np.zeros((len(dts), len(gts)))

    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((len(sigmas)))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / variances / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            oks_mat[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return oks_mat

def compute_iou(bbox_1, bbox_2):

    x1_l = bbox_1[0]
    x1_r = bbox_1[0] + bbox_1[2]
    y1_t = bbox_1[1]
    y1_b = bbox_1[1] + bbox_1[3]
    w1   = bbox_1[2]
    h1   = bbox_1[3]

    x2_l = bbox_2[0]
    x2_r = bbox_2[0] + bbox_2[2]
    y2_t = bbox_2[1]
    y2_b = bbox_2[1] + bbox_2[3]
    w2   = bbox_2[2]
    h2   = bbox_2[3]

    xi_l = max(x1_l, x2_l)
    xi_r = min(x1_r, x2_r)
    yi_t = max(y1_t, y2_t)
    yi_b = min(y1_b, y2_b)

    width  = max(0, xi_r - xi_l)
    height = max(0, yi_b - yi_t)
    a1 = w1 * h1
    a2 = w2 * h2

    if float(a1 + a2 - (width * height)) == 0:
        return 0
    else:
        iou = (width * height) / float(a1 + a2 - (width * height))

    return iou

def compute_ious(anns):
    num_boxes = len(anns)
    ious = np.zeros((num_boxes, num_boxes))

    for i in range(num_boxes):
        for j in range(i,num_boxes):
            ious[i,j] = compute_iou(anns[i]['bbox'],anns[j]['bbox'])
            if i!=j:
                ious[j,i] = ious[i,j]
    return ious
