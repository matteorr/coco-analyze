## system imports
import sys, os
import time
import copy

## loading imports
import json

## math imports
import numpy as np

## COCO imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from pycocotools.cocoanalyze import COCOanalyze

## plotting imports
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import skimage.io as io

'''
Set paths to data and load data
'''
## set paths of ground truth and detections
BASE_DIR = '.'
COCO_GT  = BASE_DIR + '/data/person_keypoints_anns.json'
SAVE_DIR = BASE_DIR + '/outputs'

TEAM_NAME = 'grmi'
DTS       = BASE_DIR + '/data/%s.json'%TEAM_NAME

print "\n==============================================="
print "COCO_GT:      [%s]"%COCO_GT
print "DTS:          [%s]"%DTS
print "SAVE_DIR:     [%s]"%SAVE_DIR
print "===============================================\n"

## load coco ground-truth
coco_gt = COCO( COCO_GT )
print("{:15}[{}] instances in [{}] images.".format('Ground-truth:',
                                                   len(coco_gt.getAnnIds()),
                                                   len(coco_gt.getImgIds())))

## create imgs_info dictionary
with open(COCO_GT,'rb') as fp:
    data = json.load(fp)
imgs_info = {i['id']:{'id':i['id'] ,
                      'width':i['width'],
                      'height':i['height']}
                       for i in data['images']}
assert(len(coco_gt.getImgIds())==len(imgs_info))

## load team detections
with open(DTS,'rb') as fp: team_dts = json.load(fp)
team_dts     = [d for d in team_dts if d['image_id'] in imgs_info]
team_img_ids = set([d['image_id'] for d in team_dts])
print("{:15}[{}] instances in [{}] images.".format('Detections:',
                                                   len(team_dts),
                                                   len(team_img_ids)))

## create detection object
coco_dt   = coco_gt.loadRes(team_dts)

## create analyze object
coco_analyze = COCOanalyze(coco_gt,coco_dt,'keypoints')

oksThrs     = .7
areaRngs    = [[32**2, 1e5**2],[32**2, 96**2],[96**2, 1e5**2]]
areaRngLbls = ['all','medium','large']

print "================================================"
print "[%s,%s,%s]"%(oks,areaRng,areaRngLbls[aind])
print "================================================"

coco_analyze.params.oksThrs    = [oksThrs]
coco_analyze.params.areaRng    = [areaRngs[0]]
coco_analyze.params.areaRngLbl = [areaRngLbls[0]]

coco_analyze.analyze(check_kpts=True, check_scores=True, check_false=True)
coco_analyze.summarize(makeplots=True,savedir=SAVE_DIR,team_name=TEAM_NAME)
results = coco_analyze.stats

matches = coco_analyze.matches
with open('{}/dt_gt_matches_[{}][{}][{}][{}].json'.format(SAVE_DIR, TEAM_NAME,
            areaRngLbls[aind],
            coco_analyze.params.maxDets[0],
            int(100*oks)), 'wb') as fp:
    json.dump(matches,fp)

corrected_dts = coco_analyze.corrected_dts
with open('{}/corrected_dts_[{}][{}][{}][{}].json'.format(SAVE_DIR, TEAM_NAME,
            areaRngLbls[aind],
            coco_analyze.params.maxDets[0],
            int(100*oks)), 'wb') as fp:
    json.dump(corrected_dts,fp)
        
false_neg_gts = coco_analyze.false_neg_gts
with open('{}/false_neg_gts_[{}][{}][{}][{}].json'.format(SAVE_DIR, TEAM_NAME,
            areaRngLbls[aind],
            coco_analyze.params.maxDets[0],
            int(100*oks)), 'wb') as fp:
    json.dump(false_neg_gts,fp)

print results

with open('%s/error_stats_%s.json'%(SAVE_DIR,TEAM_NAME),'wb') as fp:
    json.dump(results,fp)
