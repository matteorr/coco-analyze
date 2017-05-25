## imports
import os, sys, json
from matplotlib.backends.backend_pdf import PdfPages
## COCO imports
from pycocotools.coco import COCO
from pycocotools.cocoanalyze import COCOanalyze
## Analysis API imports
import analysisAPI
import datetime

def main():
    if len(sys.argv) != 5:
        raise ValueError("Please specify args: $> python run_analysis.py [annotations_path] [results_path] [save_dir] [team_name]")

    annFile  = sys.argv[1]; resFile  = sys.argv[2]
    print("{:10}[{}]\n{:10}[{}]".format('annFile:',annFile,'resFile:',resFile))
    saveDir  = sys.argv[3]
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    teamName = sys.argv[4]

    ## create dictionary with all images info
    gt_data   = json.load(open(annFile,'rb'))
    imgs_info = {i['id']:{'id':i['id'] ,
                          'width':i['width'],
                          'height':i['height']}
                           for i in gt_data['images']}

    ## load team detections
    team_dts = json.load(open(resFile,'rb'))
    team_dts = [d for d in team_dts if d['image_id'] in imgs_info]
    team_img_ids = set([d['image_id'] for d in team_dts])
    team_dts_dict = {}
    for d in team_dts:
        if d['image_id'] in team_dts_dict:
            team_dts_dict[d['image_id']].append(d)
        else:
            team_dts_dict[d['image_id']] = [d]
    pruned_team_dts = []
    for iid in team_dts_dict:
        pruned_team_dts.extend(sorted(team_dts_dict[iid], key=lambda k: -k['score'])[:20])
    team_dts = pruned_team_dts
    print("Loaded [{}] detections from [{}] images.".format(len(team_dts),len(imgs_info)))
    # # suppress the detections to be only 20 per image
    # team_img_dts = {}
    # for d in team_dts:
    #     if d['image_id'] in team_img_dts:
    #         team_img_dts[d['image_id']].append(d)
    #     else:
    #         team_img_dts[d['image_id']] = [d]
    # print len(team_img_dts)
    # suppressed_team_dts = []
    # for i in team_img_dts:
    #     top_dts = sorted(team_img_dts[i], key=lambda k: -k['score'])
    #     assert(top_dts[0]['score']>=top_dts[-1]['score'])
    #     suppressed_team_dts.extend(top_dts[:20])
    # print("Loaded [{}] instances in [{}] images.".format(len(suppressed_team_dts),len(imgs_info)))

    ## load ground truth annotations
    coco_gt = COCO( annFile )

    ## initialize COCO detections api
    coco_dt   = coco_gt.loadRes( team_dts )

    ## initialize COCO analyze api
    coco_analyze = COCOanalyze(coco_gt, coco_dt, 'keypoints')
    if teamName == 'fakekeypoints100':
        imgIds  = sorted(coco_gt.getImgIds())[0:100]
        coco_analyze.cocoEval.params.imgIds = imgIds

    #coco_analyze.evaluate(verbose=True, makeplots=True, savedir=saveDir, team_name=teamName)
    with PdfPages('%s/summary.pdf'%saveDir) as pdf:
        # plot specialized analysis of keypoint estimation errors
        #analysisAPI.errorsAUCImpact( coco_analyze, saveDir, pdf )
        analysisAPI.localizationKeypointBreakdown( coco_analyze, saveDir, pdf )
        #analysisAPI.localizationOKSImpact( coco_analyze, .75, saveDir, pdf )
        #analysisAPI.backgroundCharacteristics( coco_analyze, .5, imgs_info, saveDir )
        #analysisAPI.occlusionAndCrowdingSensitivity( coco_analyze, .75, saveDir )
        #analysisAPI.sizeSensitivity( coco_analyze, .75, saveDir )

        d = pdf.infodict()
        d['Title']        = ''
        d['Author']       = ''
        d['Subject']      = ''
        d['Keywords']     = ''
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate']      = datetime.datetime.today()

if __name__ == '__main__':
    main()
