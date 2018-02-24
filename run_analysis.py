## general imports
import os, sys, json, datetime, jinja2
from jinja2 import Template

## COCO imports
from pycocotools.coco import COCO
from pycocotools.cocoanalyze import COCOanalyze

## Analysis API imports
from analysisAPI.errorsAPImpact import errorsAPImpact
from analysisAPI.localizationErrors import localizationErrors
from analysisAPI.scoringErrors import scoringErrors
from analysisAPI.backgroundFalsePosErrors import backgroundFalsePosErrors
from analysisAPI.backgroundFalseNegErrors import backgroundFalseNegErrors
from analysisAPI.occlusionAndCrowdingSensitivity import occlusionAndCrowdingSensitivity
from analysisAPI.sizeSensitivity import sizeSensitivity

def main():
    if len(sys.argv) != 6:
        raise ValueError("Please specify args: $> python run_analysis.py [annotations_path] [detections_path] [save_dir] [team_name] [version_name]")

    latex_jinja_env = jinja2.Environment(
        block_start_string    = '\BLOCK{',
        block_end_string      = '}',
        variable_start_string = '\VAR{',
        variable_end_string   = '}',
        comment_start_string  = '\#{',
        comment_end_string    = '}',
        line_statement_prefix = '%%',
        line_comment_prefix   = '%#',
        trim_blocks           = True,
        autoescape            = False,
        loader                = jinja2.FileSystemLoader(os.path.abspath('./latex/'))
    )
    template = latex_jinja_env.get_template('report_template.tex')
    template_vars  = {}

    annFile   = sys.argv[1]; splitName = annFile.split("/")[-1]
    resFile  = sys.argv[2]
    print("{:10}[{}]\n{:10}[{}]".format('annFile:',annFile,'resFile:',resFile))
    saveDir  = sys.argv[3]
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    teamName    = sys.argv[4]
    versionName = sys.argv[5]

    ## create dictionary with all images info
    gt_data   = json.load(open(annFile,'rb'))
    imgs_info = {i['id']:{'id'      :i['id'] ,
                          'width'   :i['width'],
                          'height'  :i['height'],
                          'coco_url':i['coco_url']}
                 for i in gt_data['images']}

    ## load team detections
    dt_data  = json.load(open(resFile,'rb'))
    team_dts = {}
    for d in dt_data:
        if d['image_id'] in team_dts: team_dts[d['image_id']].append(d)
        else: team_dts[d['image_id']] = [d]
    team_split_dts = []
    for img_id in team_dts:
        if img_id in imgs_info:
            team_split_dts.extend(sorted(team_dts[img_id], key=lambda k: -k['score'])[:20])
    print("Loaded [{}] detections from [{}] images.".format(len(team_split_dts),len(imgs_info)))
    template_vars['team_name']    = teamName
    template_vars['version_name'] = versionName
    template_vars['split_name']   = splitName
    template_vars['num_dts']      = len(team_split_dts)
    template_vars['num_imgs_dts'] = len(set([d['image_id'] for d in team_split_dts]))
    template_vars['num_imgs']     = len(imgs_info)

    ## load ground truth annotations
    coco_gt = COCO( annFile )

    ## initialize COCO detections api
    coco_dt   = coco_gt.loadRes( team_split_dts )

    ## initialize COCO analyze api
    coco_analyze = COCOanalyze(coco_gt, coco_dt, 'keypoints')
    if teamName == 'fakekeypoints100':
        imgIds  = sorted(coco_gt.getImgIds())[0:100]
        coco_analyze.cocoEval.params.imgIds = imgIds

    ## regular evaluation
    coco_analyze.evaluate(verbose=True, makeplots=True, savedir=saveDir, team_name=teamName)
    template_vars['overall_prc_medium'] = '%s/prc_[%s][medium][%d].pdf'%(saveDir,teamName,coco_analyze.params.maxDets[0])
    template_vars['overall_prc_large']  = '%s/prc_[%s][large][%d].pdf'%(saveDir,teamName,coco_analyze.params.maxDets[0])
    template_vars['overall_prc_all']    = '%s/prc_[%s][all][%d].pdf'%(saveDir,teamName,coco_analyze.params.maxDets[0])

    ############################################################################
    # COMMENT OUT ANY OF THE BELOW TO SKIP FROM ANALYSIS

    ## analyze imapct on AP of all error types
    paths = errorsAPImpact( coco_analyze, saveDir )
    template_vars.update(paths)

    ## analyze breakdown of localization errors
    paths = localizationErrors( coco_analyze, imgs_info, saveDir )
    template_vars.update(paths)

    ## analyze scoring errors
    paths = scoringErrors( coco_analyze, .75, imgs_info, saveDir )
    template_vars.update(paths)

    ## analyze background false positives
    paths = backgroundFalsePosErrors( coco_analyze, imgs_info, saveDir )
    template_vars.update(paths)

    ## analyze background false negatives
    paths = backgroundFalseNegErrors( coco_analyze, imgs_info, saveDir )
    template_vars.update(paths)

    ## analyze sensitivity to occlusion and crowding of instances
    paths = occlusionAndCrowdingSensitivity( coco_analyze, .75, saveDir )
    template_vars.update(paths)

    ## analyze sensitivity to size of instances
    paths = sizeSensitivity( coco_analyze, .75, saveDir )
    template_vars.update(paths)

    output_report = open('./%s_performance_report.tex'%teamName, 'w')
    output_report.write( template.render(template_vars) )
    output_report.close()

if __name__ == '__main__':
    main()
