## imports
import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.misc import imresize
import skimage.io as io
import utilities

def backgroundFalsePosErrors( coco_analyze, imgs_info, saveDir ):
    loc_dir = saveDir + '/background_errors/false_positives'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Background False Positives]\n\n")
    tic = time.time()
    paths = {}

    oksThrs     = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
    areaRngs    = [[32**2,1e5**2]]
    areaRngLbls = ['all']

    coco_analyze.params.areaRng    = areaRngs
    coco_analyze.params.areaRngLbl = areaRngLbls
    coco_analyze.params.oksThrs    = oksThrs
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []
    coco_analyze.analyze(check_kpts=False, check_scores=False, check_bckgd=True)

    badFalsePos = coco_analyze.false_pos_dts['all','0.5']
    for tind, t in enumerate(coco_analyze.params.oksThrs):
        badFalsePos = badFalsePos & coco_analyze.false_pos_dts['all',str(t)]
    fp_dts = [d for d in coco_analyze.corrected_dts['all'] if d['id'] in badFalsePos]

    f.write("Num. detections: [%d]\n"%len(coco_analyze.corrected_dts['all']))
    for oks in oksThrs:
        f.write("OKS thresh: [%f]\n"%oks)
        f.write(" - Matches:      [%d]\n"%len(coco_analyze.bckgd_err_matches[areaRngLbls[0], str(oks), 'dts']))
        f.write(" - Bckgd. FP:    [%d]\n"%len(coco_analyze.false_pos_dts[areaRngLbls[0],str(oks)]))

    sorted_fps = sorted(fp_dts, key=lambda k: -k['score'])
    show_fp = sorted_fps[0:4] + sorted_fps[-4:]
    f.write("\nBackground False Positive Errors:\n")
    for tind, t in enumerate(show_fp):
        name    = 'bckd_false_pos_%d'%tind
        paths[name] = "%s/%s.pdf"%(loc_dir,name)
        f.write("Image_id, detection_id, score: [%d][%d][%.3f]\n"%(t['image_id'],t['id'],t['score']))
        utilities.show_dets([t],[],imgs_info[t['image_id']],paths[name])

    a = [d['score'] for d in coco_analyze.corrected_dts['all']]
    p_20 = np.percentile(a, 20); p_40 = np.percentile(a, 40)
    p_60 = np.percentile(a, 60); p_80 = np.percentile(a, 80)

    f.write("\nPercentiles of the scores of all Detections:\n")
    f.write(" - 20th perc. score:[%.3f]; num. dts:[%d]\n"%(p_20,len([d for d in coco_analyze.corrected_dts['all'] if d['score']<=p_20])))
    f.write(" - 40th perc. score:[%.3f]; num. dts:[%d]\n"%(p_40,len([d for d in coco_analyze.corrected_dts['all'] if d['score']<=p_40])))
    f.write(" - 60th perc. score:[%.3f]; num. dts:[%d]\n"%(p_60,len([d for d in coco_analyze.corrected_dts['all'] if d['score']<=p_60])))
    f.write(" - 80th perc. score:[%.3f]; num. dts:[%d]\n"%(p_80,len([d for d in coco_analyze.corrected_dts['all'] if d['score']<=p_80])))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('lightgray')
    bins = [min(a),p_20,p_40,p_60,p_80,max(a)]
    plt.hist([d['score'] for d in sorted_fps],bins=bins,color='b')
    plt.xticks(bins,['','20th%','40th%','60th%','80th%',''],rotation='vertical')
    plt.grid()
    plt.xlabel('All Detection Score Percentiles',fontsize=20)
    plt.title('Histogram of False Positive Scores',fontsize=20)
    path = "%s/bckd_false_pos_scores_histogram.pdf"%(loc_dir)
    paths['false_pos_score_hist'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    high_score_fp = [d for d in sorted_fps if p_80 <= d['score']]

    max_height = max([d['bbox'][3] for d in high_score_fp])
    min_height = min([d['bbox'][3] for d in high_score_fp])
    max_width = max([d['bbox'][2] for d in high_score_fp])
    min_width = min([d['bbox'][2] for d in high_score_fp])

    f.write("\nBackground False Positives Bounding Box Dimenstions:\n")
    f.write(" - Min width:  [%d]\n"%min_width)
    f.write(" - Max width:  [%d]\n"%max_width)
    f.write(" - Min height: [%d]\n"%min_height)
    f.write(" - Max height: [%d]\n"%max_height)

    ar_pic = np.zeros((int(max_height)+1,int(max_width)+1))
    ar_pic_2 = np.zeros((30,30))
    ar_bins = range(10)+range(10,100,10)+range(100,1000,100)+[1000]
    ar_pic_3 = np.zeros((10,10))
    ar_bins_3 = [np.power(2,x) for x in xrange(11)]

    areaRngs    = [[0, 32 ** 2],[32 ** 2, 64 ** 2],[64 ** 2, 96 ** 2],[96 ** 2, 128 ** 2],[128 ** 2, 1e5 ** 2]]
    areaRngLbls = ['small','medium','large','xlarge','xxlarge']
    small = 0; medium = 0; large = 0; xlarge = 0; xxlarge = 0

    num_people_ranges = [[0,0],[1,1],[2,4],[5,8],[9,100]]
    num_people_labels = ['none','one','small grp.','large grp.', 'crowd']
    no_people = 0; one = 0; small_grp = 0; large_grp = 0; crowd = 0

    for t in high_score_fp:
        t_width  = int(t['bbox'][2])
        t_height = int(t['bbox'][3])
        ar_pic[0:t_height,0:t_width] += 1

        if t_width < 1024 and t_height < 1024:
            col = [i for i in xrange(len(ar_bins)-1) if ar_bins[i]<t_width<ar_bins[i+1]]
            row = [i for i in xrange(len(ar_bins)-1) if ar_bins[i]<t_height<ar_bins[i+1]]
            ar_pic_2[row,col] += 1

            col = [i for i in xrange(len(ar_bins_3)-1) if ar_bins_3[i]<t_width<ar_bins_3[i+1]]
            row = [i for i in xrange(len(ar_bins_3)-1) if ar_bins_3[i]<t_height<ar_bins_3[i+1]]
            ar_pic_3[row,col] += 1
        else:
            print "False Positive bbox has a side larger than 1024 pixels."
            print "Change lists ar_bins_2 and ar_bins_3 to include larger bins."
            assert(False)

        area = t_width * t_height * .5
        if areaRngs[0][0] <= area < areaRngs[0][1]:
            small += 1
        elif areaRngs[1][0] <= area < areaRngs[1][1]:
            medium += 1
        elif areaRngs[2][0] <= area < areaRngs[2][1]:
            large += 1
        elif areaRngs[3][0] <= area < areaRngs[3][1]:
            xlarge += 1
        elif areaRngs[4][0] <= area < areaRngs[4][1]:
            xxlarge += 1

        anns = coco_analyze.cocoGt.loadAnns(coco_analyze.cocoGt.getAnnIds(t['image_id']))
        iscrowd = [ann['iscrowd'] for ann in anns]
        num_people = len(anns) if sum(iscrowd)==0 else 100
        if num_people_ranges[0][0] <= num_people <= num_people_ranges[0][1]:
            no_people += 1
        elif num_people_ranges[1][0] <= num_people <= num_people_ranges[1][1]:
            one += 1
        elif num_people_ranges[2][0] <= num_people <= num_people_ranges[2][1]:
            small_grp += 1
        elif num_people_ranges[3][0] <= num_people <= num_people_ranges[3][1]:
            large_grp += 1
        elif num_people_ranges[4][0] <= num_people <= num_people_ranges[4][1]:
            crowd += 1

    f.write("\nNumber of people in images with Background False Positives:\n")
    f.write(" - No people:         [%d]\n"%no_people)
    f.write(" - One person:        [%d]\n"%one)
    f.write(" - Small group (2-4): [%d]\n"%small_grp)
    f.write(" - Large Group (5-8): [%d]\n"%large_grp)
    f.write(" - Crowd       (>=9): [%d]\n"%crowd)

    f.write("\nArea size (in pixels) of Background False Positives:\n")
    f.write(" - Small    (%d,%d):            [%d]\n"%(areaRngs[0][0],areaRngs[0][1],small))
    f.write(" - Medium   (%d,%d):         [%d]\n"%(areaRngs[1][0],areaRngs[1][1],medium))
    f.write(" - Large    (%d,%d):         [%d]\n"%(areaRngs[2][0],areaRngs[2][1],large))
    f.write(" - X-Large  (%d,%d):        [%d]\n"%(areaRngs[3][0],areaRngs[3][1],xlarge))
    f.write(" - XX-Large (%d,%d): [%d]\n"%(areaRngs[4][0],areaRngs[4][1],xxlarge))

    plt.figure(figsize=(10,10))
    plt.imshow(ar_pic,origin='lower')
    plt.colorbar()
    plt.title('BBox Aspect Ratio',fontsize=20)
    plt.xlabel('Width (px)',fontsize=20)
    plt.ylabel('Height (px)',fontsize=20)
    path = "%s/bckd_false_pos_bbox_aspect_ratio.pdf"%(loc_dir)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(ar_pic_2,origin='lower')
    plt.xticks(xrange(1,len(ar_bins)+1),["%d"%(x) for x in ar_bins],rotation='vertical')
    plt.yticks(xrange(1,len(ar_bins)+1),["%d"%(x) for x in ar_bins])
    plt.colorbar()
    plt.grid()
    plt.title('BBox Aspect Ratio',fontsize=20)
    plt.xlabel('Width (px)',fontsize=20)
    plt.ylabel('Height (px)',fontsize=20)
    path = "%s/bckd_false_pos_bbox_aspect_ratio_2.pdf"%(loc_dir)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(ar_pic_3,origin='lower')
    plt.xticks([-.5 + x for x in range(11)],["%d"%(x) for x in ar_bins_3])
    plt.yticks([-.5 + x for x in range(11)],["%d"%(x) for x in ar_bins_3])
    plt.colorbar()
    plt.grid()
    plt.title('BBox Aspect Ratio',fontsize=20)
    plt.xlabel('Width (px)',fontsize=20)
    plt.ylabel('Height (px)',fontsize=20)
    path = "%s/bckd_false_pos_bbox_aspect_ratio_3.pdf"%(loc_dir)
    paths['false_pos_bbox_ar'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('lightgray')
    plt.bar(xrange(5),[small,medium,large,xlarge,xxlarge],color='g',align='center')
    plt.xticks(xrange(5),areaRngLbls)
    plt.grid()
    plt.title('Histogram of Area Size',fontsize=20)
    path = "%s/bckd_false_pos_area_histogram.pdf"%(loc_dir)
    paths['false_pos_bbox_area_hist'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('lightgray')
    plt.bar(xrange(5),[no_people,one,small_grp,large_grp,crowd],color='g',align='center')
    plt.xticks(xrange(5),num_people_labels)
    plt.grid()
    plt.title('Histogram of Num. of People in Images',fontsize=20)
    path = "%s/bckd_false_pos_num_people_histogram.pdf"%(loc_dir)
    paths['false_pos_num_ppl_hist'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

    return paths
