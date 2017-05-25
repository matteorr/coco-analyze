## imports
import os, sys, time, json
import numpy as np
from colour import Color
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.misc import imresize
import skimage.io as io

def errorsAUCImpact( coco_analyze, saveDir, pdf=None ):
    loc_dir = saveDir + '/AUC_impact'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
        os.makedirs(loc_dir+'/all_plots')
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [AUC Impact]\n\n")
    tic = time.time()

    oksThrs     = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
    areaRngs    = [[32**2,1e5**2],[32**2,96**2],[96**2,1e5**2]]
    areaRngLbls = ['all','medium','large']
    err_types   = ['miss','swap', 'inversion', 'jitter', 'score', 'bckgd_false_pos', 'false_neg']
    err_colors  = ['#F2E394', '#F2AE72','#D96459', '#8C4646', '#4F82BD', '#8063A3','seagreen']

    coco_analyze.params.areaRng    = areaRngs
    coco_analyze.params.areaRngLbl = areaRngLbls
    coco_analyze.params.oksThrs    = oksThrs
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    stats = []
    for eind, err in enumerate(err_types):
        if err in ['miss','swap', 'inversion', 'jitter']:
            coco_analyze.params.err_types = [err]
            coco_analyze.analyze(check_kpts=True, check_scores=False, check_bckgd=False)
        if err == 'score':
            coco_analyze.params.err_types = []
            coco_analyze.analyze(check_kpts=False, check_scores=True, check_bckgd=False)
        if err == 'bckgd_false_pos':
            coco_analyze.params.err_types = []
            coco_analyze.analyze(check_kpts=False, check_scores=False, check_bckgd=True)
        if err == 'false_neg': continue

        coco_analyze.summarize(makeplots=True, savedir=loc_dir+'/all_plots', team_name=err)
        stats += coco_analyze.stats
    stats = [dict(t) for t in set([tuple(s.items()) for s in stats])]

    baseline_perf = {}
    for s in stats:
        if s['err'] == 'baseline':
            baseline_perf[s['oks'],s['areaRngLbl']] = s['auc']

    error_perf = {}
    for eind, err in enumerate(err_types):
        f.write("\n[%s] Errors AUC impact (for OKS evaluation thresholds in %s):\n"%(err,[oksThrs[0],oksThrs[-1]]))
        for aind, areaRngLbl in enumerate(areaRngLbls):
            res = sorted([x for x in stats if x['err']==err and x['areaRngLbl']==areaRngLbl],key=lambda k: k['oks'])
            if err != 'false_neg':
                error_perf[err,areaRngLbl] = [x['auc']-baseline_perf[x['oks'],x['areaRngLbl']] for x in res]
            else:
                bfp_res = sorted([x for x in stats if x['err']=='bckgd_false_pos' and x['areaRngLbl']==areaRngLbl],key=lambda k: k['oks'])
                error_perf[err,areaRngLbl] = [x['auc']-k['auc'] for (x,k) in zip(res,bfp_res)]
            f.write("Area Range [%s]: %s\n"%(areaRngLbl, error_perf[err,areaRngLbl]))

    means = []; stds = []; colors = []; xs = []; xticks = []; x = 0
    for aind, areaRngLbl in enumerate(areaRngLbls):
        x += 1
        for eind, err in enumerate(err_types):
            if eind==3: xticks.append(x)
            xs.append(x); x += 1
            fig, ax = plt.subplots(figsize=(20,10))

            errs = error_perf[err,areaRngLbl]; means.append(np.mean(errs))
            stds.append(np.std(errs)); colors.append(err_colors[eind])

            rects1 = ax.bar(oksThrs, errs, .04, color=err_colors[eind], align='center')
            ax.set_xticks(oksThrs)

            plt.title("Error Type: [%s], Instances Size: [%s]"%(err,areaRngLbl),fontsize=20)
            plt.xlabel('OKS Evaluation Thresh', fontsize=20); plt.ylabel('AUC impact',fontsize=20)
            plt.xlim([0.45,1.]); plt.grid()
            plt.savefig('%s/%s_[%s].pdf'%(loc_dir,err.title(),areaRngLbl), bbox_inches='tight')
            plt.close()

    fig, ax = plt.subplots(figsize=(20,10))
    rects = ax.bar(xs, means, 1, color=colors, yerr=stds)
    lgd = fig.legend(rects[:len(err_types)], err_types, loc="upper center",
                     ncol=len(err_types),fancybox=True, shadow=True,fontsize=18)
    ax.set_xticks(xticks); ax.set_xticklabels(areaRngLbls,fontsize=16)
    plt.ylabel('AUC impact',fontsize=20); plt.xlabel('Instances Size',fontsize=20)
    plt.grid(); plt.savefig('%s/overall.pdf'%loc_dir, bbox_inches='tight')
    if pdf: pdf.savefig()
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

def localizationKeypointBreakdown( coco_analyze, saveDir, pdf=None ):
    loc_dir = saveDir + '/localization_errors/keypoints_breakdown'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Localization Errors Breakdown]\n\n")
    tic = time.time()

    # set parameters for keypoint localization analysis
    coco_analyze.params.areaRng    = [[32 ** 2, 1e5 ** 2]]
    coco_analyze.params.areaRngLbl = ['all']
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []
    coco_analyze.analyze(check_kpts=True, check_scores=False, check_bckgd=False)

    corrected_dts = coco_analyze.corrected_dts['all']
    dt_gt_matches = coco_analyze.localization_matches['all',coco_analyze.params.oksLocThrs,'dts']
    matched_dts   = [cdt for cdt in corrected_dts if 'good' in cdt]
    f.write("Number detections: [%d]\n"%len(corrected_dts))
    f.write("Number matches:    [%d]\n\n"%len(matched_dts))

    good = 0; jitter = 0; inversion = 0; swap = 0; miss = 0; tot = 0.
    good_keypoints = np.zeros(17)
    jitt_keypoints = np.zeros(17); inv_keypoints  = np.zeros(17)
    swap_keypoints = np.zeros(17); miss_keypoints = np.zeros(17)

    for dtm in matched_dts:
        match = dt_gt_matches[dtm['id']][0]
        gtm   = coco_analyze.cocoGt.loadAnns(match['gtId'])[0]

        good      += sum(dtm['good'])
        jitter    += sum(dtm['jitter']); inversion += sum(dtm['inversion'])
        swap      += sum(dtm['swap']);   miss      += sum(dtm['miss'])

        good_keypoints += np.array(dtm['good'])
        jitt_keypoints += np.array(dtm['jitter'])
        inv_keypoints  += np.array(dtm['inversion'])
        swap_keypoints += np.array(dtm['swap'])
        miss_keypoints += np.array(dtm['miss'])

        assert(sum(dtm['good'])+sum(dtm['jitter'])+
               sum(dtm['inversion'])+sum(dtm['swap'])+
               sum(dtm['miss'])==gtm['num_keypoints'])
        tot += gtm['num_keypoints']

    f.write("Total Num. keypoints: [%d]\n"%int(tot))
    f.write("{:30} [{}]-[{}]\n".format(" - Good,      [tot]-[perc]:", int(good), 100*(good/tot)))
    f.write("{:30} [{}]-[{}]\n".format(" - Jitter,    [tot]-[perc]:", int(jitter), 100*(jitter/tot)))
    f.write("{:30} [{}]-[{}]\n".format(" - Inversion, [tot]-[perc]:", int(inversion), 100*(inversion/tot)))
    f.write("{:30} [{}]-[{}]\n".format(" - Swap,      [tot]-[perc]:", int(swap), 100*(swap/tot)))
    f.write("{:30} [{}]-[{}]\n\n".format(" - Miss,      [tot]-[perc]:", int(miss), 100*(miss/tot)))

    # plot the pie charts with number of errors
    COLORS = [ '#1ED88B','#8C4646','#D96459','#F2E394','#F2AE72']
    LABELS = ['Good','Jit.','Inv.','Miss','Swap']
    ERRORS = [(good/tot),(jitter/tot),(inversion/tot),(miss/tot),(swap/tot)]
    TOT_LABELS = []
    for lind, l in enumerate(LABELS):
        label_str = '{:5s}: {:2.1f}'.format(l,ERRORS[lind]*100)
        TOT_LABELS.append(label_str)

    fig = plt.figure(figsize=(5,5))
    rect = 0,0,0.9,0.9
    ax1 = fig.add_axes(rect)
    explode = (0.0,0.0,0.0,0.0,0.0)
    patches, autotexts = ax1.pie( ERRORS, explode=explode, colors=COLORS)
    lgd=fig.legend(patches, TOT_LABELS, loc="upper left",ncol=1,fancybox=True, shadow=True,fontsize=18)
    plt.savefig("%s/overall.pdf"%loc_dir, bbox_inches='tight')
    if pdf: pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(15,15)); plt.axis('off')
    I = io.imread('./manikin.jpg')
    plt.imshow(I); ax = plt.gca(); ax.set_autoscale_on(False)

    rects_d = {}
    rects_d['nose']          = .47,.75,.07,.07
    rects_d['left_eye']      = .5, .83,.07,.07; rects_d['right_eye']      = .44,.83,.07,.07
    rects_d['left_ear']      = .54,.77,.07,.07; rects_d['right_ear']      = .4, .77,.07,.07
    rects_d['left_shoulder'] = .58,.68,.1, .1;  rects_d['right_shoulder'] = .32,.65,.1, .1
    rects_d['left_elbow']    = .67,.6, .1, .1;  rects_d['right_elbow']    = .27,.52,.1, .1
    rects_d['left_wrist']    = .59,.49,.1, .1;  rects_d['right_wrist']    = .34,.42,.1, .1
    rects_d['left_hip']      = .48,.5, .1, .1;  rects_d['right_hip']      = .39,.5, .1, .1
    rects_d['left_knee']     = .55,.32,.1, .1;  rects_d['right_knee']     = .4, .32,.1, .1
    rects_d['left_ankle']    = .55,.15,.1, .1;  rects_d['right_ankle']    = .4, .15,.1, .1
    order = ['nose','left_eye','right_eye','left_ear','right_ear',
             'left_shoulder','right_shoulder','left_elbow','right_elbow',
             'left_wrist','right_wrist','left_hip','right_hip',
             'left_knee','right_knee','left_ankle','right_ankle']
    COLORS = ['#8C4646','#D96459','#F2AE72','#F2E394'][::-1]
    for oi, ok in enumerate(order):
        rect = rects_d[ok]
        ax1 = fig.add_axes(rect)
        explode = (0.0,0.0,0.0,0.0)
        ERRORS = [jitt_keypoints[oi],inv_keypoints[oi],swap_keypoints[oi],miss_keypoints[oi]]
        ERRORS /= sum(ERRORS)
        patches, autotexts = ax1.pie( ERRORS, explode=explode, colors=COLORS)

    lgd=fig.legend(patches, ['Jitter','Inversion','Swap','Miss'][::-1],
        loc="upper center",ncol=len(patches),fancybox=True, shadow=True,fontsize=18)
    plt.savefig("%s/keypoint_breakdown.pdf"%loc_dir, bbox_inches='tight')
    if pdf: pdf.savefig()
    plt.close()

    f.write("Per keypoint breakdown:\n")
    f.write(" - Good:      %s\n"%good_keypoints)
    f.write(" - Jitter:    %s\n"%jitt_keypoints)
    f.write(" - Inversion: %s\n"%inv_keypoints)
    f.write(" - Swap:      %s\n"%swap_keypoints)
    f.write(" - Miss:      %s\n"%miss_keypoints)

    KEYPOINTS_L = ['Nose','Eyes','Ears','Should.','Elbows','Wrists','Hips','Knees','Ankles']
    KEYPOINTS_I = [[0],[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
    COLORS = [c.rgb for c in list(Color("white").range_to(Color("blue"),len(KEYPOINTS_L)))]

    ####################################
    err_vecs = [jitt_keypoints,inv_keypoints,swap_keypoints,miss_keypoints]
    for j, err_type in enumerate(['Jitter', 'Inversion', 'Swap', 'Miss']):
        TOT_LABELS = []
        ERRORS = []
        for i in KEYPOINTS_I:
            tot_errs = 0
            for l in i:
                tot_errs += err_vecs[j][l]
            ERRORS.append(tot_errs/float(sum(err_vecs[j])))

        for lind, l in enumerate(KEYPOINTS_L):
            label_str = '{:7s}: {:2.1f}'.format(l,100*ERRORS[lind])
            TOT_LABELS.append(label_str)

        fig = plt.figure(figsize=(10,5))
        rect = -.03,0,0.45,0.9
        ax1 = fig.add_axes(rect)
        patches, autotexts = ax1.pie( ERRORS, colors=COLORS)
        lgd=fig.legend(patches, TOT_LABELS, bbox_to_anchor=(.37, .9),
            loc="upper left",ncol=2,fancybox=True, shadow=True,fontsize=18)
        plt.title(err_type)
        plt.savefig("%s/%s.pdf"%(loc_dir,err_type), bbox_extra_artists=(lgd,), bbox_inches='tight')
        if pdf: pdf.savefig()
        plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

def localizationOKSImpact( coco_analyze, oks, saveDir, pdf=None ):
    loc_dir = saveDir + '/localization_errors/OKS_impact'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
        os.makedirs(loc_dir+'/all_plots')
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Localization OKS Impact]\n\n")
    tic = time.time()

    coco_analyze.params.areaRng    = [[32**2,1e5**2]]
    coco_analyze.params.areaRngLbl = ['all']
    coco_analyze.params.oksThrs    = [oks]
    coco_analyze.params.err_types  = ['miss','swap', 'inversion', 'jitter']
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    coco_analyze.analyze(check_kpts=True, check_scores=False, check_bckgd=False)
    coco_analyze.summarize(makeplots=True, savedir=loc_dir+'/all_plots')

    corrected_dts = coco_analyze.corrected_dts['all']
    dt_gt_matches = coco_analyze.localization_matches['all',coco_analyze.params.oksLocThrs,'dts']

    for cdt in corrected_dts:
        if cdt['id'] in dt_gt_matches:
            assert(len(dt_gt_matches[cdt['id']])==1)
            dtm = dt_gt_matches[cdt['id']][0]
            assert(cdt['image_id']==dtm['image_id'])
            assert(cdt['score']==dtm['score'])
            assert(cdt['id']==dtm['dtId'])
    matched_dts = [cdt for cdt in corrected_dts if 'good' in cdt]

    OKS_THRS      = oks
    loc_false_pos = []

    for dtm in matched_dts:
        match = dt_gt_matches[dtm['id']][0]
        if match['oks'] < OKS_THRS:
            loc_false_pos.append(dtm)

    f.write("Number detections: [%d]\n"%len(corrected_dts))
    f.write("Number matches:    [%d]\n"%len(matched_dts))
    f.write("Number Loc. FP:    [%d]\n\n"%(len(loc_false_pos)))

    err_corr_dict = {}
    for lfp in loc_false_pos:
        match = dt_gt_matches[lfp['id']][0]
        gtm   = coco_analyze.cocoGt.loadAnns(match['gtId'])[0]

        dt_kpt_arr = np.array(lfp['keypoints'])
        gt_kpt_arr = np.array(gtm['keypoints'])
        ori_oks = compute_oks(dt_kpt_arr, gt_kpt_arr, gtm['area'])

        for eind, err in enumerate(coco_analyze.params.err_types):
            dt_kpt_arr = np.array(lfp['keypoints'])     * (np.repeat(np.array(lfp[err]),3)==0) + \
                         np.array(lfp['opt_keypoints']) * (np.repeat(np.array(lfp[err]),3)==1)
            gt_kpt_arr = np.array(gtm['keypoints'])
            err_corr_oks = compute_oks(dt_kpt_arr, gt_kpt_arr, gtm['area'])

            if sum(lfp[err]) > 0:
                if ('improvement',err) in err_corr_dict:
                    err_corr_dict['improvement',err].append(err_corr_oks - ori_oks)
                else:
                    err_corr_dict['improvement',err] = [err_corr_oks - ori_oks]
                if ('count',err) in err_corr_dict:
                    err_corr_dict['count',err].append(sum(lfp[err]))
                else:
                    err_corr_dict['count',err] = [sum(lfp[err])]

            if err_corr_oks >= OKS_THRS:
                if ('num_fp_corrected',err) in err_corr_dict:
                    err_corr_dict['num_fp_corrected',err] += 1
                else:
                    err_corr_dict['num_fp_corrected',err] = 1

    data_to_plot = []
    f.write("False Positives that would be True Positives without localization error:\n")
    for eind, err in enumerate(coco_analyze.params.err_types):
        data_to_plot.append(err_corr_dict['improvement',err])
        f.write(" - %s:      [%d]\n"%(err,err_corr_dict['num_fp_corrected',err]))

    for eind, err in enumerate(coco_analyze.params.err_types):
        f.write("\nOKS Improvement after correcting [%s] Errors @ OKS Eval Thresh=%s\n"%(err,oks))
        f.write("First Quartile: %f\n"%np.percentile(err_corr_dict['improvement',err],25))
        f.write("Median:         %s\n"%np.percentile(err_corr_dict['improvement',err],50))
        f.write("Third Quartile: %s\n"%np.percentile(err_corr_dict['improvement',err],75))

    fig, ax_boxes = plt.subplots(figsize=(10,10))
    ax_boxes.set_axis_bgcolor('lightgray')
    bp = ax_boxes.boxplot(data_to_plot,0,'',patch_artist=True, showfliers=False,whis=[25, 75],vert=0)
    plt.scatter(1,.05,marker='o', edgecolor='black',color='white', linewidth=2,s=100)

    for ind,box in enumerate(bp['boxes']):
        box.set( color='#7570b3', linewidth=2)
        if ind in [3]: box.set_facecolor('#8C4646')
        if ind in [2]: box.set_facecolor('#D96459')
        if ind in [1]: box.set_facecolor('#F2AE72')
        if ind in [0]: box.set_facecolor('#F2E394')

    for i, median in enumerate(bp['medians']):
        median.set(color='#b2df8a', linewidth=0)
        handle = plt.scatter(median.get_xdata()[0],np.mean(median.get_ydata()),
                   marker='o',edgecolor='black',color='red',
                    alpha=.65,linewidth=0,s=300,zorder=10)

    plt.legend([handle], ['Median'])
    plt.yticks([1,2,3,4],['Miss','Swap','Inversion','Jitter'])
    plt.xlim([0,.6])
    plt.title('1st,3rd Quartiles of OKS improvement @ OKS Eval Thresh=%s'%oks)
    plt.grid()
    plt.savefig('%s/overall.pdf'%loc_dir,bbox_inches='tight')
    if pdf: pdf.savefig()
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

def backgroundCharacteristics( coco_analyze, oks, imgs_info, saveDir ):
    loc_dir = saveDir + '/background_errors/characteristics'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
        os.makedirs(loc_dir+'/all_plots')
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Background Characteristics]\n\n")
    tic = time.time()

    coco_analyze.params.areaRng    = [[32**2,1e5**2]]
    coco_analyze.params.areaRngLbl = ['all']
    coco_analyze.params.oksThrs    = [oks]
    coco_analyze.params.err_types  = []
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    coco_analyze.analyze(check_kpts=False,
                         check_scores=False,
                         check_bkgd=True)
    coco_analyze.summarize(makeplots=True,savedir=loc_dir+'/all_plots')

    corrected_dts  = coco_analyze.corrected_dts
    false_neg_gts  = coco_analyze.false_neg_gts
    coco_gt        = coco_analyze.cocoGt
    dt_gt_matches  = coco_analyze.false_pos_neg_matches

    bkgd_false_pos      = [d for d in corrected_dts if d['false_pos']==True]
    bkgd_false_neg_kpts = [fn for fn in false_neg_gts if fn['num_keypoints']>0]

    f.write("All detections: [%d]\n"%len(corrected_dts))
    f.write(" - Matches:     [%d]\n"%len(dt_gt_matches['dts']))
    f.write(" - Bckgd. FP:   [%d]\n\n"%len(bkgd_false_pos))
    assert(len(corrected_dts)==len(dt_gt_matches['dts'])+len(bkgd_false_pos))

    f.write("All ground-truth: [%d]\n"%len(coco_gt.getAnnIds()))
    f.write(" - Matches:       [%d]\n"%len(dt_gt_matches['gts']))
    f.write(" - Bckgd. FN:     [%d]\n"%len(false_neg_gts))
    f.write("    - >0 kpts:    [%d]\n\n"%len(bkgd_false_neg_kpts))
    assert(len(coco_gt.getAnnIds())==len(dt_gt_matches['gts'])+len(false_neg_gts))

    gt_overlap = []
    fn_overlap = []
    # avg overlap of the ground truths and the missed ground-truths
    for b in coco_gt.loadAnns(coco_gt.getAnnIds()):
        annIds = coco_gt.getAnnIds(imgIds=b['image_id'])
        img_anns = coco_gt.loadAnns(annIds)
        ious = compute_ious(img_anns)
        eye  = np.eye(len(img_anns))

        ind = annIds.index(b['id'])
        overlaps = ious[ind,:][(ious[ind,:]-eye[ind,:])>.1]

        gt_overlap.append(len(overlaps))
        if b in bkgd_false_neg_kpts:
            fn_overlap.append(len(overlaps))

    f.write("False Negative Characteristics:\n")
    f.write("Avg. num. keypoints:                  [%.2f]\n"%(np.mean([b['num_keypoints'] for b in bkgd_false_neg_kpts])))
    f.write("Avg. num. people in images:           [%.2f]\n"%(np.mean([len(coco_gt.getAnnIds(imgIds=b['image_id'])) for b in bkgd_false_neg_kpts])))
    # f.write("Avg. num. of overlaps with IoU>.1:    [%.2f]\n\n"%(np.mean(fn_overlap)))

    gt_counts = dict()
    for i in gt_overlap:
        gt_counts[i] = gt_counts.get(i, 0) + 1

    fn_counts = dict()
    for i in fn_overlap:
        fn_counts[i] = fn_counts.get(i, 0) + 1

    #print "Distribution of annotations with IoU>.1 (normalized by values over all dataset):"
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    x  = [i for i in xrange(max(fn_overlap)+1)]
    y  = [fn_counts[i]/float(gt_counts[i]) for i in x]
    plt.ylabel("Percentage of FN",fontsize=16)
    plt.xlabel("Number of Overlapping Annotations",fontsize=16)
    plt.bar(x,y,align='center',color='g',alpha=.75)
    plt.savefig('%s/fn_overlap.pdf'%loc_dir,bbox_inches='tight')
    plt.close()

    segm_heatmap = np.zeros((128,128))
    for i,b in enumerate(bkgd_false_neg_kpts):
        nx, ny = imgs_info[b['image_id']]['width'],imgs_info[b['image_id']]['height']
        the_mask = np.zeros((ny,nx))

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        for poly_verts in b['segmentation']:
            path = mplPath.Path(np.array([[x,y] for x,y in zip(poly_verts[0::2],poly_verts[1::2])]))

            grid = path.contains_points(points)
            grid = grid.reshape((ny,nx))

            the_mask += np.array(grid, dtype=int)
        segm_heatmap += imresize(the_mask,(128,128))

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    ax.imshow(segm_heatmap)
    plt.savefig('%s/fn_heatmaps.pdf'%loc_dir,bbox_inches='tight')
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

def occlusionAndCrowdingSensitivity( coco_analyze, oks, saveDir ):
    loc_dir = saveDir + '/benchmarks_sensitivity/occlusion_crowding'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Occlusion and Crowding Sensitivity]\n\n")
    tic = time.time()

    coco_analyze.params.oksThrs = [oks]
    coco_analyze.params.err_types = ['miss','swap','inversion','jitter']

    IOU_FOR_OVERLAP = .1
    overlap_groups  = [[0],[1,2],[3,4,5,6,7,8]]
    num_kpt_groups  = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17]]
    coco_gt         = coco_analyze.cocoGt

    overlap_index   = {}
    keypoints_index = {}

    image_ids = coco_gt.getImgIds()
    for img_id in image_ids:
        img_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        img_anns = [k for k in img_anns if 0**2 <= k['area'] < 1e5**2]

        ious = compute_ious(img_anns)
        eye  = np.eye(len(img_anns))

        for aind, a in enumerate(img_anns):
            num_overlaps  = sum((ious[aind,:]-eye[aind,:])>IOU_FOR_OVERLAP)
            num_keypoints = a['num_keypoints']

            overlap_index.setdefault(num_overlaps, []).append(a['id'])
            keypoints_index.setdefault(num_keypoints, []).append(a['id'])

    benchmark_overlap  = {}
    benchmark_keypoint = {}

    for ind, og in enumerate(overlap_groups):
        benchmark_overlap[ind] = {}
        benchmark_overlap[ind]['groups'] = og
        total_gts = []
        for no in og:
            total_gts += overlap_index[no]
        benchmark_overlap[ind]['gtIds'] = total_gts

    for ind, nkg in enumerate(num_kpt_groups):
        benchmark_keypoint[ind] = {}
        benchmark_keypoint[ind]['groups'] = nkg
        total_gts = []
        for nk in nkg:
            total_gts += keypoints_index[nk]
        benchmark_keypoint[ind]['gtIds'] = total_gts

    f.write("Benchmark Dimensions:\n")
    f.write("Overlap @ IoU > .1:\n")
    for k in benchmark_overlap:
        f.write("%d) %s: %d\n"%(k, benchmark_overlap[k]['groups'], len(benchmark_overlap[k]['gtIds'])))

    f.write("\nVisible Keypoints:\n")
    for k in benchmark_keypoint:
        f.write("%d) %s: %d\n"%(k, benchmark_keypoint[k]['groups'], len(benchmark_keypoint[k]['gtIds'])))

    f.write("\nJoint:\n")
    benchmark = {}
    benchmark_mat = np.zeros((len(overlap_groups),len(num_kpt_groups)))
    for i in benchmark_overlap.keys():
        for j in benchmark_keypoint.keys():
            benchmark[i,j] = {}
            benchmark[i,j]['overlaps']  = benchmark_overlap[i]['groups']
            benchmark[i,j]['keypoints'] = benchmark_keypoint[j]['groups']
            benchmark[i,j]['gtIds']     = list(set(benchmark_overlap[i]['gtIds']) & set(benchmark_keypoint[j]['gtIds']))

            benchmark_mat[i,j] =  len(benchmark[i,j]['gtIds'])
            f.write("%d-%d: %d\n"%(i,j,len(benchmark[i,j]['gtIds'])))

    fig = plt.figure(figsize=(6,6))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(benchmark_mat, cmap=plt.cm.Greens, interpolation='nearest')

    width, height = benchmark_mat.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(int(benchmark_mat[x,y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',fontsize=18)

    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.xlabel("Num. keypoints",fontsize=16)
    plt.ylabel("Num. overlapping instances",fontsize=16)
    plt.savefig('%s/benchmark_instances.pdf'%loc_dir,bbox_inches='tight')
    plt.close()

    total_keypoints = np.zeros((3,4))
    jitter_mat      = np.zeros((3,4))
    inversion_mat   = np.zeros((3,4))
    swap_mat        = np.zeros((3,4))
    miss_mat        = np.zeros((3,4))
    good_mat        = np.zeros((3,4))

    ground_truth_num_keypoints = np.zeros((3,4))

    coco_gt_ids = coco_gt.getAnnIds()
    for no, nk in sorted(benchmark.keys(), key=lambda element: (element[0], element[1])):

        benchmark_gt_ids = set(benchmark[no,nk]['gtIds'])
        gt_id_ignores = [gid for gid in coco_gt_ids if gid not in benchmark_gt_ids]
        assert(len(coco_gt_ids)==len(benchmark_gt_ids) + len(gt_id_ignores))

        # set ids to ignore
        coco_analyze.cocoEval.params.useGtIgnore = 1
        coco_analyze.cocoEval.params.gtIgnoreIds = set(gt_id_ignores)

        # run
        coco_analyze.analyze(check_kpts=True, check_scores=True, check_bkgd=True)
        coco_analyze.summarize(makeplots=True, savedir=loc_dir, team_name="%d-%d"%(no,nk))

        for gtId in benchmark[no,nk]['gtIds']:
            ground_truth_num_keypoints[no,nk] += coco_gt.loadAnns(gtId)[0]['num_keypoints']

        # analyze sensitivity of localization errors
        corrected_dts = coco_analyze.corrected_dts
        matches       = coco_analyze.matches

        matched_gts = [matches['gts'][g] for g in matches['gts'] \
                       if int(g) in benchmark[no,nk]['gtIds']]
        matched_dts = set([g[0]['dtId'] for g in matched_gts])
        for i in xrange(len(corrected_dts)):
            if corrected_dts[i]['id'] not in matched_dts:
                continue

            good = np.array(corrected_dts[i]['good'])
            miss = np.array(corrected_dts[i]['miss'])
            jitt = np.array(corrected_dts[i]['jitter'])
            swap = np.array(corrected_dts[i]['swap'])
            inve = np.array(corrected_dts[i]['inversion'])

            total_keypoints[no,nk] += sum(good+miss+jitt+swap+inve)
            jitter_mat[no,nk]      += sum(jitt)
            inversion_mat[no,nk]   += sum(inve)
            swap_mat[no,nk]        += sum(swap)
            miss_mat[no,nk]        += sum(miss)
            good_mat[no,nk]        += sum(good)

    ############## overall num keypoints
    fig = plt.figure(figsize=(6,6))

    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(total_keypoints, cmap=plt.cm.Blues, interpolation='nearest')

    width, height = total_keypoints.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(int(total_keypoints[x,y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',fontsize=18)

    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.title("Total num. keypoints",fontsize=16)
    plt.xlabel("Num. keypoints",fontsize=16)
    plt.ylabel("Num. overlapping instances",fontsize=16)
    plt.savefig('%s/benchmark_keypoints.pdf'%loc_dir,bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(20,16))
    plt.clf()
    ############### jitter
    ax = fig.add_subplot(221)
    ax.set_aspect(1)
    res = ax.imshow(jitter_mat/total_keypoints, cmap=plt.cm.RdPu, interpolation='nearest')

    width, height = jitter_mat.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("%.1f"%(100*jitter_mat[x,y]/float(total_keypoints[x,y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',fontsize=18)

    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.title("Jitter",fontsize=16)
    plt.xlabel("Num. keypoints",fontsize=16)
    plt.ylabel("Num. overlapping instances",fontsize=16)

    ############### inversion
    ax = fig.add_subplot(222)
    ax.set_aspect(1)
    res = ax.imshow(inversion_mat/total_keypoints, cmap=plt.cm.Reds, interpolation='nearest')

    width, height = inversion_mat.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("%.1f"%(100*inversion_mat[x,y]/float(total_keypoints[x,y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',fontsize=18)

    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.title("Inversion",fontsize=16)
    plt.xlabel("Num. keypoints",fontsize=16)
    plt.ylabel("Num. overlapping instances",fontsize=16)

    ############### swap
    ax = fig.add_subplot(223)
    ax.set_aspect(1)
    res = ax.imshow(swap_mat/total_keypoints, cmap=plt.cm.YlOrRd, interpolation='nearest')

    width, height = swap_mat.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("%.1f"%(100*swap_mat[x,y]/float(total_keypoints[x,y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',fontsize=18)

    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.title("Swap",fontsize=16)
    plt.xlabel("Num. keypoints",fontsize=16)
    plt.ylabel("Num. overlapping instances",fontsize=16)

    ############### miss
    ax = fig.add_subplot(224)
    ax.set_aspect(1)
    res = ax.imshow(miss_mat/total_keypoints, cmap=plt.cm.YlOrBr, interpolation='nearest')

    width, height = miss_mat.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate("%.1f"%(100*miss_mat[x,y]/float(total_keypoints[x,y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',fontsize=18)

    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.title("Miss",fontsize=16)
    plt.xlabel("Num. keypoints",fontsize=16)
    plt.ylabel("Num. overlapping instances",fontsize=16)

    plt.savefig('%s/localization_errors_sensitivity.pdf'%loc_dir,bbox_inches='tight')
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

def sizeSensitivity( coco_analyze, oks, saveDir ):
    loc_dir = saveDir + '/benchmarks_sensitivity/size'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
        os.makedirs(loc_dir + '/all_plots')
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Size Sensitivity]\n\n")
    tic = time.time()

    areaRngs    = [[32 ** 2, 64 ** 2],[64 ** 2, 96 ** 2],
                   [96 ** 2, 128 ** 2],[128 ** 2, 1e5 ** 2],[32 ** 2, 1e5 ** 2]]
    areaRngLbls = ['medium','large','xlarge','xxlarge','all']
    error_types = ['miss','swap','inversion','jitter','score','bckgd']

    coco_analyze.params.oksThrs   = [oks]
    coco_analyze.params.err_types = []
    coco_gt = coco_analyze.cocoGt
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    size_index = {}
    anns = coco_gt.loadAnns(coco_gt.getAnnIds())
    for a in anns:
        if areaRngs[0][0] < a['area'] <= areaRngs[0][1]:
            size_index.setdefault('medium', []).append(a['id'])

        if areaRngs[1][0] < a['area'] <= areaRngs[1][1]:
            size_index.setdefault('large', []).append(a['id'])

        if areaRngs[2][0] < a['area'] <= areaRngs[2][1]:
            size_index.setdefault('xlarge', []).append(a['id'])

        if areaRngs[3][0] < a['area'] <= areaRngs[3][1]:
            size_index.setdefault('xxlarge', []).append(a['id'])

    f.write("Benchmark Dimensions:\n")
    for i,a in enumerate(areaRngs[:-1]):
        f.write("%d) %s-%s: %d\n"%(i,areaRngLbls[i],a,len(size_index[areaRngLbls[i]])))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_axis_bgcolor('lightgray')
    x = [1,2,3,4]
    y = [len(size_index['medium']), len(size_index['large']), len(size_index['xlarge']), len(size_index['xxlarge'])]
    plt.bar(x,y,color='g',align='center')
    plt.xticks(x,['med','lrg','xlrg','xxlrg'])
    plt.title('Instances Size Distribution',fontsize=16)
    plt.grid()
    plt.savefig('%s/benchmarks.pdf'%loc_dir,bbox_inches='tight')
    plt.close()

    auc_total_res = []
    err_total_res = []
    for aind, areaLbl in enumerate(areaRngLbls):
        # set the parameters then evaluate
        coco_analyze.params.err_types  = []
        coco_analyze.params.areaRng    = [areaRngs[aind]]
        coco_analyze.params.areaRngLbl = [areaLbl]

        coco_analyze.analyze(check_kpts=False, check_scores=False, check_bkgd=False)
        coco_analyze.summarize(makeplots=True,savedir=loc_dir + '/all_plots',team_name='baseline')
        auc_total_res.extend(coco_analyze.stats)

        for eind, err in enumerate(error_types):
            if err in ['miss','swap','inversion','jitter']:
                coco_analyze.params.err_types  = [err]
                coco_analyze.analyze(check_kpts=True, check_scores=False, check_bkgd=False)
                coco_analyze.summarize(makeplots=True,savedir=loc_dir + '/all_plots',team_name=err)

            if err == 'score':
                coco_analyze.params.err_types  = []
                coco_analyze.analyze(check_kpts=False, check_scores=True, check_bkgd=False)
                coco_analyze.summarize(makeplots=True,savedir=loc_dir + '/all_plots',team_name='scoring')

            if err == 'bckgd':
                coco_analyze.params.err_types  = []
                coco_analyze.analyze(check_kpts=False, check_scores=False, check_bkgd=True)
                coco_analyze.summarize(makeplots=True,savedir=loc_dir + '/all_plots',team_name='background')
            err_total_res.extend(coco_analyze.stats)

    oks_75_auc = []
    for area in areaRngLbls:
        oks_75_auc.append([r['auc'] for r in auc_total_res if r['areaRng']==area][0])

    oks_75_err = {}
    baselines  = []
    results    = []
    for r in err_total_res:
        if 'err' in r:
            results.append(r)
        else:
            baselines.append(r)

    for err in ['miss','swap','inversion','jitter','score','bkg_false_pos','false_neg']:
        res = [r for r in results if r['err']==err]
        oks_auc = []
        for area in areaRngLbls:
            oks_auc.append([r['auc'] for r in res if r['areaRng']==area][0])
        oks_75_err[err] = oks_auc

    perf_miss  = list(np.array(oks_75_err['miss'])-np.array(oks_75_auc))
    perf_swap  = list(np.array(oks_75_err['swap'])-np.array(oks_75_auc))
    perf_inv   = list(np.array(oks_75_err['inversion'])-np.array(oks_75_auc))
    perf_jitt  = list(np.array(oks_75_err['jitter'])-np.array(oks_75_auc))
    perf_score = list(np.array(oks_75_err['score'])-np.array(oks_75_auc))
    perf_bk_fp = list(np.array(oks_75_err['bkg_false_pos'])-np.array(oks_75_auc))
    perf_bk_fn = list(np.array(oks_75_err['false_neg'])-np.array(oks_75_err['bkg_false_pos']))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_axis_bgcolor('lightgray')
    plt.ylabel("AUC Impact",fontsize=20)
    plt.title("Performance and Error Sensitivity over size @ OKS Eval Thresh=%.2f"%oks)

    x = [-2,-1.5,-1,-.5, .5,1,1.5,2,  3,3.5,4,4.5,  5.5,6,6.5,7,  8,8.5,9,9.5,
         10.5,11,11.5,12,  13,13.5,14,14.5,  15.5,16,16.5,17]

    y = oks_75_auc[:4] + perf_jitt[:4] + perf_inv[:4] + perf_swap[:4] + \
        perf_miss[:4]  + perf_score[:4] + perf_bk_fp[:4] + perf_bk_fn[:4]

    plt.scatter(x,y,c='b',s=150,alpha=.5,edgecolor='black',linewidth=2)

    plt.plot([-2,-.5],  [oks_75_auc[4], oks_75_auc[4]],'r--',linewidth=2)
    plt.plot([.5, 2],   [perf_jitt[4],  perf_jitt[4]],'r--',linewidth=2)
    plt.plot([3, 4.5],  [perf_inv[4],   perf_inv[4]],'r--',linewidth=2)
    plt.plot([5.5, 7],  [perf_swap[4],  perf_swap[4]],'r--',linewidth=2)
    plt.plot([8, 9.5],  [perf_miss[4],  perf_miss[4]],'r--',linewidth=2)
    plt.plot([10.5, 12],[perf_score[4], perf_score[4]],'r--',linewidth=2)
    plt.plot([13, 14.5],[perf_bk_fp[4], perf_bk_fp[4]],'r--',linewidth=2)
    plt.plot([15.5, 17],[perf_bk_fn[4], perf_bk_fn[4]],'r--',linewidth=2)

    yy = -.05
    ax.annotate('OKS %s'%oks, xy=(-1.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    ax.annotate('Jitter', xy=(1.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    ax.annotate('Inversion', xy=(3.75,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    ax.annotate('Swap', xy=(6.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    ax.annotate('Miss', xy=(8.75,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    ax.annotate('Score', xy=(11.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    ax.annotate('Bkgd. FP', xy=(13.75,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    ax.annotate('FN', xy=(16.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=16)

    plt.xticks(x,['m','l','xl','xxl','m','l','xl','xxl','m','l','xl','xxl',
                  'm','l','xl','xxl','m','l','xl','xxl','m','l','xl','xxl',
                  'm','l','xl','xxl','m','l','xl','xxl'])
    plt.xlim([-2.5,17.5])
    plt.grid()
    plt.savefig('%s/sensitivity.pdf'%loc_dir,bbox_inches='tight')
    plt.close()

    f.write("\nOKS %.2f:  Sens[%.3f], Impact[%.3f]\n"%(oks, max(oks_75_auc[:4])-min(oks_75_auc[:4]), max(oks_75_auc[:4])-oks_75_auc[4]))
    f.write("Jitter:    Sens[%.3f], Impact[%.3f]\n"%(max(perf_jitt[:4])-min(perf_jitt[:4]),max(perf_jitt[:4])-perf_jitt[4]))
    f.write("Inversion: Sens[%.3f], Impact[%.3f]\n"%(max(perf_inv[:4]) -min(perf_inv[:4]) ,max(perf_inv[:4])-perf_inv[4]))
    f.write("Swap:      Sens[%.3f], Impact[%.3f]\n"%(max(perf_swap[:4])-min(perf_swap[:4]),max(perf_swap[:4])-perf_swap[4]))
    f.write("Miss:      Sens[%.3f], Impact[%.3f]\n"%(max(perf_miss[:4])-min(perf_miss[:4]),max(perf_miss[:4])-perf_miss[4]))
    f.write("Score:     Sens[%.3f], Impact[%.3f]\n"%(max(perf_score[:4])-min(perf_score[:4]),max(perf_score[:4])-perf_score[4]))
    f.write("Bkgd FP:   Sens[%.3f], Impact[%.3f]\n"%(max(perf_bk_fp[:4])-min(perf_bk_fp[:4]),max(perf_bk_fp[:4])-perf_bk_fp[4]))
    f.write("FN:        Sens[%.3f], Impact[%.3f]\n"%(max(perf_bk_fn[:4])-min(perf_bk_fn[:4]),max(perf_bk_fn[:4])-perf_bk_fn[4]))

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

"""
Utility functions
"""
def compute_oks(dt_kpt_arr, gt_kpt_arr, area):
    xd = dt_kpt_arr[0::3]; yd = dt_kpt_arr[1::3]; vd = dt_kpt_arr[2::3]
    xg = gt_kpt_arr[0::3]; yg = gt_kpt_arr[1::3]; vg = gt_kpt_arr[2::3]

    sigmas   = np.array([.026,.025,.025,.035,.035,.079,.079,.072,.072,
                         .062,.062,.107,.107,.087,.087,.089,.089])

    dx = xd - xg
    dy = yd - yg

    e = (dx**2 + dy**2) / (sigmas * 2)**2 / (area+np.spacing(1)) / 2
    e = e[vg > 0]

    return np.sum(np.exp(-e)) / e.shape[0]

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

    for i in xrange(num_boxes):
        for j in xrange(i,num_boxes):
            ious[i,j] = compute_iou(anns[i]['bbox'],anns[j]['bbox'])
            if i!=j:
                ious[j,i] = ious[i,j]
    return ious
