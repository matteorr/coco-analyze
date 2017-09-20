## imports
import os, time
import numpy as np
from colour import Color
import matplotlib.pyplot as plt
import skimage.io as io
import utilities

def localizationErrors( coco_analyze, imgs_info, saveDir ):
    loc_dir = saveDir + '/localization_errors/keypoints_breakdown'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Localization Errors Breakdown]\n\n")
    tic = time.time()
    paths = {}

    # set parameters for keypoint localization analysis
    coco_analyze.params.areaRng    = [[32 ** 2, 1e5 ** 2]]
    coco_analyze.params.areaRngLbl = ['all']
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []
    coco_analyze.analyze(check_kpts=True, check_scores=False, check_bckgd=False)

    corrected_dts = coco_analyze.corrected_dts['all']
    dt_gt_matches = coco_analyze.localization_matches['all',str(coco_analyze.params.oksLocThrs),'dts']
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
    lgd=fig.legend(patches, TOT_LABELS, loc="upper left",ncol=1,fancybox=True, shadow=True,fontsize=20)
    paths['overall_kpts_errors'] = "%s/overall_keypoint_errors.pdf"%loc_dir
    plt.savefig(paths['overall_kpts_errors'], bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(15,15)); plt.axis('off')
    I = io.imread('./latex/manikin.jpg')
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
    COLORS = ['#8C4646','#D96459','#F2AE72','#F2E394']

    f.write("Per Keypoint breakdown: [jitter, inversion, swap, miss]\n")
    for oi, ok in enumerate(order):
        rect = rects_d[ok]
        ax1 = fig.add_axes(rect)
        explode = (0.0,0.0,0.0,0.0)
        ERRORS = [jitt_keypoints[oi],inv_keypoints[oi],swap_keypoints[oi],miss_keypoints[oi]]
        ERRORS /= sum(ERRORS)
        f.write(" - %s: %s\n"%(ok,ERRORS))
        patches, autotexts = ax1.pie( ERRORS, explode=explode, colors=COLORS[::-1])

    lgd=fig.legend(patches, ['Jitter','Inversion','Swap','Miss'][::-1],
        loc="upper center",ncol=len(patches),fancybox=True, shadow=True,fontsize=20)
    paths['kpt_errors_breakdown'] = "%s/keypoint_breakdown.pdf"%loc_dir
    plt.savefig(paths['kpt_errors_breakdown'], bbox_inches='tight')
    plt.close()

    f.write("\nPer Error breakdown: %s\n"%order)
    f.write(" - Good:      %s\n"%good_keypoints)
    f.write(" - Jitter:    %s\n"%jitt_keypoints)
    f.write(" - Inversion: %s\n"%inv_keypoints)
    f.write(" - Swap:      %s\n"%swap_keypoints)
    f.write(" - Miss:      %s\n"%miss_keypoints)

    KEYPOINTS_L = ['Nose','Eyes','Ears','Should.','Elbows','Wrists','Hips','Knees','Ankles']
    KEYPOINTS_I = [[0],[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]

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
        colors = [c.rgb for c in list(Color("white").range_to(Color(COLORS[j]),len(KEYPOINTS_L)))]
        patches, autotexts = ax1.pie( ERRORS, colors=colors)
        lgd=fig.legend(patches, TOT_LABELS, bbox_to_anchor=(.45, .9),
            loc="upper left",ncol=2,fancybox=True, shadow=True,fontsize=20)
        plt.title(err_type,fontsize=20)
        path = '%s_kpt_breakdown'%err_type
        paths[path] = "%s/%s.pdf"%(loc_dir,path)
        plt.savefig(paths[path], bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

    for err in ['miss','swap','inversion','jitter']:
        f.write("\nTop errors of type [%s]:\n"%(err))
        err_dts = [d for d in coco_analyze.corrected_dts['all'] if err in d]
        top_err_dts = sorted(err_dts, key=lambda k: -k['score'])
        top_err_dts = sorted(top_err_dts, key=lambda k: -sum(k[err]))

        for tind, t in enumerate(top_err_dts[0:7]):
            I = io.imread(imgs_info[t['image_id']]['coco_url'])
            plt.figure(figsize=(10,10)); plt.axis('off')
            plt.imshow(I)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            bbox = t['bbox']
            rect = plt.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],fill=False,edgecolor=[1, .6, 0],linewidth=3)
            ax.add_patch(rect)
            sks = np.array(utilities.skeleton)-1
            kp = np.array(t['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            for sk in sks:
                plt.plot(x[sk],y[sk], linewidth=3, color=utilities.colors[sk[0],sk[1]])

            for kk in xrange(17):
                if kk in [1,3,5,7,9,11,13,15]:
                    plt.plot(x[kk], y[kk],'o',markersize=5, markerfacecolor='r',
                                                  markeredgecolor='r', markeredgewidth=3)
                elif kk in [2,4,6,8,10,12,14,16]:
                    plt.plot(x[kk], y[kk],'o',markersize=5, markerfacecolor='g',
                                                  markeredgecolor='g', markeredgewidth=3)
                else:
                    plt.plot(x[kk], y[kk],'o',markersize=5, markerfacecolor='b',
                                                  markeredgecolor='b', markeredgewidth=3)

            title = "[%d][%d][%.3f][%d]"%(t['image_id'],t['id'],t['score'],sum(t[err]))
            f.write("%s\n"%title)
            plt.title(title,fontsize=20)
            path = '%s_%d'%(err,tind)
            paths[path] = "%s/%s.pdf"%(loc_dir,path)
            plt.savefig(paths[path], bbox_inches='tight',dpi=50)
            plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

    return paths
