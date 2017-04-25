## imports
import os
import time
import json
import numpy as np
from colour import Color
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.misc import imread, imresize

## constants
def backgroundCharacteristics( coco_analyze, oks, imgs_info ):

    coco_analyze.params.areaRng    = [[32**2,1e5**2]]
    coco_analyze.params.areaRngLbl = ['all']
    coco_analyze.params.oksThrs    = [oks]
    coco_analyze.params.err_types  = []
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    coco_analyze.analyze(check_kpts=False,
                         check_scores=False,
                         check_bkgd=True)
    coco_analyze.summarize(makeplots=True)

    corrected_dts  = coco_analyze.corrected_dts
    false_neg_gts  = coco_analyze.false_neg_gts
    coco_gt        = coco_analyze.cocoGt
    dt_gt_matches  = coco_analyze.false_pos_neg_matches

    bkgd_false_pos      = [d for d in corrected_dts if d['false_pos']==True]
    bkgd_false_neg_kpts = [fn for fn in false_neg_gts if fn['num_keypoints']>0]

    print "All detections: [%d]"%len(corrected_dts)
    print " - Matches:     [%d]"%len(dt_gt_matches['dts'])
    print " - Bckgd. FP:   [%d]\n"%len(bkgd_false_pos)
    assert(len(corrected_dts)==len(dt_gt_matches['dts'])+len(bkgd_false_pos))

    print "All ground-truth: [%d]"%len(coco_gt.getAnnIds())
    print " - Matches:       [%d]"%len(dt_gt_matches['gts'])
    print " - Bckgd. FN:     [%d]"%len(false_neg_gts)
    print "    - >0 kpts:    [%d]"%len(bkgd_false_neg_kpts)
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

    print "False Negative Characteristics:"
    print "Avg. num. keypoints:                  [%.2f]"%(np.mean([b['num_keypoints'] for b in bkgd_false_neg_kpts]))
    print "Avg. num. people in images:           [%.2f]"%(np.mean([len(coco_gt.getAnnIds(imgIds=b['image_id'])) for b in bkgd_false_neg_kpts]))
    print "Avg. num. of annotations with IoU>.1: [%.2f]"%(np.mean(fn_overlap))

    gt_counts = dict()
    for i in gt_overlap:
        gt_counts[i] = gt_counts.get(i, 0) + 1

    fn_counts = dict()
    for i in fn_overlap:
        fn_counts[i] = fn_counts.get(i, 0) + 1

    print "Distribution of annotations with IoU>.1 (normalized by values over all dataset):"
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    x  = [i for i in xrange(max(fn_overlap)+1)]
    y  = [fn_counts[i]/float(gt_counts[i]) for i in x]
    plt.bar(x,y,align='center',color='g',alpha=.75)
    plt.show()

    segm_heatmap = np.zeros((256,256))
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
        segm_heatmap += imresize(the_mask,(256,256))

    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111)
    ax.imshow(segm_heatmap)
    plt.show()

## methods
def backgroundAUCImpact( coco_analyze ):

    areaRngs    = [[32**2,1e5**2],[32**2,96**2],[96**2,1e5**2],[0**2,1e5**2]]
    areaRngLbls = ['all','medium','large','with_small']
    oksVals     = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    total_res = {}
    for areaRng,areaRngLbl in zip(areaRngs,areaRngLbls):
        coco_analyze.params.areaRng    = [areaRng]
        coco_analyze.params.areaRngLbl = [areaRngLbl]

        for oks in oksVals:
            coco_analyze.params.oksThrs = [oks]

            coco_analyze.params.err_types = []
            coco_analyze.analyze(check_kpts=False,
                                 check_scores=False,
                                 check_bkgd=True)
            coco_analyze.summarize(makeplots=True)
            total_res[int(oks*100),areaRngLbl] = [s['auc'] for s in coco_analyze.stats]

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in xrange(10):
        okst = int(100*oksVals[i])

        fp_impact = [total_res[okst,area][1] - total_res[okst,area][0] for area in areaRngLbls]
        fn_impact = [total_res[okst,area][2] - total_res[okst,area][1] for area in areaRngLbls]

        ax.bar(i,np.mean(fp_impact),color='purple',width=.45,yerr=np.std(fp_impact))
        ax.bar(i+.45,np.mean(fn_impact),color='seagreen',width=.45,yerr=np.std(fn_impact))

    plt.xticks([i+.45 for i in xrange(10)],[str(s) for s in oksVals])
    plt.title('AUC impact of Bckgd. FP and FN',fontsize=18)
    ax.grid()
    plt.show()

def sizeSensitivity( coco_analyze, oks ):

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

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_axis_bgcolor('lightgray')
    x = [1,2,3,4]
    y = [len(size_index['medium']), len(size_index['large']), len(size_index['xlarge']), len(size_index['xxlarge'])]
    plt.bar(x,y,color='g',align='center')
    plt.xticks(x,['med','lrg','xlrg','xxlrg'])
    plt.title('Instance Size Distribution',fontsize=16)
    plt.grid()
    plt.show()

    auc_total_res = []
    err_total_res = []
    for aind, areaLbl in enumerate(areaRngLbls):
        # set the parameters then evaluate
        coco_analyze.params.err_types  = []
        coco_analyze.params.areaRng    = [areaRngs[aind]]
        coco_analyze.params.areaRngLbl = [areaLbl]

        coco_analyze.analyze(check_kpts=False, check_scores=False, check_bkgd=False)
        coco_analyze.summarize(makeplots=True)
        auc_total_res.extend(coco_analyze.stats)

        for eind, err in enumerate(error_types):
            if err in ['miss','swap','inversion','jitter']:
                coco_analyze.params.err_types  = [err]
                coco_analyze.analyze(check_kpts=True, check_scores=False, check_bkgd=False)
                coco_analyze.summarize(makeplots=True)

            if err == 'score':
                coco_analyze.params.err_types  = []
                coco_analyze.analyze(check_kpts=False, check_scores=True, check_bkgd=False)
                coco_analyze.summarize(makeplots=True)

            if err == 'bckgd':
                coco_analyze.params.err_types  = []
                coco_analyze.analyze(check_kpts=False, check_scores=False, check_bkgd=True)
                coco_analyze.summarize(makeplots=True)
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
    plt.show()

    print "OKS %.2f:  Sens[%.3f], Impact[%.3f]"%(oks, max(oks_75_auc[:4])-oks_75_auc[4], max(oks_75_auc[:4])-min(oks_75_auc[:4]))
    print "Jitter:    Sens[%.3f], Impact[%.3f]"%(max(perf_jitt[:4])-perf_jitt[4], max(perf_jitt[:4])-min(perf_jitt[:4]))
    print "Inversion: Sens[%.3f], Impact[%.3f]"%(max(perf_inv[:4])-perf_inv[4], max(perf_inv[:4])-min(perf_inv[:4]))
    print "Swap:      Sens[%.3f], Impact[%.3f]"%(max(perf_swap[:4])-perf_swap[4], max(perf_swap[:4])-min(perf_swap[:4]))
    print "Miss:      Sens[%.3f], Impact[%.3f]"%(max(perf_miss[:4])-perf_miss[4], max(perf_miss[:4])-min(perf_miss[:4]))
    print "Score:     Sens[%.3f], Impact[%.3f]"%(max(perf_score[:4])-perf_score[4], max(perf_score[:4])-min(perf_score[:4]))
    print "Bkgd FP:   Sens[%.3f], Impact[%.3f]"%(max(perf_bk_fp[:4])-perf_bk_fp[4], max(perf_bk_fp[:4])-min(perf_bk_fp[:4]))
    print "FN:        Sens[%.3f], Impact[%.3f]"%(max(perf_bk_fn[:4])-perf_bk_fn[4], max(perf_bk_fn[:4])-min(perf_bk_fn[:4]))

def occlusionAndCrowdingSensitivity( coco_analyze, oks ):

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

    print "Overlap:"
    for k in benchmark_overlap:
        print k, benchmark_overlap[k]['groups'], len(benchmark_overlap[k]['gtIds'])

    print "Keypoints:"
    for k in benchmark_keypoint:
        print k, benchmark_keypoint[k]['groups'], len(benchmark_keypoint[k]['gtIds'])

    benchmark = {}
    benchmark_mat = np.zeros((len(overlap_groups),len(num_kpt_groups)))
    print "Joint:"
    for i in benchmark_overlap.keys():
        for j in benchmark_keypoint.keys():
            benchmark[i,j] = {}
            benchmark[i,j]['overlaps']  = benchmark_overlap[i]['groups']
            benchmark[i,j]['keypoints'] = benchmark_keypoint[j]['groups']
            benchmark[i,j]['gtIds']     = list(set(benchmark_overlap[i]['gtIds']) & set(benchmark_keypoint[j]['gtIds']))

            benchmark_mat[i,j] =  len(benchmark[i,j]['gtIds'])

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
    plt.show()

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
        coco_analyze.summarize(makeplots=True)

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
    plt.show()

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
    plt.show()

# def scoringOKSImpact( coco_analyze ):
#     coco_analyze.params.oksThrs       = [.5]
#     coco_analyze.params.areaRng       = [[32 ** 2, 1e5 ** 2]]
#     coco_analyze.params.areaRngLbl    = ['all']
#
#     coco_analyze.analyze( check_kpts=False, check_scores=True, check_bkgd=False)
#     coco_analyze.summarize(makeplots=True)
#
#     images_list = list(set([i['image_id'] for i in coco_analyze.corrected_dts]))
#     print "Number of images with detections: %d"%len(images_list)
#
#     coco_gt = coco_analyze.cocoGt
#
#     num_seq_inver = []
#     num_seq_swaps = []
#
#     glob_opt_score_fn = []
#     glob_ori_score_fn = []
#
#     glob_opt_score_oks = []
#     glob_ori_score_oks = []
#
#     image_ids_improv_oks = []
#
#     for image_id in images_list:
#         image_gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image_id))
#
#         image_dts = [d for d in coco_analyze.corrected_dts if d['image_id']==image_id]
#
#         opt_score = [d['id'] for d in sorted(image_dts, key=lambda k: -k['opt_score'])]
#         ori_score = [d['id'] for d in sorted(image_dts, key=lambda k: -k['score'])]
#
#         perm = get_permutation(opt_score, ori_score)
#         num_inv = sort_and_count(perm)[0]
#         num_seq_inver.append(num_inv)
#         num_seq_swaps.append(number_of_seq_swaps(perm))
#
#         if num_inv==0:
#             continue
#
#         opt_score_fn = 0
#         ori_score_fn = 0
#
#         opt_score_oks = 0.
#         ori_score_oks = 0.
#
#         for g in image_gts:
#             # detection should be ignored
#             if g['_ignore']==1:
#                 continue
#
#             ################################
#             # statistics with previous score
#             if g['id'] not in coco_analyze.score_matches['gts']:
#                 # ground truth has no match
#                 ori_score_fn += 1
#             else:
#                 # detection has a match
#                 assert(len(coco_analyze.score_matches['gts'][g['id']]) == 1)
#                 assert(coco_analyze.score_matches['gts'][g['id']][0]['ignore'] == 0)
#                 ori_score_oks += coco_analyze.score_matches['gts'][g['id']][0]['oks']
#
#             ################################
#             # statistics with new score
#             if g['id'] not in coco_analyze.opt_score_matches['gts']:
#                 # ground truth has no match
#                 opt_score_fn += 1
#             else:
#                 # detection has a match
#                 assert(len(coco_analyze.opt_score_matches['gts'][g['id']]) == 1)
#                 assert(coco_analyze.opt_score_matches['gts'][g['id']][0]['ignore'] == 0)
#                 opt_score_oks += coco_analyze.opt_score_matches['gts'][g['id']][0]['oks']
#
#         glob_opt_score_fn.append(opt_score_fn)
#         glob_ori_score_fn.append(ori_score_fn)
#         glob_opt_score_oks.append(opt_score_oks)
#         glob_ori_score_oks.append(ori_score_oks)
#
#         image_ids_improv_oks.append(image_id)
#
#     img_ids_interest = np.array(image_ids_improv_oks)[np.where(np.array(glob_opt_score_oks) != np.array(glob_ori_score_oks))[0]]
#     gt_ids = coco_gt.getAnnIds(imgIds=list(img_ids_interest))
#
#     print "Number of images with sub-optimal scoring order: %d"%len(image_ids_improv_oks)
#     print "Number of images with OKS improvement after rescoring: %d"%len(np.where(np.array(glob_opt_score_oks) != np.array(glob_ori_score_oks))[0])
#
#     opt_oks_gt = []
#     ori_oks_gt = []
#     for gt in gt_ids:
#
#         if gt in coco_analyze.score_matches['gts']:
#             ori_oks_gt.append(coco_analyze.score_matches['gts'][gt][0]['oks'])
#         else:
#             ori_oks_gt.append(0)
#
#         if gt in coco_analyze.opt_score_matches['gts']:
#             opt_oks_gt.append(coco_analyze.opt_score_matches['gts'][gt][0]['oks'])
#         else:
#             opt_oks_gt.append(0)
#
#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.set_axis_bgcolor('lightgray')
#
#     plt.scatter(ori_oks_gt, opt_oks_gt,marker='o',edgecolor='white',alpha=.75,color='blue',linewidth=1,s=75)
#     # plt.title('CMU - gt oks improvement')
#     plt.xlim([.5,1])
#     plt.xticks([.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1],[])
#     plt.yticks([.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1],[])
#     plt.ylim([.5,1])
#     plt.grid()
#     plt.show()
#
# print sum(np.array(cmu_ori_oks_gt)==0), sum(np.array(grmi_ori_oks_gt)==0)
#
# cmu_both_zero_oks_count = 0
# cmu_not_zero_oks_count = 0
# cmu_not_zero_oks_val   = 0.
# for ori,opt in zip(cmu_ori_oks_gt, cmu_opt_oks_gt):
#     if ori == 0:
#         if opt != 0:
#             cmu_not_zero_oks_count += 1
#             cmu_not_zero_oks_val   += opt
#         else:
#             cmu_both_zero_oks_count += 1
#
# grmi_both_zero_oks_count = 0
# grmi_not_zero_oks_count = 0
# grmi_not_zero_oks_val   = 0.
# for ori,opt in zip(grmi_ori_oks_gt, grmi_opt_oks_gt):
#     if ori == 0:
#         if opt != 0:
#             grmi_not_zero_oks_count += 1
#             grmi_not_zero_oks_val   += opt
#         else:
#             grmi_both_zero_oks_count += 1
#
# print cmu_both_zero_oks_count, grmi_both_zero_oks_count
# print cmu_not_zero_oks_count, grmi_not_zero_oks_count
# print cmu_not_zero_oks_val, grmi_not_zero_oks_val
#
# fig, ax = plt.subplots(figsize=(1,10))
# ax.set_axis_bgcolor('lightgray')
# plt.scatter(grmi_ori_oks_gt, grmi_opt_oks_gt,marker='o',edgecolor='white',alpha=.75,color='red',linewidth=1,s=75)
# plt.xticks([0],[])
# plt.yticks([.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1],[])
# plt.xlim([-.01,+.01])
# plt.ylim([.5,1])
# plt.grid()
# plt.show()

def scoringAUCImpact( coco_analyze ):
    areaRngs    = [[32**2,1e5**2],[32**2,96**2],[96**2,1e5**2]]
    areaRngLbls = ['all','medium','large']
    oksVals     = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    total_res = []

    for areaRng,areaRngLbl in zip(areaRngs,areaRngLbls):
        coco_analyze.params.areaRng    = [areaRng]
        coco_analyze.params.areaRngLbl = [areaRngLbl]

        for oks in oksVals:
            coco_analyze.params.oksThrs = [oks]
            coco_analyze.analyze(check_kpts=False,
                                 check_scores=True,
                                 check_bkgd=False)
            coco_analyze.summarize(makeplots=True)
            total_res += coco_analyze.stats

    baselines = {}
    for b in total_res:
        if 'err' not in b:
            baselines[b['oks'],b['areaRng']] = b['auc']

    total_res = sorted(total_res, key=lambda k: k['oks'])

    ## get scoring improvements
    score_imp_med = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                     if 'err' in x and x['err']=='score' and x['areaRng']=='medium']
    score_imp_lrg = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                     if 'err' in x and x['err']=='score' and x['areaRng']=='large']
    score_imp_all = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                     if 'err' in x and x['err']=='score' and x['areaRng']=='all']

    fig, ax = plt.subplots(figsize=(20,10))

    x  = [i+.2 for i in xrange(10)]
    y  = [i for i in score_imp_med]

    rects1 = ax.bar(x, y, color='blue',alpha=.65,align='center',width=.4)
    plt.title(' medium - score AUC impact',fontsize=16)
    plt.xlim([-0.2,10])
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(20,10))

    x  = [i+.2 for i in xrange(10)]
    y  = [i for i in score_imp_med]

    rects1 = ax.bar(x, y, color='blue',alpha=.65,align='center',width=.4)
    plt.title(' large - score AUC impact',fontsize=16)
    plt.xlim([-0.2,10])
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(20,10))

    x  = [i+.2 for i in xrange(10)]
    y  = [i for i in score_imp_med]

    rects1 = ax.bar(x, y, color='blue',alpha=.65,align='center',width=.4)
    plt.title(' all - score AUC impact',fontsize=16)
    plt.xlim([-0.2,10])
    plt.grid()
    plt.show()

def localizationOKSImpact( coco_analyze, oks ):

    coco_analyze.params.areaRng    = [[32**2,1e5**2]]
    coco_analyze.params.areaRngLbl = ['all']
    coco_analyze.params.oksThrs    = [oks]
    coco_analyze.params.err_types  = ['miss','swap', 'inversion', 'jitter']
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    coco_analyze.analyze(check_kpts=True, check_scores=False, check_bkgd=False)
    coco_analyze.summarize(makeplots=True)

    corrected_dts = coco_analyze.corrected_dts
    dt_gt_matches = coco_analyze.matches
    coco_gt       = coco_analyze.cocoGt

    for cdt in corrected_dts:
        if cdt['id'] in dt_gt_matches['dts']:
            assert(len(dt_gt_matches['dts'][cdt['id']])==1)
            dtm = dt_gt_matches['dts'][cdt['id']][0]
            assert(cdt['image_id']==dtm['image_id'])
            assert(cdt['score']==dtm['score'])
            assert(cdt['id']==dtm['dtId'])

    matched_dts = [cdt for cdt in corrected_dts if 'good' in cdt]

    sigmas   = np.array([.026,.025,.025,.035,.035,.079,.079,.072,.072,.062,.062,.107,.107,.087,.087,.089,.089])

    OKS_THRS      = oks
    loc_false_pos = []

    for dtm in matched_dts:
        match = dt_gt_matches['dts'][dtm['id']][0]

        if match['oks'] < OKS_THRS:
            loc_false_pos.append(dtm)

    print " - Loc. False Positives: [%d]/[%d]"%(len(loc_false_pos),len(matched_dts))

    jitt_impr = []
    inv_impr  = []
    swap_impr = []
    miss_impr = []
    opt_impr  = []

    jitt_count = []
    inv_count  = []
    swap_count = []
    miss_count = []
    opt_count  = []

    jitt_fp_fix = 0
    inv_fp_fix  = 0
    swap_fp_fix = 0
    miss_fp_fix = 0
    opt_fp_fix  = 0

    for lfp in loc_false_pos:
        match = dt_gt_matches['dts'][lfp['id']][0]
        gtm   = coco_gt.loadAnns(match['gtId'])[0]

        ########################################################
        # original oks value
        dt_kpt_arr = np.array(lfp['keypoints'])
        gt_kpt_arr = np.array(gtm['keypoints'])

        xd = dt_kpt_arr[0::3]; yd = dt_kpt_arr[1::3]; vd = dt_kpt_arr[2::3]
        xg = gt_kpt_arr[0::3]; yg = gt_kpt_arr[1::3]; vg = gt_kpt_arr[2::3]

        dx = xd - xg
        dy = yd - yg

        e = (dx**2 + dy**2) / (sigmas * 2)**2 / (gtm['area']+np.spacing(1)) / 2
        e = e[vg > 0]
        ori_oks = np.sum(np.exp(-e)) / e.shape[0]

        ########################################################
        # without jitter oks value
        dt_kpt_arr = np.array(lfp['keypoints'])     * (np.repeat(np.array(lfp['jitter']),3)==0) + \
                     np.array(lfp['opt_keypoints']) * (np.repeat(np.array(lfp['jitter']),3)==1)
        gt_kpt_arr = np.array(gtm['keypoints'])

        xd = dt_kpt_arr[0::3]; yd = dt_kpt_arr[1::3]; vd = dt_kpt_arr[2::3]
        xg = gt_kpt_arr[0::3]; yg = gt_kpt_arr[1::3]; vg = gt_kpt_arr[2::3]

        dx = xd - xg
        dy = yd - yg

        e = (dx**2 + dy**2) / (sigmas * 2)**2 / (gtm['area']+np.spacing(1)) / 2
        e = e[vg > 0]
        jitt_oks = np.sum(np.exp(-e)) / e.shape[0]

        if sum(lfp['jitter']) > 0:
            jitt_impr.append(jitt_oks - ori_oks)
            jitt_count.append(sum(lfp['jitter']))

        if jitt_oks >= OKS_THRS:
            jitt_fp_fix += 1

        ########################################################
        # without inversion oks value
        dt_kpt_arr = np.array(lfp['keypoints'])     * (np.repeat(np.array(lfp['inversion']),3)==0) + \
                     np.array(lfp['opt_keypoints']) * (np.repeat(np.array(lfp['inversion']),3)==1)
        gt_kpt_arr = np.array(gtm['keypoints'])

        xd = dt_kpt_arr[0::3]; yd = dt_kpt_arr[1::3]; vd = dt_kpt_arr[2::3]
        xg = gt_kpt_arr[0::3]; yg = gt_kpt_arr[1::3]; vg = gt_kpt_arr[2::3]

        dx = xd - xg
        dy = yd - yg

        e = (dx**2 + dy**2) / (sigmas * 2)**2 / (gtm['area']+np.spacing(1)) / 2
        e = e[vg > 0]
        inv_oks = np.sum(np.exp(-e)) / e.shape[0]

        if sum(lfp['inversion']) > 0:
            inv_impr.append(inv_oks  - ori_oks)
            inv_count.append(sum(lfp['inversion']))

        if inv_oks >= OKS_THRS:
            inv_fp_fix += 1

        ########################################################
        # without swap oks value
        dt_kpt_arr = np.array(lfp['keypoints'])     * (np.repeat(np.array(lfp['swap']),3)==0) + \
                     np.array(lfp['opt_keypoints']) * (np.repeat(np.array(lfp['swap']),3)==1)
        gt_kpt_arr = np.array(gtm['keypoints'])

        xd = dt_kpt_arr[0::3]; yd = dt_kpt_arr[1::3]; vd = dt_kpt_arr[2::3]
        xg = gt_kpt_arr[0::3]; yg = gt_kpt_arr[1::3]; vg = gt_kpt_arr[2::3]

        dx = xd - xg
        dy = yd - yg

        e = (dx**2 + dy**2) / (sigmas * 2)**2 / (gtm['area']+np.spacing(1)) / 2
        e = e[vg > 0]
        swap_oks = np.sum(np.exp(-e)) / e.shape[0]

        if sum(lfp['swap']) > 0:
            swap_impr.append(swap_oks - ori_oks)
            swap_count.append(sum(lfp['swap']))

        if swap_oks >= OKS_THRS:
            swap_fp_fix += 1

        ########################################################
        # without miss oks value
        dt_kpt_arr = np.array(lfp['keypoints'])     * (np.repeat(np.array(lfp['miss']),3)==0) + \
                     np.array(lfp['opt_keypoints']) * (np.repeat(np.array(lfp['miss']),3)==1)
        gt_kpt_arr = np.array(gtm['keypoints'])

        xd = dt_kpt_arr[0::3]; yd = dt_kpt_arr[1::3]; vd = dt_kpt_arr[2::3]
        xg = gt_kpt_arr[0::3]; yg = gt_kpt_arr[1::3]; vg = gt_kpt_arr[2::3]

        dx = xd - xg
        dy = yd - yg

        e = (dx**2 + dy**2) / (sigmas * 2)**2 / (gtm['area']+np.spacing(1)) / 2
        e = e[vg > 0]
        miss_oks = np.sum(np.exp(-e)) / e.shape[0]

        if sum(lfp['miss']) > 0:
            miss_impr.append(miss_oks - ori_oks)
            miss_count.append(sum(lfp['miss']))

        if miss_oks >= OKS_THRS:
            miss_fp_fix += 1

        ########################################################
        # optimal oks value
        dt_kpt_arr = np.array(lfp['opt_keypoints'])
        gt_kpt_arr = np.array(gtm['keypoints'])

        xd = dt_kpt_arr[0::3]; yd = dt_kpt_arr[1::3]; vd = dt_kpt_arr[2::3]
        xg = gt_kpt_arr[0::3]; yg = gt_kpt_arr[1::3]; vg = gt_kpt_arr[2::3]

        dx = xd - xg
        dy = yd - yg

        e = (dx**2 + dy**2) / (sigmas * 2)**2 / (gtm['area']+np.spacing(1)) / 2
        e = e[vg > 0]
        opt_oks = np.sum(np.exp(-e)) / e.shape[0]

        opt_impr.append(opt_oks  - ori_oks)
        opt_count.append(sum(lfp['jitter']) + sum(lfp['inversion']) + sum(lfp['swap']) + sum(lfp['miss']))

        if opt_oks >= OKS_THRS:
            opt_fp_fix += 1

    data_to_plot = [jitt_impr,inv_impr,swap_impr,miss_impr]

    fig, ax_boxes = plt.subplots(figsize=(10,10))
    ax_boxes.set_axis_bgcolor('lightgray')
    bp = ax_boxes.boxplot(data_to_plot[::-1],0,'',patch_artist=True,
                          showfliers=False,whis=[25, 75],vert=0)
    plt.scatter(1,.05,marker='o',
            edgecolor='black',color='white',
            linewidth=2,s=100)

    for ind,box in enumerate(bp['boxes']):
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        if ind in [3]:
            # change fill color
            box.set_facecolor('#8C4646')
        if ind in [2]:
            # change fill color
            box.set_facecolor('#D96459')
        if ind in [1]:
            # change fill color
            box.set_facecolor('#F2AE72')
        if ind in [0]:
            # change fill color
            box.set_facecolor('#F2E394')

    for i, median in enumerate(bp['medians']):
        median.set(color='#b2df8a', linewidth=0)
        plt.scatter(median.get_xdata()[0],np.mean(median.get_ydata()),
                   marker='o',edgecolor='black',color='red',
                    alpha=.65,linewidth=0,s=300,zorder=10)

    plt.yticks([1,2,3,4],['miss','swap','inversion','jitter'])
    plt.xlim([0,.6])
    plt.title('OKS improvement without localization errors')
    plt.grid()
    plt.show()

def localizationAUCImpact( coco_analyze ):

    areaRngs    = [[32**2,1e5**2],[32**2,96**2],[96**2,1e5**2]]
    areaRngLbls = ['all','medium','large']
    oksVals     = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    total_res = []

    for areaRng,areaRngLbl in zip(areaRngs,areaRngLbls):
        coco_analyze.params.areaRng    = [areaRng]
        coco_analyze.params.areaRngLbl = [areaRngLbl]

        for oks in oksVals:
            coco_analyze.params.oksThrs = [oks]

            for err in ['miss','swap', 'inversion', 'jitter']:
                coco_analyze.params.err_types = [err]
                coco_analyze.analyze(check_kpts=True,
                                     check_scores=False,
                                     check_bkgd=False)
                coco_analyze.summarize(makeplots=True)
                total_res += coco_analyze.stats

    baselines = {}
    for b in total_res:
        if 'err' not in b:
            baselines[b['oks'],b['areaRng']] = b['auc']

    ## get misses improvements
    misses_imp_med = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                      if 'err' in x and x['err']=='miss' and x['areaRng']=='medium']
    misses_imp_lrg = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                      if 'err' in x and x['err']=='miss' and x['areaRng']=='large']
    misses_imp_all = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                      if 'err' in x and x['err']=='miss' and x['areaRng']=='all']

    ## get swap improvements
    swap_imp_med = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                    if 'err' in x and x['err']=='swap' and x['areaRng']=='medium']
    swap_imp_lrg = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                    if 'err' in x and x['err']=='swap' and x['areaRng']=='large']
    swap_imp_all = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                    if 'err' in x and x['err']=='swap' and x['areaRng']=='all']

    ## get inversion improvements
    inversion_imp_med = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                         if 'err' in x and x['err']=='inversion' and x['areaRng']=='medium']
    inversion_imp_lrg = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                         if 'err' in x and x['err']=='inversion' and x['areaRng']=='large']
    inversion_imp_all = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                         if 'err' in x and x['err']=='inversion' and x['areaRng']=='all']

    ## get jitter improvements
    jitter_imp_med = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                      if 'err' in x and x['err']=='jitter' and x['areaRng']=='medium']
    jitter_imp_lrg = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                      if 'err' in x and x['err']=='jitter' and x['areaRng']=='large']
    jitter_imp_all = [x['auc']-baselines[x['oks'],x['areaRng']] for x in total_res \
                      if 'err' in x and x['err']=='jitter' and x['areaRng']=='all']

    fig, ax = plt.subplots(figsize=(20,10))

    x_miss = [1,6,11]
    y_miss = [np.mean(misses_imp_med),np.mean(misses_imp_lrg),np.mean(misses_imp_all)]
    e_miss = [np.std(misses_imp_med),np.std(misses_imp_lrg),np.std(misses_imp_all)]
    rects1 = ax.bar(x_miss, y_miss, 1, color='#F2E394', yerr=e_miss)

    x_swap = [2,7,12]
    y_swap = [np.mean(swap_imp_med),np.mean(swap_imp_lrg),np.mean(swap_imp_all)]
    e_swap = [np.std(swap_imp_med),np.std(swap_imp_lrg),np.std(swap_imp_all)]
    rects1 = ax.bar(x_swap, y_swap, 1, color='#F2AE72', yerr=e_swap)

    x_inv = [3,8,13]
    y_inv = [np.mean(inversion_imp_med),np.mean(inversion_imp_lrg),np.mean(inversion_imp_all)]
    e_inv = [np.std(inversion_imp_med),np.std(inversion_imp_lrg),np.std(inversion_imp_all)]
    rects1 = ax.bar(x_inv, y_inv, 1, color='#D96459', yerr=e_inv)

    x_jit = [4,9,14]
    y_jit = [np.mean(jitter_imp_med),np.mean(jitter_imp_lrg),np.mean(jitter_imp_all)]
    e_jit = [np.std(jitter_imp_med),np.std(jitter_imp_lrg),np.std(jitter_imp_all)]
    rects1 = ax.bar(x_jit, y_jit, 1, color='#8C4646', yerr=e_jit)

    ax.set_xticks([3,8,13])
    ax.set_xticklabels(['medium','large','all'],fontsize=16)
    plt.title('AUC error impact',fontsize=20)
    plt.grid()
    plt.show()

def localizationKeypointBreakdown( coco_analyze ):
    # analyze only the keypoints
    coco_analyze.analyze(check_kpts=True, check_scores=False, check_bkgd=False)

    corrected_dts = coco_analyze.corrected_dts
    dt_gt_matches = coco_analyze.matches
    coco_gt       = coco_analyze.cocoGt
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []

    if len(corrected_dts) == 0:
        return 0

    for cdt in corrected_dts:
        if cdt['id'] in dt_gt_matches['dts']:
            # make sure that a detection has only one match
            assert(len(dt_gt_matches['dts'][cdt['id']])==1)
            dtm = dt_gt_matches['dts'][cdt['id']][0]

            # make sure that the fields correspond
            assert(cdt['image_id']==dtm['image_id'])
            assert(cdt['score']==dtm['score'])
            assert(cdt['id']==dtm['dtId'])

    # list of all the detections that were matched and for which the keypoints
    # breakdown is performed
    matched_dts = [cdt for cdt in corrected_dts if 'good' in cdt]
    assert(
        len([c for c in corrected_dts if 'inversion' in c])==
        len([c for c in corrected_dts if 'miss' in c])==
        len([c for c in corrected_dts if 'swap' in c])==
        len([c for c in corrected_dts if 'jitter' in c])==
        len([c for c in corrected_dts if 'good' in c])
    )

    print "Number detections: [%d]"%len(corrected_dts)
    print "Number matches:    [%d]"%len(matched_dts)

    good      = 0
    jitter    = 0
    inversion = 0
    swap      = 0
    miss      = 0
    tot       = 0.

    jitt_keypoints = np.zeros(17)
    inv_keypoints  = np.zeros(17)
    swap_keypoints = np.zeros(17)
    miss_keypoints = np.zeros(17)

    for dtm in matched_dts:
        match = dt_gt_matches['dts'][dtm['id']][0]
        gtm   = coco_gt.loadAnns(match['gtId'])[0]

        good      += sum(dtm['good'])
        jitter    += sum(dtm['jitter'])
        inversion += sum(dtm['inversion'])
        swap      += sum(dtm['swap'])
        miss      += sum(dtm['miss'])

        jitt_keypoints += np.array(dtm['jitter'])
        inv_keypoints  += np.array(dtm['inversion'])
        swap_keypoints += np.array(dtm['swap'])
        miss_keypoints += np.array(dtm['miss'])

        assert(sum(dtm['good'])+sum(dtm['jitter'])+
               sum(dtm['inversion'])+sum(dtm['swap'])+
               sum(dtm['miss'])==gtm['num_keypoints'])
        tot += gtm['num_keypoints']

    print "Total Num. keypoints: [%d]"%int(tot)
    print("{:30} [{}]-[{}]").format(" - Good,      [tot]-[perc]:", int(good), 100*(good/tot))
    print("{:30} [{}]-[{}]").format(" - Jitter,    [tot]-[perc]:", int(jitter), 100*(jitter/tot))
    print("{:30} [{}]-[{}]").format(" - Inversion, [tot]-[perc]:", int(inversion), 100*(inversion/tot))
    print("{:30} [{}]-[{}]").format(" - Swap,      [tot]-[perc]:", int(swap), 100*(swap/tot))
    print("{:30} [{}]-[{}]").format(" - Miss,      [tot]-[perc]:", int(miss), 100*(miss/tot))

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
    plt.show()

    print "\nPer keypoint breakdown:"
    print " - Jitter:    ", jitt_keypoints
    print " - Inversion: ", inv_keypoints
    print " - Swap:      ",  swap_keypoints
    print " - Miss:      ",  miss_keypoints

    KEYPOINTS_L = ['Nose','Eyes','Ears','Shoulders','Elbows','Wrists','Hips','Knees','Ankles']
    KEYPOINTS_I = [[0],[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
    COLORS = [c.rgb for c in list(Color("white").range_to(Color("blue"),len(KEYPOINTS_L)))]

    ####################################
    # jitter errors
    TOT_LABELS = []
    ERRORS = []
    for i in KEYPOINTS_I:
        tot_errs = 0
        for l in i:
            tot_errs += jitt_keypoints[l]
        ERRORS.append(tot_errs/float(sum(jitt_keypoints)))

    for lind, l in enumerate(KEYPOINTS_L):
        label_str = '{:5s}: {:2.1f}'.format(l,100*ERRORS[lind])
        TOT_LABELS.append(label_str)

    fig = plt.figure(figsize=(5,5))
    rect = 0,0,0.9,0.9
    ax1 = fig.add_axes(rect)
    patches, autotexts = ax1.pie( ERRORS, colors=COLORS)
    lgd=fig.legend(patches, TOT_LABELS, bbox_to_anchor=(1, 1), loc="upper left",ncol=2,fancybox=True, shadow=True,fontsize=18)
    plt.title('Jitter')
    plt.show()

    ####################################
    # inversion errors
    TOT_LABELS = []
    ERRORS = []
    for i in KEYPOINTS_I:
        tot_errs = 0
        for l in i:
            tot_errs += inv_keypoints[l]
        ERRORS.append(tot_errs/float(sum(inv_keypoints)))

    for lind, l in enumerate(KEYPOINTS_L):
        label_str = '{:5s}: {:2.1f}'.format(l,100*ERRORS[lind])
        TOT_LABELS.append(label_str)

    fig = plt.figure(figsize=(5,5))
    rect = 0,0,0.9,0.9
    ax1 = fig.add_axes(rect)
    patches, autotexts = ax1.pie( ERRORS, colors=COLORS)
    lgd=fig.legend(patches, TOT_LABELS, bbox_to_anchor=(1, 1), loc="upper left",ncol=2,fancybox=True, shadow=True,fontsize=18)
    plt.title('Inversion')
    plt.show()

    ####################################
    # swap errors
    TOT_LABELS = []
    ERRORS = []
    for i in KEYPOINTS_I:
        tot_errs = 0
        for l in i:
            tot_errs += swap_keypoints[l]
        ERRORS.append(tot_errs/float(sum(swap_keypoints)))

    for lind, l in enumerate(KEYPOINTS_L):
        label_str = '{:5s}: {:2.1f}'.format(l,100*ERRORS[lind])
        TOT_LABELS.append(label_str)

    fig = plt.figure(figsize=(5,5))
    rect = 0,0,0.9,0.9
    ax1 = fig.add_axes(rect)
    patches, autotexts = ax1.pie( ERRORS, colors=COLORS)
    lgd=fig.legend(patches, TOT_LABELS, bbox_to_anchor=(1, 1), loc="upper left",ncol=2,fancybox=True, shadow=True,fontsize=18)
    plt.title('Swap')
    plt.show()

    ####################################
    # miss errors
    TOT_LABELS = []
    ERRORS = []
    for i in KEYPOINTS_I:
        tot_errs = 0
        for l in i:
            tot_errs += miss_keypoints[l]
        ERRORS.append(tot_errs/float(sum(miss_keypoints)))

    for lind, l in enumerate(KEYPOINTS_L):
        label_str = '{:5s}: {:2.1f}'.format(l,100*ERRORS[lind])
        TOT_LABELS.append(label_str)

    fig = plt.figure(figsize=(5,5))
    rect = 0,0,0.9,0.9
    ax1 = fig.add_axes(rect)
    patches, autotexts = ax1.pie( ERRORS, colors=COLORS)
    lgd=fig.legend(patches, TOT_LABELS, bbox_to_anchor=(1, 1), loc="upper left",ncol=2,fancybox=True, shadow=True,fontsize=18)
    plt.title('Miss')
    plt.show()

"""
These methods are used to compute the iou between the bounding boxes of all the
annotations in an image
"""
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

"""
These methods are used to find if the order of detections is the same after
applying the optimal scoring and if not how many elements were swapped between
the list sorted by the original scores and by the optimal scores
"""
def merge_and_count(a, b):
    assert a == sorted(a) and b == sorted(b)
    c = []
    count = 0
    i, j = 0, 0
    while i < len(a) and j < len(b):
        c.append(min(b[j], a[i]))
        if b[j] < a[i]:
            count += len(a) - i # number of elements remaining in `a`
            j+=1
        else:
            i+=1
    # now we reached the end of one the lists
    c += a[i:] + b[j:] # append the remainder of the list to C
    return count, c

def sort_and_count(L):
    if len(L) == 1: return 0, L
    n = len(L) // 2
    a, b = L[:n], L[n:]
    ra, a = sort_and_count(a)
    rb, b = sort_and_count(b)
    r, L = merge_and_count(a, b)
    return ra+rb+r, L

def number_of_seq_swaps(permutation):
    # decompose the permutation into disjoint cycles
    nswaps = 0
    seen = set()
    for i in xrange(len(permutation)):
        if i not in seen:
            j = i # begin new cycle that starts with `i`
            while permutation[j] != i:
                j = permutation[j]
                seen.add(j)
                nswaps += 1
    return nswaps

def get_permutation(L1, L2):
    if sorted(L1) != sorted(L2):
        raise ValueError("L2 must be permutation of L1 (%s, %s)" % (L1,L2))

    permutation = map(dict((v, i) for i, v in enumerate(L1)).get, L2)
    assert [L1[p] for p in permutation] == L2
    return permutation
