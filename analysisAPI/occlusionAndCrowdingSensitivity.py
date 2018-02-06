## imports
import os, time
import numpy as np
import matplotlib.pyplot as plt
import utilities

def occlusionAndCrowdingSensitivity( coco_analyze, oks, saveDir ):
    loc_dir = saveDir + '/benchmarks_sensitivity/occlusion_crowding'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Occlusion and Crowding Sensitivity]\n\n")
    tic = time.time()
    paths = {}

    arearangelabel = 'all'; maxdets = coco_analyze.params.maxDets[0]
    coco_analyze.params.oksThrs    = [oks]
    coco_analyze.params.areaRng    = [[32**2,1e5**2]]
    coco_analyze.params.areaRngLbl = [arearangelabel]
    coco_analyze.params.err_types = ['miss','swap','inversion','jitter']

    # the values below can be changed based on the desired grouping for
    # the analysis of the number of overlaps and number of keypoints 
    IOU_FOR_OVERLAP = .1
    overlap_groups  = [[0],[1,2],[3,4,5,6,7,8]]
    num_kpt_groups  = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17]]

    coco_gt         = coco_analyze.cocoGt
    image_ids       = coco_gt.getImgIds()
    coco_gt_ids     = coco_gt.getAnnIds()

    overlap_index   = {}; keypoints_index = {}
    for img_id in image_ids:
        img_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        img_anns = [k for k in img_anns if 0**2 <= k['area'] < 1e5**2]

        ious = utilities.compute_ious(img_anns)
        eye  = np.eye(len(img_anns))
        for aind, a in enumerate(img_anns):
            num_overlaps  = sum((ious[aind,:]-eye[aind,:])>IOU_FOR_OVERLAP)
            num_keypoints = a['num_keypoints']

            overlap_index.setdefault(num_overlaps, []).append(a['id'])
            keypoints_index.setdefault(num_keypoints, []).append(a['id'])

    benchmark_overlap  = {}; benchmark_keypoint = {}
    for ind, og in enumerate(overlap_groups):
        benchmark_overlap[ind] = {}
        benchmark_overlap[ind]['groups'] = og
        total_gts = []
        for no in og:
            if no in overlap_index:
                # there are `overlap_index[no]` annotations in the dataset that
                # have `no` overlapping annotations with IOU larger than `IOU_FOR_OVERLAP`
                total_gts += overlap_index[no]
            else:
                # if the dataset does not contain annotations overlapping with `no`  
                # other annotations the dictionary `overlap_index` will not contain
                # a key with the value specified in the `overlap_groups` list above
                pass
        benchmark_overlap[ind]['gtIds'] = total_gts
    for ind, nkg in enumerate(num_kpt_groups):
        benchmark_keypoint[ind] = {}
        benchmark_keypoint[ind]['groups'] = nkg
        total_gts = []
        for nk in nkg:
            if nk in keypoints_index:
                # there are `keypoints_index[nk]` annotations in the dataset that
                # have `nk` number of visible keypoints
                total_gts += keypoints_index[nk]
            else:
                # if the dataset does not contain annotations with `nk` number of   
                # visible keypoints the dictionary `keypoints_index` will not contain
                # a key with the value specified in the `num_kpt_groups` list above
                pass
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
                        verticalalignment='center',fontsize=20)
    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.title("Total num. instances",fontsize=20)
    plt.xlabel("Num. keypoints",fontsize=20)
    plt.ylabel("Num. overlapping instances",fontsize=20)
    path = '%s/benchmark_instances.pdf'%loc_dir
    paths['occlusion_crowding_instances'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

    total_keypoints = np.zeros((3,4));
    err_mat = {}
    err_types = ['good', 'miss', 'jitter', 'swap', 'inversion']
    for e in err_types: err_mat[e] = np.zeros((3,4))
    ground_truth_num_keypoints     = np.zeros((3,4))
    for no, nk in sorted(benchmark.keys(), key=lambda element: (element[0], element[1])):
        f.write("\nBenchmark Index [%d,%d]:\n"%(no,nk))

        benchmark_gt_ids = set(benchmark[no,nk]['gtIds'])
        gt_id_ignores = [gid for gid in coco_gt_ids if gid not in benchmark_gt_ids]
        assert(len(coco_gt_ids)==len(benchmark_gt_ids) + len(gt_id_ignores))

        # set ids to ignore
        coco_analyze.cocoEval.params.useGtIgnore = 1
        coco_analyze.cocoEval.params.gtIgnoreIds = set(gt_id_ignores)

        # run
        coco_analyze.analyze(check_kpts=True, check_scores=True, check_bckgd=True)
        coco_analyze.summarize(makeplots=True, savedir=loc_dir, team_name="%d-%d"%(no,nk))
        paths['occlusion_crowding_%d%d'%(no,nk)] = '%s/error_prc_[%d-%d][%d][%s][%d].pdf'%(loc_dir,no,nk,int(oks*100),arearangelabel,maxdets)

        for s in coco_analyze.stats:
            f.write("%s: ap[%.3f], max_rec[%.3f]\n"%(s['err'],s['auc'],s['recall']))

        for gtId in benchmark[no,nk]['gtIds']:
            ground_truth_num_keypoints[no,nk] += coco_gt.loadAnns(gtId)[0]['num_keypoints']

        # analyze sensitivity of localization errors
        corrected_dts = coco_analyze.corrected_dts['all']
        matches       = coco_analyze.localization_matches['all',str(.1),'gts']
        matched_gts = [matches[g] for g in matches \
                       if int(g) in benchmark[no,nk]['gtIds']]
        matched_dts = set([g[0]['dtId'] for g in matched_gts])
        for cdt in corrected_dts:
            if cdt['id'] not in matched_dts:
                continue

            for e in err_types:
                err = np.array(cdt[e])
                err_mat[e][no,nk]      += sum(err)
                total_keypoints[no,nk] += sum(err)

    ############## overall num keypoints
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(total_keypoints, cmap=plt.cm.Blues, interpolation='nearest')

    width, height = total_keypoints.shape
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(int(total_keypoints[x,y])), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',fontsize=20)

    plt.xticks(range(height),['<=5','<=10','<=15','>15'])
    plt.yticks(range(width),['0','1/2','>=3'])
    plt.title("Total num. keypoints",fontsize=20)
    plt.xlabel("Num. keypoints",fontsize=20)
    plt.ylabel("Num. overlapping instances",fontsize=20)
    path = '%s/benchmark_keypoints.pdf'%loc_dir
    paths['occlusion_crowding_kpts'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

    cmaps = [plt.cm.YlOrBr, plt.cm.RdPu, plt.cm.YlOrRd, plt.cm.Reds]
    for eind,e in enumerate(err_types[1:]):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(err_mat[e]/total_keypoints, cmap=cmaps[eind], interpolation='nearest')

        width, height = err_mat[e].shape
        for x in xrange(width):
            for y in xrange(height):
                ax.annotate("%.1f"%(100*err_mat[e][x,y]/float(total_keypoints[x,y])), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',fontsize=20)

        plt.xticks(range(height),['<=5','<=10','<=15','>15'])
        plt.yticks(range(width),['0','1/2','>=3'])
        plt.title("(%%) %s"%e.title(),fontsize=20)
        plt.xlabel("Num. keypoints",fontsize=20)
        plt.ylabel("Num. overlapping instances",fontsize=20)
        path = '%s/%s_benchmark.pdf'%(loc_dir,e)
        paths['occlusion_crowding_%s'%e] = path
        plt.savefig(path,bbox_inches='tight')
        plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

    return paths
