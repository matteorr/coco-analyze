## imports
import os, time
import numpy as np
import matplotlib.pyplot as plt
import utilities

def scoringErrors( coco_analyze, oks, imgs_info, saveDir ):
    loc_dir = saveDir + '/scoring_errors'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Scoring Errors]\n\n")
    tic = time.time()
    paths = {}

    # set parameters for the scoring errors analysis
    coco_analyze.params.areaRng    = [[32 ** 2, 1e5 ** 2]]
    coco_analyze.params.areaRngLbl = ['all']
    coco_analyze.params.oksThrs    = [oks]
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []
    coco_analyze.analyze(check_kpts=False, check_scores=True, check_bckgd=False)
    coco_analyze.summarize(makeplots=True, savedir=loc_dir, team_name='scoring')
    paths['opt_score_prc'] = \
        '%s/error_prc_[scoring][%d][%s][%d].pdf'%(loc_dir, int(oks*100),
            coco_analyze.params.areaRngLbl[0],
            coco_analyze.params.maxDets[0])
    corrected_dts = coco_analyze.corrected_dts['all']

    # dictionary of all corrected detections grouped by image id
    all_dts = {}
    for d in coco_analyze.corrected_dts['all']:
        if d['image_id'] not in all_dts:
            all_dts[d['image_id']] = {}
            all_dts[d['image_id']]['dts'] = [d]
        else:
            all_dts[d['image_id']]['dts'].append(d)

    subopt_order_images = []
    all_gts = {}; all_dtgt_oks = {}
    for imgId in imgs_info:
        if imgId in all_dts:
            dts = all_dts[imgId]['dts']
            all_dts[imgId]['score'] = np.argsort([-d['score'] for d in dts], kind='mergesort')
            all_dts[imgId]['opt_score'] = np.argsort([-d['opt_score'] for d in dts], kind='mergesort')

            if list(all_dts[imgId]['score']) != list(all_dts[imgId]['opt_score']):
                subopt_order_images.append(imgId)
        else:
            dts = []

        gts = coco_analyze.cocoGt.loadAnns(coco_analyze.cocoGt.getAnnIds(imgIds=imgId))
        not_ignore_gts = []
        for g in gts:
            # gt ignores are discarded
            if g['ignore'] or (g['area']<coco_analyze.params.areaRng[0][0] or g['area']>coco_analyze.params.areaRng[0][1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0
                not_ignore_gts.append(g)

        # compute the oks matrix between the dts and gts of each image
        image_oks_mat = utilities.compute_oks(dts, not_ignore_gts)
        if len(image_oks_mat) == 0:
            all_gts[imgId]      = not_ignore_gts
            all_dtgt_oks[imgId] = []

        else:
            # sort the ground truths by their max oks value with any detection
            maxoksvals = [-max(image_oks_mat[:,j]) for j in xrange(len(not_ignore_gts))]
            gtind      = np.argsort(maxoksvals, kind='mergesort')
            all_gts[imgId]      = [not_ignore_gts[j] for j in gtind]
            all_dtgt_oks[imgId] = image_oks_mat[:,gtind]

    ## check how many images have optimal score and original score with same order
    perc = 100*len(subopt_order_images)/float(len(all_dts))
    f.write("Num. of imgs with sub-optimal detections order: [%d]/[%d] (%.2f%%).\n\n"%(len(subopt_order_images), len(all_dts), perc))

    ## find scoring errors before and after rescoring
    min_match_oks = .5
    scoring_errors = {'score':[],'opt_score':[]}
    for score_type in scoring_errors.keys():
        for ind, imgId in enumerate(all_dts.keys()):
            dind       = all_dts[imgId][score_type]
            sorted_dts = [all_dts[imgId]['dts'][i] for i in dind]
            gtIds      = [g['id'] for g in all_gts[imgId]]
            if len(sorted_dts) * len(gtIds) == 0: continue

            used_dts   = []
            for gind, gt in enumerate(all_gts[imgId]):
                assert(gt['_ignore']==0)

                oks = all_dtgt_oks[imgId][dind,gind]
                dts_with_oks = np.where(oks >= min_match_oks)[0]
                # remove the matched dts
                dts_available = [(i,sorted_dts[i]['id'],oks[i],sorted_dts[i][score_type]) \
                                 for i in dts_with_oks if sorted_dts[i]['id'] not in used_dts]
                if len(dts_available) == 0: break

                max_oks_dt = np.argmax([d[2] for d in dts_available])
                used_dts.append(dts_available[max_oks_dt][1])

                if len( dts_available ) > 1:
                    # check for scoring error
                    max_score_dt = np.argmax([d[3] for d in dts_available])
                    if max_score_dt!=max_oks_dt:
                        # this is a scoring error
                        error = {}
                        error['gt']           = gt
                        error['imgId']        = imgId
                        error['matched_dt']   = sorted_dts[dts_available[max_score_dt][0]]
                        error['top_match_dt'] = sorted_dts[dts_available[max_oks_dt][0]]
                        error['high_oks']     = dts_available[max_oks_dt][2]
                        error['low_oks']      = dts_available[max_score_dt][2]
                        scoring_errors[score_type].append(error)

    f.write("Num. of scoring errors:\n")
    f.write(" - Original Score: %d\n"%len(scoring_errors['score']))
    f.write(" - Optimal Score:  %d\n"%len(scoring_errors['opt_score']))

    f.write("\nMost relevant scoring errors:\n")
    ## print the top scoring errors of the algorithm
    ori_scoring_errors = scoring_errors['score']
    ori_scoring_errors.sort(key=lambda k: -np.sqrt((k['matched_dt']['score']-k['top_match_dt']['score'])*(k['high_oks']-k['low_oks'])))
    for ind, err in enumerate(ori_scoring_errors[0:12]):
        relevance = np.sqrt((err['matched_dt']['score']-err['top_match_dt']['score'])*(err['high_oks']-err['low_oks']))
        f.write("================================================\n")
        f.write( "- gt id: [%d]\n"%err['gt']['id'] )
        f.write( "- dt id, high score, low  oks:  [%d][%.3f][%.3f]\n"%(err['matched_dt']['id'], err['matched_dt']['score'], err['low_oks']) )
        f.write( "- dt id, low  score, high oks:  [%d][%.3f][%.3f]\n"%(err['top_match_dt']['id'], err['top_match_dt']['score'], err['high_oks']) )
        f.write( "- Relevance: [%.3f]\n\n"%relevance )

        name        = 'score_err_%d_high_score'%ind
        paths[name] = '%s/%s.pdf'%(loc_dir,name)
        utilities.show_dets([err['matched_dt']],
                  [err['gt']],
                  imgs_info[err['imgId']],save_path=paths[name])

        name        = 'score_err_%d_high_oks'%ind
        paths[name] = '%s/%s.pdf'%(loc_dir,name)
        utilities.show_dets([err['top_match_dt']],
                  [err['gt']],
                  imgs_info[err['imgId']],save_path=paths[name])

    # for all the images with dts and gts compute the following quantities
    # - number of dts with oks > min_match_oks for each gt
    # - histogram of oks for the detection with highest oks
    # - histogram of oks for all the other detections
    # - histogram of original/optimal scores for the detection with highest oks
    # - histogram of original/optimal scores for all the other detections
    num_dts_high_oks           = []
    high_oks_dt_oks_hist       = []; other_dt_oks_hist       = []
    high_oks_dt_ori_score_hist = []; other_dt_ori_score_hist = []
    high_oks_dt_opt_score_hist = []; other_dt_opt_score_hist = []

    for ind, imgId in enumerate(all_dts.keys()):
        dts   = [(d['id'],d['score'],d['opt_score']) for d in all_dts[imgId]['dts']]
        gtIds = [g['id'] for g in all_gts[imgId]]
        if len(dts) * len(gtIds) == 0: continue

        for gind, gt in enumerate(all_gts[imgId]):
            assert(gt['_ignore']==0)

            dts_oks        = all_dtgt_oks[imgId][:,gind]
            dts_high_oks_i = np.where(dts_oks > .1)[0]
            num_dts_high_oks.append(len(dts_high_oks_i))

            if len(dts_high_oks_i) >= 2:
                # study the case where multiple detections have high oks

                # add the oks of the detections to the histogram of oks
                oks_vals = sorted([(dts_oks[i],dts[i]) for i in dts_high_oks_i], key=lambda k: -k[0])
                high_oks_dt_oks_hist.append(oks_vals[0][0])
                other_dt_oks_hist.extend([k[0] for k in oks_vals[1:]])

                high_oks_dt_ori_score_hist.append(oks_vals[0][1][1])
                other_dt_ori_score_hist.extend([k[1][1] for k in oks_vals[1:]])

                high_oks_dt_opt_score_hist.append(oks_vals[0][1][2])
                other_dt_opt_score_hist.extend([k[1][2] for k in oks_vals[1:]])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('lightgray')
    plt.hist(num_dts_high_oks,bins=[i-.5 for i in xrange(max(num_dts_high_oks)+1)],color='green')
    plt.grid()
    plt.xticks([i for i in xrange(max(num_dts_high_oks))])
    plt.title('Histogram of Detection Redundancy',fontsize=20)
    plt.xlabel('Number of Detections with OKS > .1',fontsize=20)
    plt.ylabel('Number of Ground Truth Instances',fontsize=20)
    path = '%s/num_dts_high_oks.pdf'%loc_dir
    paths['num_dts_high_oks'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    y1,binEdges=np.histogram(high_oks_dt_ori_score_hist,bins=19)
    bincenters1 = 0.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters1,y1,'-',linewidth=3,c='b',label='Max OKS Detection')
    min_val1 = min(bincenters1)
    max_val1 = max(bincenters1)

    y2,binEdges=np.histogram(other_dt_ori_score_hist,bins=19)
    bincenters2 = 0.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters2,y2,'--',linewidth=3,c='b',label='Lower OKS Detection(s)')
    min_val2 = min(bincenters2)
    max_val2 = max(bincenters2)

    min_val = min(min_val1,min_val2)
    max_val = max(max_val1,max_val2)

    overlapbins = [min(x,y) for x,y in zip(y1,y2)]
    width = (max_val-min_val)/20.
    ax.bar(np.linspace(min_val,max_val,19), overlapbins, color='red', alpha=.65, width=width,align='center')
    plt.grid()
    plt.xlim([min_val-(max_val-min_val)/20.,max_val+(max_val-min_val)/20.])

    plt.grid()
    plt.legend(loc='upper center',fontsize=20)
    plt.title('Histogram of Original Detection Scores',fontsize=20)
    plt.xlabel('Original Confidence Score',fontsize=20)
    plt.ylabel('Number of Detections',fontsize=20)
    path = '%s/dts_ori_score_hist.pdf'%loc_dir
    paths['dts_ori_score_hist'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    y1,binEdges=np.histogram(high_oks_dt_opt_score_hist,bins=19)
    bincenters1 = 0.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters1,y1,'-',linewidth=3,c='b',label='Max OKS Detection')
    min_val1 = min(bincenters1)
    max_val1 = max(bincenters1)

    y2,binEdges=np.histogram(other_dt_opt_score_hist,bins=19)
    bincenters2 = 0.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters2,y2,'--',linewidth=3,c='b',label='Lower OKS Detection(s)')
    min_val2 = min(bincenters2)
    max_val2 = max(bincenters2)

    min_val = min(min_val1,min_val2)
    max_val = max(max_val1,max_val2)

    overlapbins = [min(x,y) for x,y in zip(y1,y2)]
    width = (max_val-min_val)/20.
    ax.bar(np.linspace(min_val,max_val,19), overlapbins, color='red', alpha=.65, width=width,align='center')
    plt.grid()
    plt.xlim([min_val-(max_val-min_val)/20.,max_val+(max_val-min_val)/20.])

    plt.grid()
    plt.legend(loc='upper center',fontsize=20)
    plt.title('Histogram of Optimal Detection Scores',fontsize=20)
    plt.xlabel('Optimal Confidence Score',fontsize=20)
    plt.ylabel('Number of Detections',fontsize=20)
    path = '%s/dts_opt_score_hist.pdf'%loc_dir
    paths['dts_opt_score_hist'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

    return paths
