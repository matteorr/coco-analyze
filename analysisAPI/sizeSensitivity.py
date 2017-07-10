## imports
import os, time
import numpy as np
import matplotlib.pyplot as plt

def sizeSensitivity( coco_analyze, oks, saveDir ):
    loc_dir = saveDir + '/benchmarks_sensitivity/size'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
        os.makedirs(loc_dir + '/all_plots')
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [Size Sensitivity]\n\n")
    tic = time.time()
    paths = {}

    areaRngs    = [[32 ** 2, 64 ** 2],[64 ** 2, 96 ** 2],[96 ** 2, 128 ** 2],
                   [128 ** 2, 1e5 ** 2],[32 ** 2, 1e5 ** 2]]
    areaRngLbls = ['medium','large','xlarge','xxlarge','all']
    err_types = ['miss','swap','inversion','jitter','score','bckgd_false_pos', 'false_neg']

    coco_analyze.params.oksThrs    = [oks]
    coco_analyze.params.err_types  = []
    coco_analyze.params.areaRng    = areaRngs
    coco_analyze.params.areaRngLbl = areaRngLbls
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
    plt.title('Instances Size Distribution',fontsize=20)
    plt.grid()
    path = '%s/size_benchmarks.pdf'%loc_dir
    paths['instance_size_hist'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

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
        f.write("\nPerformance Breakdown over Area for [%s]:\n"%err)
        for s in coco_analyze.stats:
            if s['err']==err:
                f.write("%s: ap[%.3f], max_rec[%.3f]\n"%(s['areaRngLbl'],s['auc'],s['recall']))

        stats += coco_analyze.stats
    stats = [dict(t) for t in set([tuple(s.items()) for s in stats])]
    f.write("\nPerformance Breakdown over Area for [Original Dts]:\n")
    for a in areaRngLbls:
        b = [s for s in stats if s['areaRngLbl']==a and s['err']=='baseline'][0]
        f.write("%s: ap[%.3f], max_rec[%.3f]\n"%(a,b['auc'],b['recall']))

    err_perf = {}
    for s in stats:
        if s['err'] != 'false_neg':
            err_perf[s['err'],s['areaRngLbl']] = s['auc']
        else:
            bckgd_fp = [ss for ss in stats if (ss['err'],ss['areaRngLbl'])==('bckgd_false_pos',s['areaRngLbl'])][0]
            err_perf[s['err'],s['areaRngLbl']] = s['auc'] - bckgd_fp['auc']
    baseline = [err_perf['baseline',area] for area in areaRngLbls]

    size_performance = {}
    for err in err_types:
        if err=='false_neg':
            size_performance[err] = [err_perf[err,area] for area in areaRngLbls]
        else:
            size_performance[err] = [err_perf[err,area]-err_perf['baseline',area] for area in areaRngLbls]
    f.write("\nAP Improvement over Baseline at all area ranges: %s\n"%areaRngLbls)
    for k in size_performance:
        f.write("%s: %s\n"%(k, size_performance[k]))

    oks_75_auc = baseline
    perf_jitt  = size_performance['jitter']
    perf_inv   = size_performance['inversion']
    perf_swap  = size_performance['swap']
    perf_miss  = size_performance['miss']
    perf_score = size_performance['score']
    perf_bk_fp = size_performance['bckgd_false_pos']
    perf_bk_fn = size_performance['false_neg']

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_axis_bgcolor('lightgray')
    plt.ylabel("AP Improvement",fontsize=20)
    plt.title("Error Sensitivity over size @ OKS Eval Thresh=%.2f"%oks,fontsize=20)

    x = [.5,1,1.5,2,  3,3.5,4,4.5,  5.5,6,6.5,7,  8,8.5,9,9.5,
         10.5,11,11.5,12,  13,13.5,14,14.5,  15.5,16,16.5,17]
    y = perf_jitt[:4] + perf_inv[:4] + perf_swap[:4] + \
        perf_miss[:4]  + perf_score[:4] + perf_bk_fp[:4] + perf_bk_fn[:4]

    plt.scatter(x,y,c='b',s=150,alpha=.5,edgecolor='black',linewidth=2)
    plt.plot([.5, 2],   [perf_jitt[4],  perf_jitt[4]],'r--',linewidth=2)
    plt.plot([3, 4.5],  [perf_inv[4],   perf_inv[4]],'r--',linewidth=2)
    plt.plot([5.5, 7],  [perf_swap[4],  perf_swap[4]],'r--',linewidth=2)
    plt.plot([8, 9.5],  [perf_miss[4],  perf_miss[4]],'r--',linewidth=2)
    plt.plot([10.5, 12],[perf_score[4], perf_score[4]],'r--',linewidth=2)
    plt.plot([13, 14.5],[perf_bk_fp[4], perf_bk_fp[4]],'r--',linewidth=2)
    plt.plot([15.5, 17],[perf_bk_fn[4], perf_bk_fn[4]],'r--',linewidth=2)

    yy = -.05/2.
    ax.annotate('Jitter', xy=(1.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=20)

    ax.annotate('Inversion', xy=(3.75,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=20)

    ax.annotate('Swap', xy=(6.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=20)

    ax.annotate('Miss', xy=(8.75,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=20)

    ax.annotate('Score', xy=(11.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=20)

    ax.annotate('Bkgd. FP', xy=(13.75,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=20)

    ax.annotate('FN', xy=(16.25,yy),
            horizontalalignment='center',
            verticalalignment='center',fontsize=20)

    plt.xticks(x,['m','l','xl','xxl','m','l','xl','xxl','m','l','xl','xxl',
                  'm','l','xl','xxl','m','l','xl','xxl','m','l','xl','xxl',
                  'm','l','xl','xxl'])
    plt.xlim([0,17.5])
    plt.ylim([-.05,max(y)+.05])
    plt.grid()
    path = '%s/errors_sensitivity.pdf'%loc_dir
    paths['err_size_sensitivity'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('lightgray')
    plt.ylabel("AP",fontsize=20)
    plt.title("AP Sensitivity over size @ OKS Eval Thresh=%.2f"%oks,fontsize=20)
    x = [1,2,3,4]
    y = oks_75_auc[:4]
    plt.bar(x,y,color='b',alpha=.7,align='center',width=.85)
    plt.plot([.5,4.5], [oks_75_auc[4], oks_75_auc[4]],'r--',linewidth=3)
    plt.xticks(x,['m','l','xl','xxl'])
    plt.xlim([0,5])
    plt.grid()
    path = '%s/ap_sensitivity.pdf'%loc_dir
    paths['ap_size_sensitivity'] = path
    plt.savefig(path,bbox_inches='tight')
    plt.close()

    f.write("\nOKS %.2f:  Sensitivity[%.3f], Impact[%.3f]\n"%(oks, max(oks_75_auc[:4])-min(oks_75_auc[:4]), max(oks_75_auc[:4])-oks_75_auc[4]))
    f.write("Jitter:    Sensitivity[%.3f], Impact[%.3f]\n"%(max(perf_jitt[:4])-min(perf_jitt[:4]),max(perf_jitt[:4])-perf_jitt[4]))
    f.write("Inversion: Sensitivity[%.3f], Impact[%.3f]\n"%(max(perf_inv[:4]) -min(perf_inv[:4]) ,max(perf_inv[:4])-perf_inv[4]))
    f.write("Swap:      Sensitivity[%.3f], Impact[%.3f]\n"%(max(perf_swap[:4])-min(perf_swap[:4]),max(perf_swap[:4])-perf_swap[4]))
    f.write("Miss:      Sensitivity[%.3f], Impact[%.3f]\n"%(max(perf_miss[:4])-min(perf_miss[:4]),max(perf_miss[:4])-perf_miss[4]))
    f.write("Score:     Sensitivity[%.3f], Impact[%.3f]\n"%(max(perf_score[:4])-min(perf_score[:4]),max(perf_score[:4])-perf_score[4]))
    f.write("Bkgd FP:   Sensitivity[%.3f], Impact[%.3f]\n"%(max(perf_bk_fp[:4])-min(perf_bk_fp[:4]),max(perf_bk_fp[:4])-perf_bk_fp[4]))
    f.write("FN:        Sensitivity[%.3f], Impact[%.3f]\n"%(max(perf_bk_fn[:4])-min(perf_bk_fn[:4]),max(perf_bk_fn[:4])-perf_bk_fn[4]))

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

    return paths
