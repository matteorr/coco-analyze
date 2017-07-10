## imports
import os, time
import numpy as np
import matplotlib.pyplot as plt

def errorsAPImpact( coco_analyze, saveDir ):
    loc_dir = saveDir + '/AP_improvement'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
        os.makedirs(loc_dir+'/all_plots')
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [AP Improvement]\n\n")
    tic = time.time()

    paths = {}

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
        f.write("\n[%s] Errors AP improvement (for OKS evaluation thresholds in %s):\n"%(err,[oksThrs[0],oksThrs[-1]]))
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
            plt.xlabel('OKS Evaluation Thresh', fontsize=20); plt.ylabel('AP Improvement',fontsize=20)
            plt.xlim([0.45,1.]); plt.grid()
            plt.savefig('%s/%s_[%s].pdf'%(loc_dir,err.title(),areaRngLbl), bbox_inches='tight')
            plt.close()

    for aind, areaRngLbl in enumerate(areaRngLbls):
        fig, ax = plt.subplots(figsize=(10,10))

        indx_start = aind*len(err_types)
        indx_end   = (aind+1)*len(err_types)

        rects = ax.bar(xs[indx_start:indx_end], means[indx_start:indx_end], 1, color=colors[indx_start:indx_end], yerr=stds[indx_start:indx_end])
        lgd = fig.legend(rects[:len(err_types)], err_types, loc="upper center",
                         ncol=4,fancybox=True, shadow=True,fontsize=20)
        ax.set_xticks([xticks[aind]]); ax.set_xticklabels([areaRngLbls[aind]],fontsize=20)
        plt.ylabel('AP Improvement',fontsize=20); plt.xlabel('Instances Size',fontsize=20)
        path = '%s/ap_improv_areas_%s.pdf'%(loc_dir, areaRngLbl)
        if areaRngLbl == 'all': paths['ap_improv_areas'] = path
        plt.grid(); plt.savefig(path, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=(20,10))
    rects = ax.bar(xs, means, 1, color=colors, yerr=stds)
    lgd = fig.legend(rects[:len(err_types)], err_types, loc="upper center",
                     ncol=len(err_types),fancybox=True, shadow=True,fontsize=20)
    ax.set_xticks(xticks); ax.set_xticklabels(areaRngLbls,fontsize=20)
    plt.ylabel('AP Improvement',fontsize=20); plt.xlabel('Instances Size',fontsize=20)
    plt.grid(); plt.savefig('%s/ap_improv_areas_overall.pdf'%loc_dir, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    for eind, err in enumerate(err_types):
        errs = [np.mean([error_perf[err,'all'][i],error_perf[err,'medium'][i],error_perf[err,'large'][i]]) for i in xrange(len(oksThrs))]
        stds = [np.std([error_perf[err,'all'][i],error_perf[err,'medium'][i],error_perf[err,'large'][i]]) for i in xrange(len(oksThrs))]
        y1   = [x+y for (x,y) in zip(errs,stds)]
        y2   = [x-y for (x,y) in zip(errs,stds)]

        plt.plot(oksThrs,errs,color=err_colors[eind],linewidth=2,marker='o')
        ax.fill_between(oksThrs, y1, y2, facecolor=err_colors[eind], alpha=.6, interpolate=True)

    ax.set_xticks(oksThrs), plt.xlim([oksThrs[0]-.05,oksThrs[-1]+.05])
    plt.ylabel('AP Improvement',fontsize=20); plt.xlabel('Oks Threshold',fontsize=20)
    # plt.title('Instances Size: [all]',fontsize=20)
    paths['ap_improv_oks'] = '%s/ap_improv_oks.pdf'%loc_dir
    plt.grid(); plt.savefig(paths['ap_improv_oks'], bbox_inches='tight')
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

    return paths
