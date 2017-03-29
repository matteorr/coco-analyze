__author__  = 'mrr'
__version__ = '2.0'

import numpy as np
import datetime
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from . import mask as maskUtils
import copy

from cocoeval import COCOeval

## added imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.family']     = 'monospace'
from colour import Color

class COCOanalyze:
    # Interface for analyzing the keypoints detections on the Microsoft COCO dataset.

    def __init__(self, cocoGt, cocoDt, iouType='keypoints'):
        '''
        Initialize COCOanalyze using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''

        # ground truth COCO API
        self.cocoGt   = cocoGt
        # detections COCO API
        self.cocoDt   = cocoDt
        # evaluation COCOeval API
        self.cocoEval = COCOeval(cocoGt,cocoDt,iouType)

        # gt for analysis
        self._gts = cocoGt.loadAnns(cocoGt.getAnnIds())
        # dt for analysis
        self._dts = cocoDt.loadAnns(cocoDt.getAnnIds())
        # store the original detections without any modification
        self._original_dts = {d['id']:d for d in copy.deepcopy(self._dts)}

        # dt-gt matches for keypoint error and false positive analysis
        self.matches = {}

        # dt-gt matches for score error analysis
        self.score_matches     = {}
        self.opt_score_matches = {}

        # dt-gt matches for background error analysis
        self.false_pos_neg_matches = {}

        # dt with corrections
        self.corrected_dts = []
        # ground truths with info about false negatives
        self.false_neg_gts = []

        # evaluation parameters
        self.params        = {}
        self.params        = Params(iouType=iouType)
        self.params.imgIds = sorted(cocoGt.getImgIds())
        self.params.catIds = sorted(cocoGt.getCatIds())

        # get the max number of detections each team has per image
        self.cocoEval._prepare()
        self.params.teamMaxDets = [max([len(self.cocoEval._dts[k]) for k in self.cocoEval._dts.keys()])]

        # result summarization
        self.stats = []

    def evaluate(self, verbose=False, makeplots=False, savedir=None, team_name=None):
        # at any point the evaluate function is called it will run the COCOeval
        # API on the current detections

        # set the cocoEval params based on the params from COCOanalyze
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = sorted(self.params.oksThrs)

        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.cocoEval.summarize(verbose)

        # for all areaRngLbl and maxDets plot pr curves at all iouThrs values
        recalls = self.cocoEval.params.recThrs[:]
        # dimension of precision: [TxRxKxAxM]
        ps_mat  = self.cocoEval.eval['precision'][::-1,:,:,:,:]

        # stats are the same returned from cocoEval.stats
        self.stats = self.cocoEval.stats

        if makeplots:
            self._plot(recalls=recalls,ps_mat=ps_mat,params=self.cocoEval.params,
                       savedir=savedir,team_name=team_name)

    def analyze(self, check_kpts=True, check_scores=True, check_bkgd=True):
        if self.corrected_dts:
            # reset dts to the original dts so the same study can be repeated
            for d in self._dts:
                d['keypoints'] = self._original_dts[d['id']]['keypoints']
                d['score']     = self._original_dts[d['id']]['score']
        self.corrected_dts = copy.deepcopy(self._dts)
        # reset false negatives to empty
        self.false_neg_gts = []

        # find keypoint errors in detections that are matched to ground truths
        # (localization false positives)
        if check_kpts:
            self.correct_keypoints()

            # change the detections in the cocoEval object to the corrected kpts
            for cdt in self.corrected_dts:
                if 'opt_keypoints' not in cdt: continue
                dtid     = cdt['id']
                image_id = cdt['image_id']

                # loop through all detections in the image and change only the
                # corresponsing detection cdt being analyzed
                for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                    if d['id'] == dtid:

                        kpts_mask = np.zeros(len(cdt['good']))
                        if 'miss' in self.params.err_types:
                            kpts_mask += np.array(cdt['miss'])

                        if 'swap' in self.params.err_types:
                            kpts_mask += np.array(cdt['swap'])

                        if 'inversion' in self.params.err_types:
                            kpts_mask += np.array(cdt['inversion'])

                        if 'jitter' in self.params.err_types:
                            kpts_mask += np.array(cdt['jitter'])

                        d['keypoints'] = \
                            cdt['keypoints'] * (np.repeat(np.array(cdt['good']),3)==1) + \
                            cdt['opt_keypoints'] * (np.repeat(kpts_mask,3)==1)

                        break

        self.params.check_kpts = check_kpts

        # find scoring errors in all detections
        if check_scores:
            self.correct_scores()

        # change the detections in the cocoEval object to the original kpts
        for cdt in self.corrected_dts:
            #if 'opt_keypoints' not in cdt: continue
            dtid     = cdt['id']
            image_id = cdt['image_id']

            # loop through all detections in the image and change only the
            # corresponsing detection cdt being analyzed
            for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                if d['id'] == dtid:
                    d['keypoints'] = cdt['keypoints']
                    break

        self.params.check_scores = check_scores

        # false positive and false negatives are dealt with in summarize()
        self.params.check_bkgd = check_bkgd

    def correct_keypoints(self):
        tic = time.time()
        print('<{}:{}>Analyzing keypoint errors...'.format(__author__,__version__))

        # find all matches between dts and gts at the lowest iou thresh
        dtMatches, gtMatches = self._find_dt_matches(self.params.oksLocThrs)
        self.matches['dts'] = dtMatches
        self.matches['gts'] = gtMatches

        # find which errors affect the oks of detections that are matched
        corrected_dts = self._find_kpt_errors()

        # save the optimal scores in a temporary dictionary and
        # change the keypoints to the optimized versions
        corrected_dts_dict = {}
        for cdt in corrected_dts:
            corrected_dts_dict[cdt['id']] = cdt
        assert(len(corrected_dts) == len(corrected_dts_dict))

        for d in self.corrected_dts:
            if d['id'] in corrected_dts_dict:
                d['opt_keypoints'] = corrected_dts_dict[d['id']]['keypoints']
                d['inversion']     = corrected_dts_dict[d['id']]['inversion']
                d['good']          = corrected_dts_dict[d['id']]['good']
                d['jitter']        = corrected_dts_dict[d['id']]['jitter']
                d['miss']          = corrected_dts_dict[d['id']]['miss']
                d['swap']          = corrected_dts_dict[d['id']]['swap']

        toc = time.time()
        print('<{}:{}>DONE (t={:0.2f}s).'.format(__author__,__version__,toc-tic))

    def correct_scores(self):
        tic = time.time()
        print('<{}:{}>Analyzing detection scores...'.format(__author__,__version__))

        # find matches before changing the scores
        dtMatches, gtMatches = self._find_dt_matches(min(self.params.oksThrs))
        self.score_matches['dts'] = dtMatches
        self.score_matches['gts'] = gtMatches

        # run the evaluation with no limit on max number of detections
        # note that for optimal score the oks thresh doesnt matter
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.iouThrs    = [min(self.params.oksThrs)] #[self.params.oksThrs[-1]]
        self.cocoEval.params.maxDets    = self.params.teamMaxDets

        # run the evaluation with check scores flag
        self.cocoEval.evaluate(check_scores=True)
        evalImgs = [e for e in filter(None,self.cocoEval.evalImgs)]

        # save the optimal scores in a temporary dictionary and
        # change the keypoints scores to the optimal value
        opt_scores = {}
        for e in evalImgs:
            dtIds       = e['dtIds']
            dtOptScores = e['dtOptScores']
            for i,j in zip(dtIds,dtOptScores):
                opt_scores[i] = j
        assert(len(opt_scores) == len(self._dts))

        for d in self.corrected_dts:
            d['opt_score'] = opt_scores[d['id']]

        toc = time.time()
        print('<{}:{}>DONE (t={:0.2f}s).'.format(__author__,__version__,toc-tic))

    def _find_dt_matches(self, oksThrs):
        # evalute at oks localization threshold (.1) to get all possible matches

        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.iouThrs    = [oksThrs]
        self.cocoEval.evaluate()

        evalImgs = [e for e in filter(None,self.cocoEval.evalImgs)]
        # evalImgs = [e for e in filter(None,self.cocoEval.evalImgs) if
        #             e['aRng']==self.params.areaRng]

        dtMatches = {}
        gtMatches = {}

        for i, e in enumerate(evalImgs):
            # add all matches to the dtMatches dictionary
            for dind, did in enumerate(e['dtIds']):
                gtMatch = int(e['dtMatches'][0][dind])

                if gtMatch != 0:
                    # check that a detection is not already matched
                    assert(did not in dtMatches)
                    dtMatches[did] = [{'gtId'   :gtMatch,
                                      'dtId'    :did,
                                      'oks'     :e['dtIous'][0][dind],
                                      'score'   :e['dtScores'][dind],
                                      'ignore'  :int(e['dtIgnore'][0][dind]),
                                      'image_id':e['image_id']}]
                    # add the gt match as well since multiple dts can have same gt
                    entry = {'dtId'    :did,
                             'gtId'    :gtMatch,
                             'oks'     :e['dtIous'][0][dind],
                             'ignore'  :int(e['dtIgnore'][0][dind]),
                             'image_id':e['image_id']}
                    gtMatches.setdefault(gtMatch, []).append(entry)

            # add matches to the gtMatches dictionary
            for gind, gid in enumerate(e['gtIds']):
                dtMatch = int(e['gtMatches'][0][gind])

                if dtMatch != 0:
                    entry = {'dtId'    :dtMatch,
                             'gtId'    :gid,
                             'oks'     :e['gtIous'][0][gind],
                             'ignore'  :int(e['gtIgnore'][gind]),
                             'image_id':e['image_id']}
                    if gid in gtMatches:
                        if entry not in gtMatches[gid]:
                            gtMatches[gid].append(entry)
                    else:
                        gtMatches[gid] = [entry]

        return dtMatches, gtMatches

    def _find_kpt_errors(self):
        tic = time.time()
        print "Finding all errors causing false positives..."
        zero_kpt_gts  = 0
        corrected_dts = []

        # this contains all the detections that have been matched with a gt
        for did in self.matches['dts']:

            '''
            # get the info on the [dt,gt] match
            # load the detection and ground truth annotations
            '''
            dtm        = self.matches['dts'][did][0]
            image_id   = dtm['image_id']

            dt         = self.cocoDt.loadAnns(did)[0]
            dt_kpt_x   = np.array(dt['keypoints'][0::3])
            dt_kpt_y   = np.array(dt['keypoints'][1::3])
            dt_kpt_v   = np.array(dt['keypoints'][2::3])
            dt_kpt_arr = np.delete(np.array(dt['keypoints']), slice(2, None, 3))

            gt         = self.cocoGt.loadAnns(dtm['gtId'])[0]
            gt_kpt_x   = np.array(gt['keypoints'][0::3])
            gt_kpt_y   = np.array(gt['keypoints'][1::3])
            gt_kpt_v   = np.array(gt['keypoints'][2::3])

            '''
            # if the gt match has no keypoint annotations the analysis
            # cannot be carried out.
            '''
            if gt['num_keypoints'] == 0:
                zero_kpt_gts += 1
                continue

            '''
            # for every detection match return a dictionary with the following info:
            #  - image_id
            #  - detection_id
            #  - 'corrected_keypoints': list containing good value for each keypoint
            #  - 'jitt': binary list identifying jitter errors
            #  - 'inv':  binary list identifying inversion errors
            #  - 'miss': binary list identifying miss errors
            #  - 'swap': binary list identifying swap errors
            '''
            # load all annotations for the image being analyzed
            image_anns = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=image_id))
            num_anns   = len(image_anns)

            # create a matrix with all keypoints from gts in the image
            # dimensions are (2n, 34), 34 for x and y coordinates of kpts
            # 2n for both the original and inverted gt vectors
            gts_kpt_mat = np.zeros((2*num_anns, 2*self.params.num_kpts))
            vflags      = np.zeros((2*num_anns, self.params.num_kpts))
            areas       = np.zeros(2*num_anns)
            indx        = 1

            for a in image_anns:
                # get the keypoint vector and its inverted version
                xs = np.array(a['keypoints'][0::3])
                ys = np.array(a['keypoints'][1::3])
                vs = np.array(a['keypoints'][2::3])
                inv_vs = vs[self.params.inv_idx]

                keypoints     = np.insert(ys, np.arange(self.params.num_kpts), xs)
                inv_keypoints = np.insert(ys[self.params.inv_idx],
                                          np.arange(self.params.num_kpts),
                                          xs[self.params.inv_idx])

                # check if it is the ground truth match, if so put at index 0 and n
                if a['id']==gt['id']:
                    areas[0]                = a['area']
                    areas[num_anns]         = a['area']
                    gts_kpt_mat[0,:]        = keypoints
                    gts_kpt_mat[num_anns,:] = inv_keypoints

                    vflags[0,:]        = vs
                    vflags[num_anns,:] = inv_vs

                else:
                    areas[indx]                  = a['area']
                    areas[indx+num_anns]         = a['area']
                    gts_kpt_mat[indx,:]          = keypoints
                    gts_kpt_mat[indx+num_anns,:] = inv_keypoints

                    vflags[indx,:]          = vs
                    vflags[indx+num_anns,:] = inv_vs

                    indx += 1

            # compute OKS of every individual dt keypoint with corresponding gt
            dist = gts_kpt_mat - dt_kpt_arr
            sqrd_dist = np.add.reduceat(np.square(dist), range(0,2*self.params.num_kpts,2),axis=1)


            kpts_oks_mat = np.exp( -sqrd_dist / (self.params.sigmas*2)**2 / (areas[:,np.newaxis]+np.spacing(1)) / 2 ) * (vflags>0) +\
                           -1 * (vflags==0)

            div = np.sum(vflags>0,axis=1)
            div[div==0] = self.params.num_kpts

            oks_mat = (np.sum(kpts_oks_mat * (vflags>0), axis=1) / div) * ( np.sum(vflags>0,axis=1) > 0 ) + \
                       -1 * ( np.sum(vflags>0,axis=1) == 0 )
            assert(np.isclose(oks_mat[0],dtm['oks'],atol=1e-08))

            # NOTE: if a 0 or a -1 appear in the oks_max array it doesn't matter
            # since that will automatically become a miss
            oks_max    = np.amax(kpts_oks_mat,axis=0)
            assert(np.all(vflags[:,np.where(oks_max<0)]==0))
            oks_max[np.where(oks_max<0)] = 0
            oks_argmax = np.argmax(kpts_oks_mat,axis=0)

            # good keypoints are those that have oks max > 0.85 and argmax 0
            good_kpts = np.logical_and.reduce((oks_max > self.params.jitterKsThrs[1],
                                               oks_argmax == 0, gt_kpt_v != 0))*1

            # jitter keypoints have  0.5 <= oksm < 0.85 and oks_argmax == 0
            jitt_kpts = np.logical_and.reduce((oks_max >= self.params.jitterKsThrs[0],
                                               oks_max <  self.params.jitterKsThrs[1], oks_argmax == 0))
            jitt_kpts = np.logical_and(jitt_kpts, gt_kpt_v != 0)*1

            # inverted keypoints are those that have oks => 0.5 but on the inverted keypoint entry
            inv_kpts   = np.logical_and.reduce((oks_max >= self.params.jitterKsThrs[0],
                                                oks_argmax == num_anns, gt_kpt_v != 0))*1

            # swapped keypoints are those that have oks => 0.5 but on keypoint of other person
            swap_kpts  = np.logical_and.reduce((oks_max >= self.params.jitterKsThrs[0],
                                                oks_argmax != 0, oks_argmax != num_anns))
            swap_kpts  = np.logical_and(swap_kpts, gt_kpt_v != 0)*1

            # missed keypoints are those that have oks max < 0.5
            miss_kpts  = np.logical_and(oks_max < self.params.jitterKsThrs[0],
                                        gt_kpt_v != 0)*1

            # compute what it means in terms of pixels to be at a certain oks score
            # for simplicity it's computed only along one dimension and added only to x
            dist_to_oks_low  = np.sqrt(-np.log(self.params.jitterKsThrs[0])*2*gt['area']*(self.params.sigmas**2))
            dist_to_oks_high = np.sqrt(-np.log(self.params.jitterKsThrs[1])*2*gt['area']*(self.params.sigmas**2))
            # note that for swaps we use the current ground truth match area because we
            # have to translate the oks to the scale of correct ground truth
            # round oks values to deal with numerical instabilities
            round_oks_max   = oks_max + np.spacing(1)*(oks_max==0) - np.spacing(1)*(oks_max==1)

            dist_to_oks_max = np.sqrt(-np.log(round_oks_max)*2*gt['area']*(self.params.sigmas**2))

            # correct keypoints vectors using info from all the flag vectors
            correct_kpts_x = dt_kpt_x * good_kpts + \
                            (gt_kpt_x + dist_to_oks_high) * jitt_kpts + \
                            (gt_kpt_x + dist_to_oks_low)  * miss_kpts + \
                             dt_kpt_x * (gt_kpt_v == 0) + \
                            (gt_kpt_x + dist_to_oks_max) * inv_kpts  + \
                            (gt_kpt_x + dist_to_oks_max) * swap_kpts

            correct_kpts_y = dt_kpt_y * good_kpts + \
                             gt_kpt_y * jitt_kpts + \
                             gt_kpt_y * miss_kpts + \
                             dt_kpt_y * (gt_kpt_v == 0) + \
                             gt_kpt_y * inv_kpts  + gt_kpt_y * swap_kpts

            correct_kpts       = np.zeros(self.params.num_kpts*3).tolist()
            correct_kpts[0::3] = correct_kpts_x.tolist()
            correct_kpts[1::3] = correct_kpts_y.tolist()
            correct_kpts[2::3] = dt_kpt_v

            new_dt = {}
            new_dt['id']        = dt['id']
            new_dt['image_id']  = int(dt['image_id'])
            new_dt['keypoints'] = correct_kpts

            new_dt['good']      = good_kpts.tolist()
            new_dt['jitter']    = jitt_kpts.tolist()
            new_dt['inversion'] = inv_kpts.tolist()
            new_dt['swap']      = swap_kpts.tolist()
            new_dt['miss']      = miss_kpts.tolist()

            corrected_dts.append(new_dt)

        print "Done (t=%0.2fs)."%(time.time()-tic)
        return corrected_dts

    def summarize(self, makeplots=False, savedir=None, team_name=None):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        if not self.corrected_dts:
            raise Exception('<{}:{}>Please run analyze() first'.format(__author__,__version__))

        # set the cocoEval params based on the params from COCOanalyze
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = sorted(self.params.oksThrs)

        # clear the stats for the summarize results
        self.stats = []

        # compute all the precision recall curves and return precise breakdown of
        # all error type in terms of keypoint, scoring, false positives and negatives
        ps_mat = self._summarize_baseline()

        # summarize keypoint errors
        if self.params.check_kpts and self.params.err_types:
            ps_mat_kpt_errors = self._summarize_kpt_errors()
            ps_mat = np.append(ps_mat,ps_mat_kpt_errors,axis=0)

        # summarize scoring errors
        if self.params.check_scores:
            ps_mat_score_errors = self._summarize_score_errors()
            # append the two error types and invert the precision matrix for plotting
            ps_mat = np.append(ps_mat,ps_mat_score_errors,axis=0)

        # summarize detections that are unmatched (hallucinated false positives)
        # and ground truths that are unmatched (false negatives)
        if self.params.check_bkgd:
            ps_mat_false_errors = self._summarize_false_errors()
            ps_mat = np.append(ps_mat,ps_mat_false_errors,axis=0)

        # plot
        if makeplots:
            recalls = self.cocoEval.params.recThrs[:]

            self.cocoEval.params.maxDets = self.params.maxDets
            self.cocoEval.params.iouThrs = sorted(self.params.oksThrs)
            params = self.cocoEval.params

            err_labels = []
            colors_vec = []
            if self.params.check_kpts:
                for err in self.params.err_types:
                    if err == 'miss':
                        err_labels.append('Miss')
                        colors_vec.append('#F2E394')
                    if err == 'swap':
                        err_labels.append('Swap')
                        colors_vec.append('#F2AE72')
                    if err == 'inversion':
                        err_labels.append('Inv.')
                        colors_vec.append('#D96459')
                    if err == 'jitter':
                        err_labels.append('Jit.')
                        colors_vec.append('#8C4646')
                # err_labels += ['Miss',    'Swap',    'Inv.',    'Jit.']
                # colors_vec += ['#F2E394', '#F2AE72', '#D96459', '#8C4646']

            if self.params.check_scores:
                err_labels += ['Score']
                colors_vec += ['#4F82BD']

            if self.params.check_bkgd:
                err_labels += ['Bkg.',   'FN']
                colors_vec += ['#8063A3','seagreen']

            self._plot(recalls, ps_mat, params, err_labels, colors_vec, savedir, team_name)

    def _summarize_baseline(self):
        '''
        Run the evaluation on the original detections to get the baseline for
        algorithm performance that will be compared to corrected detections.
        '''
        # get parameters
        evalParams = self.cocoEval.params
        oksThrs    = sorted(self.params.oksThrs)

        # define the precision matrix that will contain all results
        counts = [len(oksThrs),
                  len(evalParams.recThrs),
                  len(evalParams.catIds),
                  len(evalParams.areaRng),
                  len(evalParams.maxDets)]
        ps = np.zeros(counts)

        # set area range and the oks thresholds
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.iouThrs    = oksThrs

        # evaluate
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.cocoEval.summarize(verbose=True)

        # insert results into the precision matrix
        ps[:,:,:,:,:]  = self.cocoEval.eval['precision'][::-1,:,:,:,:]

        # add the stats to the stats list
        for oind, oks in enumerate(oksThrs):
            stat = {}
            stat['oks'] = oks
            pind = oind
            oks_idx_skip = 3 if len(oksThrs) > 1 else 0
            rind = len(oksThrs) + oks_idx_skip + oind
            stat['auc']     = self.cocoEval.stats[pind]
            stat['recall']  = self.cocoEval.stats[rind]
            stat['areaRng'] = self.params.areaRngLbl[0]
            self.stats.append(stat)

        return ps

    def _summarize_kpt_errors(self):
        '''
        Use the corrected detections to recompute performance of algorithm.
        Visualize results and plot precision recall curves if required by input variables.
        '''
        # indx_list = [0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,
        #              27,28,30,31,33,34,36,37,39,40,42,43,45,46,48,49]
        indx_list = [i for i in xrange(self.params.num_kpts*3) if (i-2)%3 != 0]

        # set the error_types
        err_types = self.params.err_types
        assert(len(err_types)>0)

        # get evaluation params
        evalParams = self.cocoEval.params

        # define the precision matrix that will contain all results
        counts = [len(err_types),
                  len(evalParams.recThrs),
                  len(evalParams.catIds),
                  len(evalParams.areaRng),
                  len(evalParams.maxDets)]
        ps = np.zeros(counts)

        # set area range and the oks thresholds
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl

        # set the oks thresh to the smallest value of oks thresh
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.iouThrs = [min(oksThrs)]

        # compute performance after solving for each error type
        for eind,err in enumerate(err_types):
            print('<{}:{}>Correcting error type [{}]:'.format(__author__,__version__,err))

            # correct
            for cdt in self.corrected_dts:
                # this detection doesn't have keypoint errors (wasn't matched)
                if err not in cdt.keys():
                    continue

                # check if detection has error of that type
                if sum(cdt[err]) != 0:
                    dtid     = cdt['id']
                    image_id = cdt['image_id']

                    corrected_kpts = np.array(cdt['opt_keypoints'])

                    for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                        if d['id'] == dtid:
                            keys = np.delete(np.array(d['keypoints']), slice(2, None, 3)) * \
                                   np.repeat(np.logical_not(cdt[err])*1,2) + \
                                   np.delete(np.array(corrected_kpts), slice(2, None, 3)) * \
                                   np.repeat(cdt[err],2)

                            d['keypoints'] = np.array(d['keypoints'])
                            d['keypoints'][indx_list] = keys
                            d['keypoints'] = d['keypoints'].tolist()

                            break

            # evaluate
            self.cocoEval.evaluate()
            self.cocoEval.accumulate()
            self.cocoEval.summarize(verbose=True)

            stat = {}
            stat['oks']     = self.cocoEval.params.iouThrs[0]
            stat['err']     = err
            stat['auc']     = self.cocoEval.stats[0]
            stat['recall']  = self.cocoEval.stats[1]
            stat['areaRng'] = self.params.areaRngLbl[0]
            self.stats.append(stat)

            # store
            ps[eind,:,:,:,:] = self.cocoEval.eval['precision'][0,:,:,:,:]

        return ps

    def _summarize_score_errors(self):
        '''
        Use the corrected detections to recompute performance of algorithm.
        Visualize results and plot precision recall curves if required by input variables.
        '''
        # load corrected detections and rerun evaluation
        err = "score"
        print('<{}:{}>Correcting error type [{}]:'.format(__author__,__version__,err))
        for cdt in self.corrected_dts:
            dtid     = cdt['id']
            image_id = cdt['image_id']

            for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                if d['id'] == dtid:
                    d['score'] = cdt['opt_score']
                    break

        dtMatches, gtMatches = self._find_dt_matches(min(self.params.oksThrs))
        self.opt_score_matches['dts'] = dtMatches
        self.opt_score_matches['gts'] = gtMatches

        # set area range and the oks thresh to last value of input array
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.iouThrs    = [min(self.params.oksThrs)]
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.cocoEval.summarize(verbose=True)

        stat = {}
        stat['oks']     = self.cocoEval.params.iouThrs[0]
        stat['err']     = 'score'
        stat['auc']     = self.cocoEval.stats[0]
        stat['recall']  = self.cocoEval.stats[1]
        stat['areaRng'] = self.params.areaRngLbl[0]
        self.stats.append(stat)

        return self.cocoEval.eval['precision']

    def _summarize_false_errors(self):
        err = "bkg. fp, fn"
        print('<{}:{}>Correcting error type [{}]:'.format(__author__,__version__,err))
        # compute matches with current value of detections to determine new matches
        dtMatches, gtMatches = self._find_dt_matches(min(self.params.oksThrs))
        self.false_pos_neg_matches['dts'] = dtMatches
        self.false_pos_neg_matches['gts'] = gtMatches

        # assert that detection and ground truth matches are consistent
        for d in dtMatches:
            # assert that every detection matched has a corresponding gt in the gt matches dictionary
            assert(dtMatches[d][0]['gtId'] in gtMatches)
            # assert that this detection is in the dt matches of the gt it is matched to
            assert(d in [dt['dtId'] for dt in gtMatches[dtMatches[d][0]['gtId']]])

        # assert that all ground truth with multiple detection matches should be ignored
        count = 0
        for g in gtMatches:
            count+= len(gtMatches[g])
            if len(gtMatches[g])>1:
                # if this gt already has multiple matches assert it is a crowd
                # since crowd gt can be matched to multiple detections
                assert(self.cocoGt.anns[g]['iscrowd']==1)
            assert(gtMatches[g][0]['dtId'] in dtMatches)
        assert(count==len(dtMatches))

        # store info about false positives in corrected detections
        false_pos = set([dt['id'] for dt in self._dts if dt['id'] not in dtMatches])
        for cdt in self.corrected_dts:
            cdt['false_pos'] = True if cdt['id'] in false_pos else False

        # store info about false negatives in corrected ground truths
        false_neg = set([gt for gt in self.cocoGt.getAnnIds() if gt not in gtMatches])
        for gt in self._gts:
            if gt['id'] in false_neg:
                self.false_neg_gts.append(gt)

        # print "Number of false dts:  [%d]"%len(false_pos)
        # print "Number of missed gts: [%d]"%len(false_neg)

        # set unmatched detections to ignore and remeasure performance with accumulate
        for e in self.cocoEval.evalImgs:
            if e is None: continue

            for dind,dtid in enumerate(e['dtIds']):
                if dtid in false_pos:
                    e['dtIgnore'][0][dind] = True

        self.cocoEval.accumulate()
        self.cocoEval.summarize(verbose=True)
        ps_mat_false_pos = self.cocoEval.eval['precision']

        stat = {}
        stat['oks']     = self.cocoEval.params.iouThrs[0]
        stat['err']     = 'bkg_false_pos'
        stat['auc']     = self.cocoEval.stats[0]
        stat['recall']  = self.cocoEval.stats[1]
        stat['areaRng'] = self.params.areaRngLbl[0]
        self.stats.append(stat)

        # set unmatched ground truths to ignore and remeausre performance
        for e in self.cocoEval.evalImgs:
            if e is None: continue

            for gind,gtid in enumerate(e['gtIds']):
                if gtid in false_neg:
                    e['gtIgnore'][gind] = 1

        self.cocoEval.accumulate()
        self.cocoEval.summarize(verbose=True)
        ps_mat_false_neg = self.cocoEval.eval['precision']

        stat = {}
        stat['oks']     = self.cocoEval.params.iouThrs[0]
        stat['err']     = 'false_neg'
        stat['auc']     = self.cocoEval.stats[0]
        stat['recall']  = self.cocoEval.stats[1]
        stat['areaRng'] = self.params.areaRngLbl[0]
        self.stats.append(stat)

        return np.append(ps_mat_false_pos, ps_mat_false_neg,axis=0)

    @staticmethod
    def _plot(recalls, ps_mat, params, err_labels=[], color_vec=[], savedir=None, team_name=None):
        '''
        Plot and save or show the precision recall curves.
        '''
        iouThrs    = params.iouThrs
        iouType    = params.iouType
        areaRngLbl = params.areaRngLbl
        maxDets    = params.maxDets
        catId      = 0

        iouType = 'Oks' if params.iouType == 'keypoints' else 'IoU'

        if err_labels:
            labels = [str(o) for o in iouThrs][::-1] + err_labels
            colors = list(Color("white").range_to(Color("seagreen"),len(labels)))
            colors[-len(err_labels):] = \
                [Color(c) for c in color_vec]

        else:
            labels = [str(o) for o in iouThrs][::-1]
            colors = list(Color("white").range_to(Color("seagreen"),len(labels)))

        for aind,a in enumerate(areaRngLbl):
            for mind,m in enumerate(maxDets):
                fig=plt.figure(figsize=(10,8))
                #ax=plt.gca()
                ax = fig.add_axes([0.1, 0.15, 0.56, 0.7])
                plt.title('areaRng:[{}], maxDets:[{}]'.format(a,m),fontsize=18)
                legend_patches=[]

                for lind, l in enumerate(labels):
                    precisions = ps_mat[lind,:,catId,aind,mind]
                    plt.plot(recalls,precisions,c='k',ls='-',lw=2)

                    if lind > 0:
                        prev_precisions = ps_mat[lind-1,:,catId,aind,mind]
                        plt.fill_between(recalls,
                                         prev_precisions, precisions,
                                         where=precisions >= prev_precisions,
                                         facecolor=colors[lind].rgb, interpolate=True)

                    m_map = np.mean(precisions[precisions>-1])
                    interm_m_map = '%.3f'%m_map
                    m_map_val_str = interm_m_map[1-int(interm_m_map[0]):5-int(interm_m_map[0])]

                    if l not in err_labels:
                        the_label = '{} {:<3}: {}'.format(iouType,str(l)[1:],m_map_val_str)
                    else:
                        the_label = '{:<7}: {}'.format(l,m_map_val_str)
                    patch = mpatches.Patch(facecolor=colors[lind].rgb,
                                           edgecolor='k',
                                           linewidth=1.5,
                                           label=the_label)
                    legend_patches.append(patch)

                plt.xlim([0,1]); plt.ylim([0,1]); plt.grid()
                plt.xlabel('recall',fontsize=18); plt.ylabel('precision',fontsize=18)

                lgd = plt.legend(handles=legend_patches[::-1], ncol=1,
                                 bbox_to_anchor=(1, 1), loc='upper left',
                                 fancybox=True, shadow=True,fontsize=18 )

                if savedir == None:
                    plt.show()
                else:
                    prefix = 'error_prc' if err_labels else 'prc'
                    oks_str = '[%s]'%(int(100*iouThrs[0])) if err_labels else ''

                    savepath = '{}/{}_[{}][{}][{}]{}.pdf'.format(savedir,prefix,team_name,a,m,oks_str)
                    plt.savefig(savepath,bbox_inches='tight')
                    plt.close()

    def __str__(self):
        print self.stats

class Params:
    '''
    Params for coco evaluation api
    '''
    def setKpParams(self):

        self.imgIds = []
        self.catIds = []

        self.kpts_name = \
            [u'nose',
             u'left_eye', u'right_eye',
             u'left_ear', u'right_ear',
             u'left_shoulder', u'right_shoulder',
             u'left_elbow', u'right_elbow',
             u'left_wrist', u'right_wrist',
             u'left_hip', u'right_hip',
             u'left_knee', u'right_knee',
             u'left_ankle', u'right_ankle']
        self.inv_kpts_name = \
            [u'nose',
             u'right_eye', u'left_eye',
             u'right_ear', u'left_ear',
             u'right_shoulder', u'left_shoulder',
             u'right_elbow', u'left_elbow',
             u'right_wrist', u'left_wrist',
             u'right_hip', u'left_hip',
             u'right_knee', u'left_knee',
             u'right_ankle', u'left_ankle']
        self.num_kpts = len(self.kpts_name)
        self.inv_idx      = [  self.inv_kpts_name.index(self.kpts_name[i]) for i in xrange(self.num_kpts)]

        self.sigmas = np.array([.026,.025,.025,
                                     .035,.035,
                                     .079,.079,
                                     .072,.072,
                                     .062,.062,
                                     .107,.107,
                                     .087,.087,
                                     .089,.089])

        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.oksThrs = np.array([ 0.5 ,  0.55,  0.6 ,  0.65,  0.7 ,  0.75,  0.8 ,  0.85,  0.9 ,  0.95])
        # the threshold that determines the limit for localization error
        self.oksLocThrs = .1
        # oks thresholds that define a jitter error
        self.jitterKsThrs = [.5,.85]

        self.maxDets     = [20]
        self.teamMaxDets = []

        self.areaRng    = [[32 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all']

        self.err_types = ['miss','swap','inversion','jitter']
        self.check_kpts   = True
        self.check_scores = True
        self.check_bkgd  = True

    def __init__(self, iouType='keypoints'):
        if iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType *%s* not supported'%iouType)
        self.iouType = iouType
