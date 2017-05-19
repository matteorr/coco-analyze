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

        # dt-gt matches for all types of errors
        self.localization_matches = {}
        self.ori_score_matches    = {}
        self.opt_score_matches    = {}
        self.bckgd_err_matches    = {}

        # dt with corrections
        self.corrected_dts = []
        # false positive dts
        self.false_pos_dts = {}
        # ground truths with info about false negatives
        self.false_neg_gts = {}

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

    def analyze(self, check_kpts=True, check_scores=True, check_bckgd=True):
        if self.corrected_dts:
            # reset dts to the original dts so the same study can be repeated
            for d in self._dts:
                d['keypoints'] = self._original_dts[d['id']]['keypoints']
                d['score']     = self._original_dts[d['id']]['score']
        self.corrected_dts = copy.deepcopy(self._dts)
        self.false_neg_gts = {}

        self.params.check_kpts = check_kpts
        if check_kpts:
            # find keypoint errors in detections that are matched to ground truths
            self.correct_keypoints()

        self.params.check_scores = check_scores
        if check_scores:
            # find scoring errors in all detections
            self.correct_scores()

        # false positive and false negatives are dealt with in summarize()
        self.params.check_bckgd = check_bckgd
        if check_bckgd:
            # find background false positive errors and false negatives
            self._find_bckgd_errors()

        # the above analysis changed keypoints and scores in the cocoEval _dts
        # object so before moving on we restore them to their original value.
        for cdt in self.corrected_dts:
            dtid     = cdt['id']
            image_id = cdt['image_id']

            # loop through all detections in the image and change only the
            # corresponsing detection cdt being analyzed
            for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                if d['id'] == dtid:
                    if check_kpts:
                        d['keypoints'] = cdt['keypoints']
                    if check_scores:
                        d['score'] = cdt['score']
                    break

    def correct_keypoints(self):
        tic = time.time()
        print('<{}:{}>Analyzing keypoint errors...'.format(__author__,__version__))

        # find all matches between dts and gts at the lowest iou thresh
        # allowed for localization. Matches with lower oks are not valid
        dtMatches, gtMatches = self._find_dt_matches([self.params.oksLocThrs])
        self.localization_matches['dts'] = dtMatches[self.params.oksLocThrs]
        self.localization_matches['gts'] = gtMatches[self.params.oksLocThrs]

        # find which errors affect the oks of detections that are matched
        corrected_dts = self._find_kpt_errors()

        # save the optimal keypoints in a temporary dictionary and
        # change the keypoints to the optimized versions
        corrected_dts_dict = {}
        for cdt in corrected_dts:
            corrected_dts_dict[cdt['id']] = cdt
        assert(len(corrected_dts) == len(corrected_dts_dict))

        for cdt in self.corrected_dts:
            if cdt['id'] in corrected_dts_dict:
                cdt['opt_keypoints'] = corrected_dts_dict[cdt['id']]['keypoints']
                cdt['inversion']     = corrected_dts_dict[cdt['id']]['inversion']
                cdt['good']          = corrected_dts_dict[cdt['id']]['good']
                cdt['jitter']        = corrected_dts_dict[cdt['id']]['jitter']
                cdt['miss']          = corrected_dts_dict[cdt['id']]['miss']
                cdt['swap']          = corrected_dts_dict[cdt['id']]['swap']

                # change the detections in the cocoEval object to the corrected kpts
                dtid     = cdt['id']
                image_id = cdt['image_id']

                # loop through all detections in the image and change only the
                # corresponsing detection cdt being analyzed
                for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                    if d['id'] == dtid:
                        err_kpts_mask = np.zeros(len(cdt['good']))
                        if 'miss' in self.params.err_types:
                            err_kpts_mask += np.array(cdt['miss'])

                        if 'swap' in self.params.err_types:
                            err_kpts_mask += np.array(cdt['swap'])

                        if 'inversion' in self.params.err_types:
                            err_kpts_mask += np.array(cdt['inversion'])

                        if 'jitter' in self.params.err_types:
                            err_kpts_mask += np.array(cdt['jitter'])

                        d['keypoints'] = \
                            cdt['opt_keypoints'] * (np.repeat(err_kpts_mask,3)==1) + \
                            cdt['keypoints']     * (np.repeat(err_kpts_mask,3)==0)
                            #cdt['keypoints'] * (np.repeat(np.array(cdt['good']),3)==1)
                        break

        toc = time.time()
        print('<{}:{}>DONE (t={:0.2f}s).'.format(__author__,__version__,toc-tic))

    def correct_scores(self):
        tic = time.time()
        print('<{}:{}>Analyzing detection scores...'.format(__author__,__version__))

        # find matches before changing the scores
        dtMatches, gtMatches = self._find_dt_matches([self.params.oksLocThrs])
        self.ori_score_matches['dts'] = dtMatches[self.params.oksLocThrs]
        self.ori_score_matches['gts'] = gtMatches[self.params.oksLocThrs]

        # run the evaluation with no limit on max number of detections
        # note that for optimal score the oks thresh doesnt matter
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.iouThrs    = [min(self.params.oksThrs)]
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

        for cdt in self.corrected_dts:
            cdt['opt_score'] = opt_scores[cdt['id']]

            # change the detections in the cocoEval object to the corrected score
            dtid     = cdt['id']
            image_id = cdt['image_id']

            # loop through all detections in the image and change only the
            # corresponsing detection cdt being analyzed
            for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                if d['id'] == dtid:
                    d['score'] = cdt['opt_score']
                    break

        dtMatches, gtMatches = self._find_dt_matches([self.params.oksLocThrs])
        self.opt_score_matches['dts'] = dtMatches[self.params.oksLocThrs]
        self.opt_score_matches['gts'] = gtMatches[self.params.oksLocThrs]

        toc = time.time()
        print('<{}:{}>DONE (t={:0.2f}s).'.format(__author__,__version__,toc-tic))

    def _find_dt_matches(self, oksThrs):
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = oksThrs
        self.cocoEval.evaluate()

        evalImgs = [e for e in filter(None,self.cocoEval.evalImgs)]
        # evalImgs = [e for e in filter(None,self.cocoEval.evalImgs) if
        #             e['aRng']==self.params.areaRng]

        dtMatches = {}
        gtMatches = {}
        for oind, oks in enumerate(oksThrs):
            dtMatchesOKS = {}
            gtMatchesOKS = {}
            for i, e in enumerate(evalImgs):
                # add all matches to the dtMatches dictionary
                for dind, did in enumerate(e['dtIds']):
                    gtMatch = int(e['dtMatches'][oind][dind])

                    if gtMatch != 0:
                        # check that a detection is not already matched
                        assert(did not in dtMatchesOKS)
                        dtMatchesOKS[did] = [{'gtId'   :gtMatch,
                                              'dtId'    :did,
                                              'oks'     :e['dtIous'][oind][dind],
                                              'score'   :e['dtScores'][dind],
                                              'ignore'  :int(e['dtIgnore'][oind][dind]),
                                              'image_id':e['image_id']}]
                        # add the gt match as well since multiple dts can have same gt
                        entry = {'dtId'    :did,
                                 'gtId'    :gtMatch,
                                 'oks'     :e['dtIous'][oind][dind],
                                 'ignore'  :int(e['dtIgnore'][oind][dind]),
                                 'image_id':e['image_id']}
                        gtMatchesOKS.setdefault(gtMatch, []).append(entry)

                # add matches to the gtMatches dictionary
                for gind, gid in enumerate(e['gtIds']):
                    dtMatch = int(e['gtMatches'][oind][gind])

                    if dtMatch != 0:
                        entry = {'dtId'    :dtMatch,
                                 'gtId'    :gid,
                                 'oks'     :e['gtIous'][oind][gind],
                                 'ignore'  :int(e['gtIgnore'][gind]),
                                 'image_id':e['image_id']}
                        if gid in gtMatchesOKS:
                            if entry not in gtMatchesOKS[gid]:
                                gtMatchesOKS[gid].append(entry)
                        else:
                            gtMatchesOKS[gid] = [entry]
            dtMatches[oks] = dtMatchesOKS
            gtMatches[oks] = gtMatchesOKS

        return dtMatches, gtMatches

    def _find_kpt_errors(self):
        zero_kpt_gts  = 0
        corrected_dts = []

        # this contains all the detections that have been matched with a gt
        for did in self.localization_matches['dts']:
            # get the info on the [dt,gt] match
            # load the detection and ground truth annotations
            dtm        = self.localization_matches['dts'][did][0]
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

            # if the gt match has no keypoint annotations the analysis
            # cannot be carried out.
            if gt['num_keypoints'] == 0:
                zero_kpt_gts += 1
                continue

            # for every detection match return a dictionary with the following info:
            #  - image_id
            #  - detection_id
            #  - 'corrected_keypoints': list containing good value for each keypoint
            #  - 'jitt': binary list identifying jitter errors
            #  - 'inv':  binary list identifying inversion errors
            #  - 'miss': binary list identifying miss errors
            #  - 'swap': binary list identifying swap errors

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
        return corrected_dts

    def _find_bckgd_errors(self):
        tic = time.time()
        print('<{}:{}>Analyzing background false positives and false negatives...'.format(__author__,__version__))
        oksThrs    = sorted(self.params.oksThrs)

        # compute matches with current value of detections to determine new matches
        dtMatches, gtMatches = self._find_dt_matches(oksThrs)
        self.bckgd_err_matches['dts'] = dtMatches
        self.bckgd_err_matches['gts'] = gtMatches

        for oind, oks in enumerate(oksThrs):
            dtMatchesOKS = dtMatches[oks]
            gtMatchesOKS = gtMatches[oks]

            # assert that detection and ground truth matches are consistent
            for d in dtMatchesOKS:
                # assert that every detection matched has a corresponding gt in the gt matches dictionary
                assert(dtMatchesOKS[d][0]['gtId'] in gtMatchesOKS)
                # assert that this detection is in the dt matches of the gt it is matched to
                assert(d in [dt['dtId'] for dt in gtMatchesOKS[dtMatchesOKS[d][0]['gtId']]])

            # assert that all ground truth with multiple detection matches should be ignored
            count = 0
            for g in gtMatchesOKS:
                count+= len(gtMatchesOKS[g])
                if len(gtMatchesOKS[g])>1:
                    # if this gt already has multiple matches assert it is a crowd
                    # since crowd gt can be matched to multiple detections
                    assert(self.cocoGt.anns[g]['iscrowd']==1)
                assert(gtMatchesOKS[g][0]['dtId'] in dtMatchesOKS)
            assert(count==len(dtMatchesOKS))

            # store info about false positives in corrected detections
            false_pos = set([dt['id'] for dt in self._dts if dt['id'] not in dtMatchesOKS])
            self.false_pos_dts[oks] = set()
            for cdt in self.corrected_dts:
                if cdt['id'] in false_pos: self.false_pos_dts[oks].add(cdt['id'])

            # store info about false negatives in corrected ground truths
            false_neg = set([gt for gt in self.cocoGt.getAnnIds() if gt not in gtMatchesOKS])
            self.false_neg_gts[oks] = set()
            for gt in self._gts:
                if gt['id'] in false_neg: self.false_neg_gts[oks].add(gt['id'])

        toc = time.time()
        print('<{}:{}>DONE (t={:0.2f}s).'.format(__author__,__version__,toc-tic))

    def summarize(self, makeplots=False, savedir=None, team_name=None):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        if not self.corrected_dts:
            raise Exception('<{}:{}>Please run analyze() first'.format(__author__,__version__))

        # clear the stats for the summarize results
        self.stats = []
        oksThrs    = sorted(self.params.oksThrs)[::-1]
        areaRngLbl = self.params.areaRngLbl
        maxDets    = sorted(self.params.maxDets)

        # compute all the precision recall curves and return precise breakdown of
        # all error type in terms of keypoint, scoring, false positives and negatives
        ps_mat, rs_mat = self._summarize_baseline()
        err_types = ['baseline']
        stats = self._summarize(err_types, ps_mat,
                                           rs_mat,
                                           oksThrs, areaRngLbl, maxDets)
        self.stats.extend(stats)

        # summarize keypoint errors
        if self.params.check_kpts and self.params.err_types:
            ps_mat_kpt_errors, rs_mat_kpt_errors = self._summarize_kpt_errors()
            ps_mat = np.append(ps_mat,ps_mat_kpt_errors,axis=0)
            err_types = self.params.err_types
            stats = self._summarize(err_types, ps_mat_kpt_errors,
                                               rs_mat_kpt_errors,
                                               oksThrs, areaRngLbl, maxDets)
            self.stats.extend(stats)

        # summarize scoring errors
        if self.params.check_scores:
            ps_mat_score_errors, rs_mat_score_errors = self._summarize_score_errors()
            ps_mat = np.append(ps_mat,ps_mat_score_errors,axis=0)
            err_types = ['score']
            stats = self._summarize(err_types, ps_mat_score_errors,
                                               rs_mat_score_errors,
                                               oksThrs, areaRngLbl, maxDets)
            self.stats.extend(stats)

        # summarize detections that are unmatched (hallucinated false positives)
        # and ground truths that are unmatched (false negatives)
        if self.params.check_bckgd:
            ps_mat_bckgd_errors, rs_mat_bckgd_errors = self._summarize_bckgd_errors()
            ps_mat = np.append(ps_mat,ps_mat_bckgd_errors,axis=0)
            err_types = ['bckg_false_pos','false_neg']
            stats = self._summarize(err_types, ps_mat_bckgd_errors,
                                               rs_mat_bckgd_errors,
                                               oksThrs, areaRngLbl, maxDets)
            self.stats.extend(stats)

        print ps_mat.shape
        for s in self.stats: print s

        assert(False)

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

        if self.params.check_scores:
            err_labels += ['Score']
            colors_vec += ['#4F82BD']

        if self.params.check_bckgd:
            err_labels += ['Bkg.',   'FN']
            colors_vec += ['#8063A3','seagreen']

        if makeplots:
            self._plot(recalls, ps_mat, params, err_labels, colors_vec, savedir, team_name)

    def _summarize_baseline(self):
        '''
        Run the evaluation on the original detections to get the baseline for
        algorithm performance that will be compared to corrected detections.
        '''
        # set area range and the oks thresholds
        oksThrs    = sorted(self.params.oksThrs)
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = oksThrs

        # evaluate
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()

        # insert results into the precision matrix
        ps = self.cocoEval.eval['precision'][::-1,:,:,:,:]
        rs = self.cocoEval.eval['recall'][::-1,:,:,:]
        return ps, rs

    def _summarize_kpt_errors(self):
        '''
        Use the corrected detections to recompute performance of algorithm.
        Visualize results and plot precision recall curves if required by input variables.
        '''
        indx_list = [i for i in xrange(self.params.num_kpts*3) if (i-2)%3 != 0]

        # set the error_types
        err_types = self.params.err_types
        assert(len(err_types)>0)

        # set area range and the oks thresholds
        oksThrs    = sorted(self.params.oksThrs)
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = oksThrs

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
            # self.cocoEval.summarize(verbose=True)

            ps_err = self.cocoEval.eval['precision'][::-1,:,:,:,:]
            rs_err = self.cocoEval.eval['recall'][::-1,:,:,:]
            ps = ps_err if eind == 0 else np.append(ps, ps_err, axis=0)
            rs = rs_err if eind == 0 else np.append(rs, rs_err, axis=0)

        return ps, rs

    def _summarize_score_errors(self):
        '''
        Use the corrected detections to recompute performance of algorithm.
        Visualize results and plot precision recall curves if required by input variables.
        '''
        # set area range and the oks thresholds
        oksThrs    = sorted(self.params.oksThrs)
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = oksThrs

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

        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        # self.cocoEval.summarize(verbose=True)

        # insert results into the precision matrix
        ps  = self.cocoEval.eval['precision'][::-1,:,:,:,:]
        rs  = self.cocoEval.eval['recall'][::-1,:,:,:]
        return ps, rs

    def _summarize_bckgd_errors(self):
        '''
        Use the corrected detections to recompute performance of algorithm.
        Visualize results and plot precision recall curves if required by input variables.
        '''
        oksThrs    = sorted(self.params.oksThrs)
        # set area range and the oks thresh to last value of input array
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = oksThrs

        # evaluate performance, note this is with the keypoints corrected and scores
        # adjusted (or not) based on the value of the variables check_kpts, check_scores
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        # self.cocoEval.summarize(verbose=True)

        err = "bckgd. fp, fn"
        print('<{}:{}>Correcting error type [{}]:'.format(__author__,__version__,err))
        for oind, oks in enumerate(oksThrs):
            # set unmatched detections to ignore and remeasure performance
            #print "changed this to ignore all dts"
            for e in self.cocoEval.evalImgs:
                if e is None: continue

                for dind, dtid in enumerate(e['dtIds']):
                    # check if detection is a background false pos at this oks
                    #e['dtIgnore'][oind][dind] = True
                    if dtid in self.false_pos_dts[oks]:
                        e['dtIgnore'][oind][dind] = True

        # accumulate results after having set all this ignores
        self.cocoEval.accumulate()
        # self.cocoEval.summarize(verbose=True)

        ps_mat_false_pos = self.cocoEval.eval['precision'][::-1,:,:,:,:]
        rs_mat_false_pos = self.cocoEval.eval['recall'][::-1,:,:,:]

        T = len(oksThrs)
        R = len(self.cocoEval.params.recThrs)
        K = 1
        A = len(self.params.areaRng)
        M = len(self.params.maxDets)
        ps_mat_false_neg = np.zeros([T,R,K,A,M])
        rs_mat_false_neg = np.zeros([T,K,A,M])

        for oind, oks in enumerate(oksThrs):
            # False negatives at a lower oks are also a false negative
            # at a higher oks, so there is no need to reset the gtignore flag

            # set unmatched ground truths to ignore and remeasure performance
            for e in self.cocoEval.evalImgs:
                if e is None: continue
                for gind, gtid in enumerate(e['gtIds']):
                    if gtid in self.false_neg_gts[oks]:
                        e['gtIgnore'][gind] = 1

            # accumulate results after having set all this ignores
            self.cocoEval.accumulate()
            # self.cocoEval.summarize(verbose=True)

            ps_mat_false_neg[oind,:,:,:,:] = self.cocoEval.eval['precision'][oind,:,:,:,:]
            rs_mat_false_neg[oind,:,:,:]   = self.cocoEval.eval['recall'][oind,:,:,:]

        ps = np.append(ps_mat_false_pos, ps_mat_false_neg,axis=0)
        rs = np.append(rs_mat_false_pos, rs_mat_false_neg,axis=0)

        return ps, rs

    @staticmethod
    def _summarize(err_types, ps_mat, rs_mat, oksThrs, areaRngLbl, maxDets):
        stats = []
        l = len(oksThrs)

        for eind, err in enumerate(err_types):
            ps_mat_err_slice = ps_mat[eind*l:(eind+1)*l,:,:,:,:]
            rs_mat_err_slice = rs_mat[eind*l:(eind+1)*l,:,:,:]

            for oind, oks in enumerate(oksThrs):
                for aind, arearng in enumerate(areaRngLbl):
                    for mind, maxdts in enumerate(maxDets):
                        stat = {}
                        stat['oks']        = oks
                        stat['areaRngLbl'] = arearng
                        stat['maxDets']    = maxdts
                        stat['err']        = err

                        p = ps_mat_err_slice[oind,:,:,aind,mind]
                        r = rs_mat_err_slice[oind,:,aind,mind]

                        stat['auc']     = -1 if len(p[p>-1])==0 else np.mean(p[p>-1])
                        stat['recall']  = -1 if len(r[r>-1])==0 else np.mean(r[r>-1])
                        stats.append(stat)
        return stats

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
                    if len(precisions[precisions>-1])==0: m_map=.0
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

                    savepath = '{}/{}_[{}]{}[{}][{}].pdf'.format(savedir,prefix,team_name,oks_str,a,m)
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
        self.check_bckgd  = True

    def __init__(self, iouType='keypoints'):
        if iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType *%s* not supported'%iouType)
        self.iouType = iouType
