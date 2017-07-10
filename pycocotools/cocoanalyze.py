__author__  = 'mrr'
__version__ = '2.0'

import numpy as np; import time; import copy
from cocoeval import COCOeval
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.family']     = 'monospace'
from colour import Color

import skimage.io as io

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
        # dt with corrections
        self.corrected_dts = {}
        # false positive dts
        self.false_pos_dts = {}
        # ground truths with info about false negatives
        self.false_neg_gts = {}
        # dt-gt matches
        self.localization_matches = {}
        self.bckgd_err_matches    = {}
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
        self._cleanup()

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
            self._plot(recalls=recalls,ps_mat=ps_mat,params=self.params,
                       savedir=savedir,team_name=team_name)

    def analyze(self, check_kpts=True, check_scores=True, check_bckgd=True):
        if self.corrected_dts:
            # reset dts to the original dts so the same study can be repeated
            self._cleanup()

        for areaRngLbl in self.params.areaRngLbl:
            self.corrected_dts[areaRngLbl] = copy.deepcopy(self._dts)

        self.false_neg_gts = {}
        self.false_pos_dts = {}

        # find keypoint errors in detections that are matched to ground truths
        self.params.check_kpts = check_kpts
        if check_kpts: self.find_keypoint_errors()

        # find scoring errors in all detections
        self.params.check_scores = check_scores
        if check_scores: self.find_score_errors()

        # find background false positive errors and false negatives
        self.params.check_bckgd = check_bckgd
        if check_bckgd: self.find_bckgd_errors()

    def find_keypoint_errors(self):
        tic = time.time()
        print('Analyzing keypoint errors...')
        # find all matches between dts and gts at the lowest iou thresh
        # allowed for localization. Matches with lower oks are not valid
        oksLocThrs = [self.params.oksLocThrs]
        areaRng    = self.params.areaRng
        areaRngLbl = self.params.areaRngLbl
        dtMatches, gtMatches = self._find_dt_gt_matches(oksLocThrs, areaRng, areaRngLbl)

        for aind, arearnglbl in enumerate(areaRngLbl):
            self.localization_matches[arearnglbl, str(self.params.oksLocThrs), 'dts'] = \
                dtMatches[arearnglbl, str(self.params.oksLocThrs)]
            self.localization_matches[arearnglbl, str(self.params.oksLocThrs), 'gts'] = \
                gtMatches[arearnglbl, str(self.params.oksLocThrs)]

        # find which errors affect the oks of detections that are matched
        corrected_dts = self._find_kpt_errors()

        for areaRngLbl in self.params.areaRngLbl:
            corrected_dts_dict = {}

            for cdt in corrected_dts[areaRngLbl]:
                corrected_dts_dict[cdt['id']] = cdt
            assert(len(corrected_dts[areaRngLbl]) == len(corrected_dts_dict))

            for cdt in self.corrected_dts[areaRngLbl]:
                if cdt['id'] in corrected_dts_dict:
                    cdt['opt_keypoints'] = corrected_dts_dict[cdt['id']]['keypoints']
                    cdt['inversion']     = corrected_dts_dict[cdt['id']]['inversion']
                    cdt['good']          = corrected_dts_dict[cdt['id']]['good']
                    cdt['jitter']        = corrected_dts_dict[cdt['id']]['jitter']
                    cdt['miss']          = corrected_dts_dict[cdt['id']]['miss']
                    cdt['swap']          = corrected_dts_dict[cdt['id']]['swap']

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def _find_dt_gt_matches(self, oksThrs, areaRng, areaRngLbl):
        self.cocoEval.params.areaRng    = areaRng
        self.cocoEval.params.areaRngLbl = areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = oksThrs
        self.cocoEval.evaluate()

        dtMatches = {}
        gtMatches = {}

        for aind, arearnglbl in enumerate(areaRngLbl):
            #evalImgs = [e for e in filter(None,self.cocoEval.evalImgs)]
            evalImgs = [e for e in filter(None,self.cocoEval.evalImgs) if
                        e['aRng']==areaRng[aind]]

            for oind, oks in enumerate(oksThrs):
                dtMatchesAreaOks = {}
                gtMatchesAreaOks = {}

                for i, e in enumerate(evalImgs):
                    # add all matches to the dtMatches dictionary
                    for dind, did in enumerate(e['dtIds']):
                        gtMatch = int(e['dtMatches'][oind][dind])

                        if gtMatch != 0:
                            # check that a detection is not already matched
                            assert(did not in dtMatchesAreaOks)
                            dtMatchesAreaOks[did] = [{'gtId'    :gtMatch,
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
                            gtMatchesAreaOks.setdefault(gtMatch, []).append(entry)

                    # add matches to the gtMatches dictionary
                    for gind, gid in enumerate(e['gtIds']):
                        dtMatch = int(e['gtMatches'][oind][gind])

                        if dtMatch != 0:
                            entry = {'dtId'    :dtMatch,
                                     'gtId'    :gid,
                                     'oks'     :e['gtIous'][oind][gind],
                                     'ignore'  :int(e['gtIgnore'][gind]),
                                     'image_id':e['image_id']}
                            if gid in gtMatchesAreaOks:
                                if entry not in gtMatchesAreaOks[gid]:
                                    gtMatchesAreaOks[gid].append(entry)
                            else:
                                gtMatchesAreaOks[gid] = [entry]

                dtMatches[arearnglbl,str(oks)] = dtMatchesAreaOks
                gtMatches[arearnglbl,str(oks)] = gtMatchesAreaOks
        return dtMatches, gtMatches

    def _find_kpt_errors(self):
        zero_kpt_gts  = 0
        corrected_dts = {}

        oksLocThrs  = self.params.oksLocThrs
        areaRngLbls = self.params.areaRngLbl

        for aind, areaRngLbl in enumerate(areaRngLbls):
            localization_matches_dts = self.localization_matches[areaRngLbl, str(oksLocThrs), 'dts']
            corrected_dts[areaRngLbl] = []
            # this contains all the detections that have been matched with a gt
            for did in localization_matches_dts:
                # get the info on the [dt,gt] match
                # load the detection and ground truth annotations
                dtm        = localization_matches_dts[did][0]
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

                kpts_oks_mat = \
                  np.exp( -sqrd_dist / (self.params.sigmas*2)**2 / (areas[:,np.newaxis]+np.spacing(1)) / 2 ) * (vflags>0) +\
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

                corrected_dts[areaRngLbl].append(new_dt)
        return corrected_dts

    def _correct_dt_keypoints(self, areaRngLbl):
        # change the detections in the cocoEval object to the corrected kpts
        for cdt in self.corrected_dts[areaRngLbl]:
            if 'opt_keypoints' not in cdt: continue
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
                    break

    def find_score_errors(self):
        tic = time.time()
        print('Analyzing detection scores...')
        # NOTE: optimal score is measures at the lowest oks evaluation thresh
        self.cocoEval.params.iouThrs = [min(self.params.oksThrs)]
        self.cocoEval.params.maxDets = self.params.teamMaxDets

        # if the keypoint analyisis is required then the keypoints must be
        # corrected at all area range values requested before running scoring analysis
        if self.params.check_kpts:
            evalImgs = []
            for aind, areaRngLbl in enumerate(self.params.areaRngLbl):
                # restore original dts and gt ignore flags for new area range
                self._cleanup()
                self._correct_dt_keypoints(areaRngLbl)

                # run the evaluation with check scores flag
                self.cocoEval.params.areaRng    = [self.params.areaRng[aind]]
                self.cocoEval.params.areaRngLbl = [areaRngLbl]
                self.cocoEval.evaluate(check_scores=True)
                evalImgs.extend([e for e in filter(None,self.cocoEval.evalImgs)])
        else:
            # run the evaluation with check scores flag
            self.cocoEval.params.areaRng    = self.params.areaRng
            self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
            self.cocoEval.evaluate(check_scores=True)
            evalImgs = [e for e in filter(None,self.cocoEval.evalImgs)]

        for aind, areaRngLbl in enumerate(self.params.areaRngLbl):
            evalImgsArea = [e for e in filter(None,evalImgs) if
                            e['aRng']==self.params.areaRng[aind]]

            max_oks = {};
            for e in evalImgsArea:
                dtIds       = e['dtIds']
                dtScoresMax = e['dtIousMax']
                for i,j in zip(dtIds,dtScoresMax):
                    max_oks[i] = j
            # if assertion fails not all the detections have been evaluated
            assert(len(max_oks) == len(self._dts))

            # do soft non max suppression
            _soft_nms_dts = self._soft_nms(max_oks)
            for cdt in self.corrected_dts[areaRngLbl]:
                d = _soft_nms_dts[cdt['id']]
                cdt['opt_score'] = d['opt_score']
                cdt['max_oks']   = d['max_oks']

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def _soft_nms(self, max_oks):
        _soft_nms_dts = {}

        variances = (self.params.sigmas * 2)**2
        for imgId in self.params.imgIds:
            B = []; D = []; available = set()
            for d in self.cocoEval._dts[imgId, self.params.catIds[0]]:
                dt = {}; dt['keypoints'] = d['keypoints']
                dt['max_oks']   = max_oks[d['id']]
                dt['opt_score'] = max_oks[d['id']]
                _soft_nms_dts[d['id']] = dt
                B.append(dt)
            if len(B) == 0: continue

            while len(B) > 0:
                B.sort(key=lambda k: -k['opt_score'])
                M = B[0]
                D.append(M); B.remove(M)
                m_kpts =  np.array(M['keypoints'])
                m_xs = m_kpts[0::3]; m_ys = m_kpts[1::3]
                x0,x1,y0,y1 = np.min(m_xs), np.max(m_xs), np.min(m_ys), np.max(m_ys)
                m_area = (x1-x0)*(y1-y0)

                for dt in B:
                    d_kpts = np.array(dt['keypoints'])
                    d_xs = d_kpts[0::3]; d_ys = d_kpts[1::3]
                    x0,x1,y0,y1 = np.min(d_xs), np.max(d_xs), np.min(d_ys), np.max(d_ys)
                    d_area = (x1-x0)*(y1-y0)

                    deltax = d_xs - m_xs; deltay = d_ys - m_ys
                    # using the average of both areas as area for oks computation
                    e = (deltax**2 + deltay**2) / variances / ((.5*(m_area+d_area))+np.spacing(1)) / 2
                    oks = np.sum(np.exp(-e)) / e.shape[0]

                    old_score = dt['opt_score']
                    e = (oks ** 2) / .5 # .5 is a hyperparameter from soft_nms paper
                    new_score = old_score * np.exp(-e)
                    dt['opt_score'] = new_score
                    #print old_score, oks, new_score
        return _soft_nms_dts

    def _correct_dt_scores(self, areaRngLbl):
        # change the detections in the cocoEval object to the corrected score
        for cdt in self.corrected_dts[areaRngLbl]:
            dtid     = cdt['id']
            image_id = cdt['image_id']
            # loop through all detections in the image and change only the
            # corresponsing detection cdt being analyzed
            for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                if d['id'] == dtid:
                    d['score'] = cdt['opt_score']
                    break

    def find_bckgd_errors(self):
        tic = time.time()
        print('Analyzing background false positives and false negatives...')
        # compute matches with current value of detections to determine new matches
        oksThrs = sorted(self.params.oksThrs)

        for areaRng, areaRngLbl in zip(self.params.areaRng, self.params.areaRngLbl):
            self._cleanup()
            # correct keypoints and score if the analysis flags are True
            if self.params.check_kpts:   self._correct_dt_keypoints(areaRngLbl)
            if self.params.check_scores: self._correct_dt_scores(areaRngLbl)
            # get the matches at all oks thresholds for every area range
            dtMatches, gtMatches = self._find_dt_gt_matches(oksThrs, [areaRng], [areaRngLbl])

            for oind, oks in enumerate(oksThrs):
                dtMatchesAreaOks = dtMatches[areaRngLbl, str(oks)]
                gtMatchesAreaOks = gtMatches[areaRngLbl, str(oks)]

                self.bckgd_err_matches[areaRngLbl, str(oks), 'dts'] = dtMatches[areaRngLbl, str(oks)]
                self.bckgd_err_matches[areaRngLbl, str(oks), 'gts'] = gtMatches[areaRngLbl, str(oks)]

                # assert that detection and ground truth matches are consistent
                for d in dtMatchesAreaOks:
                    # assert that every detection matched has a corresponding gt in the gt matches dictionary
                    assert(dtMatchesAreaOks[d][0]['gtId'] in gtMatchesAreaOks)
                    # assert that this detection is in the dt matches of the gt it is matched to
                    assert(d in [dt['dtId'] for dt in gtMatchesAreaOks[dtMatchesAreaOks[d][0]['gtId']]])

                # assert that all ground truth with multiple detection matches should be ignored
                count = 0
                for g in gtMatchesAreaOks:
                    count+= len(gtMatchesAreaOks[g])
                    if len(gtMatchesAreaOks[g])>1:
                        # if this gt already has multiple matches assert it is a crowd
                        # since crowd gt can be matched to multiple detections
                        assert(self.cocoGt.anns[g]['iscrowd']==1)
                    assert(gtMatchesAreaOks[g][0]['dtId'] in dtMatchesAreaOks)
                assert(count==len(dtMatchesAreaOks))

                false_pos = set([dt['id'] for dt in self._dts if dt['id'] not in dtMatchesAreaOks])
                self.false_pos_dts[areaRngLbl, str(oks)] = set()
                for cdt in self.corrected_dts[areaRngLbl]:
                    if cdt['id'] in false_pos: self.false_pos_dts[areaRngLbl, str(oks)].add(cdt['id'])

                false_neg = set([gt for gt in self.cocoGt.getAnnIds() if gt not in gtMatchesAreaOks])
                self.false_neg_gts[areaRngLbl, str(oks)] = set()
                for gt in self._gts:
                    if gt['id'] in false_neg: self.false_neg_gts[areaRngLbl, str(oks)].add(gt['id'])

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def summarize(self, makeplots=False, savedir=None, team_name=None):
        '''
        Run the evaluation on the original detections to get the baseline for
        algorithm performance and after correcting the detections.
        '''
        if not self.corrected_dts:
            raise Exception('<{}:{}>Please run analyze() first'.format(__author__,__version__))

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
            err_types = ['bckgd_false_pos','false_neg']
            stats = self._summarize(err_types, ps_mat_bckgd_errors,
                                               rs_mat_bckgd_errors,
                                               oksThrs, areaRngLbl, maxDets)
            self.stats.extend(stats)

        err_labels = []
        colors_vec = []
        if self.params.check_kpts:
            for err in self.params.err_types:
                if err == 'miss':
                    err_labels.append('w/o Miss')
                    colors_vec.append('#F2E394')
                if err == 'swap':
                    err_labels.append('w/o Swap')
                    colors_vec.append('#F2AE72')
                if err == 'inversion':
                    err_labels.append('w/o Inv.')
                    colors_vec.append('#D96459')
                if err == 'jitter':
                    err_labels.append('w/o Jit.')
                    colors_vec.append('#8C4646')

        if self.params.check_scores:
            err_labels += ['Opt. Score']
            colors_vec += ['#4F82BD']

        if self.params.check_bckgd:
            err_labels += ['w/o Bkg. FP', 'w/o FN']
            colors_vec += ['#8063A3','seagreen']

        if makeplots:
            self._plot(self.cocoEval.params.recThrs[:], ps_mat,
                       self.params, err_labels, colors_vec, savedir, team_name)

    def _summarize_baseline(self):
        self._cleanup()
        # set area range and the oks thresholds
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.areaRng    = self.params.areaRng
        self.cocoEval.params.areaRngLbl = self.params.areaRngLbl
        self.cocoEval.params.maxDets    = self.params.maxDets
        self.cocoEval.params.iouThrs    = oksThrs
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        ps = self.cocoEval.eval['precision'][::-1,:,:,:,:]
        rs = self.cocoEval.eval['recall'][::-1,:,:,:]
        return ps, rs

    def _summarize_kpt_errors(self):
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        indx_list = [i for i in xrange(self.params.num_kpts*3) if (i-2)%3 != 0]
        err_types = self.params.err_types
        assert(len(err_types)>0)
        T = len(oksThrs); E = len(self.params.err_types)
        R = len(self.cocoEval.params.recThrs); K = 1
        A = len(self.params.areaRng); M = len(self.params.maxDets)
        ps_mat_kpts = np.zeros([T*E,R,K,A,M])
        rs_mat_kpts = np.zeros([T*E,K,A,M])

        for aind, arearnglbl in enumerate(self.params.areaRngLbl):
            print('Correcting area range [{}]:'.format(arearnglbl))
            self._cleanup()

            self.cocoEval.params.areaRng    = [self.params.areaRng[aind]]
            self.cocoEval.params.areaRngLbl = [arearnglbl]
            corrected_dts = self.corrected_dts[arearnglbl]
            # compute performance after solving for each error type
            for eind, err in enumerate(err_types):
                print('Correcting error type [{}]:'.format(err))
                tind_start = T * eind
                tind_end   = T * (eind+1)

                for cdt in corrected_dts:
                    # this detection doesn't have keypoint errors (wasn't matched)
                    if err not in cdt.keys():
                        continue
                    # check if detection has error of that type
                    if sum(cdt[err]) != 0:
                        dtid     = cdt['id']
                        image_id = cdt['image_id']
                        corrected_kpts = np.array(cdt['opt_keypoints'])
                        # correct only those keypoints
                        for d in self.cocoEval._dts[image_id, self.params.catIds[0]]:
                            if d['id'] == dtid:
                                oth_kpts_mask = np.repeat(np.logical_not(cdt[err])*1,2)
                                err_kpts_mask = np.repeat(cdt[err],2)
                                all_kpts = np.delete(np.array(d['keypoints']), slice(2, None, 3))
                                opt_kpts = np.delete(np.array(corrected_kpts), slice(2, None, 3))

                                kpts = all_kpts * oth_kpts_mask + \
                                       opt_kpts * err_kpts_mask

                                d['keypoints'] = np.array(d['keypoints'])
                                d['keypoints'][indx_list] = kpts
                                d['keypoints'] = d['keypoints'].tolist()
                                break
                self.cocoEval.evaluate()
                self.cocoEval.accumulate()

                ps_mat_kpts[tind_start:tind_end,:,:,aind,:] = self.cocoEval.eval['precision'][::-1,:,:,0,:]
                rs_mat_kpts[tind_start:tind_end,:,aind,:]   = self.cocoEval.eval['recall'][::-1,:,0,:]
        return ps_mat_kpts, rs_mat_kpts

    def _summarize_score_errors(self):
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        T = len(oksThrs); R = len(self.cocoEval.params.recThrs); K = 1
        A = len(self.params.areaRng); M = len(self.params.maxDets)
        ps_mat_score = np.zeros([T,R,K,A,M])
        rs_mat_score = np.zeros([T,K,A,M])

        for aind, arearnglbl in enumerate(self.params.areaRngLbl):
            print('Correcting area range [{}]:'.format(arearnglbl))
            print('Correcting error type [{}]:'.format("score"))
            self._cleanup()

            self.cocoEval.params.areaRng    = [self.params.areaRng[aind]]
            self.cocoEval.params.areaRngLbl = [arearnglbl]
            if self.params.check_kpts: self._correct_dt_keypoints(arearnglbl)
            self._correct_dt_scores(arearnglbl)

            self.cocoEval.evaluate()
            self.cocoEval.accumulate()

            # insert results into the precision matrix
            ps_mat_score[:,:,:,aind,:] = self.cocoEval.eval['precision'][::-1,:,:,0,:]
            rs_mat_score[:,:,aind,:]   = self.cocoEval.eval['recall'][::-1,:,0,:]
        return ps_mat_score, rs_mat_score

    def _summarize_bckgd_errors(self):
        oksThrs = sorted(self.params.oksThrs)
        self.cocoEval.params.maxDets = self.params.maxDets
        self.cocoEval.params.iouThrs = oksThrs
        T = len(oksThrs); R = len(self.cocoEval.params.recThrs); K = 1
        A = len(self.params.areaRng); M = len(self.params.maxDets)
        ps_mat_false_pos = np.zeros([T,R,K,A,M])
        rs_mat_false_pos = np.zeros([T,K,A,M])
        ps_mat_false_neg = np.zeros([T,R,K,A,M])
        rs_mat_false_neg = np.zeros([T,K,A,M])

        for aind, arearnglbl in enumerate(self.params.areaRngLbl):
            print('Correcting area range [{}]:'.format(arearnglbl))
            print('Correcting error type [{}]:'.format("bckgd. fp, fn"))
            self._cleanup()

            self.cocoEval.params.areaRng    = [self.params.areaRng[aind]]
            self.cocoEval.params.areaRngLbl = [arearnglbl]
            if self.params.check_kpts:   self._correct_dt_keypoints(arearnglbl)
            if self.params.check_scores: self._correct_dt_scores(arearnglbl)

            self.cocoEval.evaluate()
            self.cocoEval.accumulate()

            for oind, oks in enumerate(oksThrs):
                # set unmatched detections to ignore and remeasure performance
                for e in self.cocoEval.evalImgs:
                    if e is None: continue
                    for dind, dtid in enumerate(e['dtIds']):
                        # check if detection is a background false pos at this oks
                        if dtid in self.false_pos_dts[arearnglbl,str(oks)]:
                            e['dtIgnore'][oind][dind] = True
                # accumulate results after having set all this ignores
                self.cocoEval.accumulate()
            ps_mat_false_pos[:,:,:,aind,:] = self.cocoEval.eval['precision'][::-1,:,:,0,:]
            rs_mat_false_pos[:,:,aind,:]   = self.cocoEval.eval['recall'][::-1,:,0,:]

            # False negatives at a lower oks are also a false negative
            # at a higher oks, so there is no need to reset the gtignore flag
            for oind, oks in enumerate(oksThrs):
                # set unmatched ground truths to ignore and remeasure performance
                for e in self.cocoEval.evalImgs:
                    if e is None: continue
                    for gind, gtid in enumerate(e['gtIds']):
                        if gtid in self.false_neg_gts[arearnglbl,str(oks)]:
                            e['gtIgnore'][gind] = 1
                # accumulate results after having set all this ignores
                self.cocoEval.accumulate()
                ps_mat_false_neg[oind,:,:,aind,:] = self.cocoEval.eval['precision'][oind,:,:,0,:]
                rs_mat_false_neg[oind,:,aind,:]   = self.cocoEval.eval['recall'][oind,:,0,:]

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

    def _cleanup(self):
        # restore detections and gt ignores to their original value
        for d in self._dts:
            d['keypoints'] = self._original_dts[d['id']]['keypoints']
            d['score']     = self._original_dts[d['id']]['score']
        for g in self._gts:
            g['_ignore'] = 0

    @staticmethod
    def _plot(recalls, ps_mat, params, err_labels=[], color_vec=[], savedir=None, team_name=None):
        iouThrs    = params.oksThrs[::-1]
        areaRngLbl = params.areaRngLbl
        maxDets    = params.maxDets
        catId      = 0

        if err_labels:
            labels = ['Orig. Dts.'] + err_labels
            colors = list(Color("white").range_to(Color("seagreen"),len(labels)))
            colors[-len(err_labels):] = \
                [Color(c) for c in color_vec]
        else:
            labels = ['Oks %.2f'%o for o in iouThrs]
            colors = list(Color("white").range_to(Color("seagreen"),len(labels)))

        for aind, a in enumerate(areaRngLbl):
            for mind, m in enumerate(maxDets):
                if not err_labels:
                    fig=plt.figure(figsize=(10,8))
                    ax = fig.add_axes([0.1, 0.15, 0.56, 0.7])
                    plt.title('areaRng:[{}], maxDets:[{}]'.format(a,m),fontsize=18)
                    oks_ps_mat = ps_mat

                for tind, t in enumerate(iouThrs):
                    legend_patches = []
                    if err_labels:
                        fig=plt.figure(figsize=(10,8))
                        ax = fig.add_axes([0.1, 0.15, 0.56, 0.7])
                        plt.title('oksThrs:[{}], areaRng:[{}], maxDets:[{}]'.format(t,a,m),fontsize=18)
                        thresh_idx = [tind + i * len(iouThrs) for i in xrange(len(labels))]
                        oks_ps_mat = ps_mat[thresh_idx,:,:,:,:]

                    for lind, l in enumerate(labels):
                        precisions = oks_ps_mat[lind,:,catId,aind,mind]
                        plt.plot(recalls,precisions,c='k',ls='-',lw=2)

                        if lind > 0:
                            prev_precisions = oks_ps_mat[lind-1,:,catId,aind,mind]
                            plt.fill_between(recalls,
                                         prev_precisions, precisions,
                                         where=precisions >= prev_precisions,
                                         facecolor=colors[lind].rgb, interpolate=True)

                        m_map = np.mean(precisions[precisions>-1])
                        if len(precisions[precisions>-1])==0: m_map=.0
                        interm_m_map = '%.3f'%m_map
                        m_map_val_str = interm_m_map[1-int(interm_m_map[0]):5-int(interm_m_map[0])]

                        if err_labels:
                            the_label = '{:<11}: {}'.format(l,m_map_val_str)
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
                        prefix   = 'error_prc' if err_labels else 'prc'
                        oks_str  = '[%s]'%(int(100*t)) if err_labels else ''
                        savepath = '{}/{}_[{}]{}[{}][{}].pdf'.format(savedir,prefix,team_name,oks_str,a,m)
                        plt.savefig(savepath,bbox_inches='tight')
                        plt.close()

                    if not err_labels:
                        break

    def __str__(self):
        print self.stats

class Params:
    # Params for coco analyze api
    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        self.kpts_name     = \
            [u'nose',u'left_eye', u'right_eye',u'left_ear', u'right_ear',
             u'left_shoulder', u'right_shoulder', u'left_elbow', u'right_elbow',
             u'left_wrist', u'right_wrist', u'left_hip', u'right_hip',
             u'left_knee', u'right_knee', u'left_ankle', u'right_ankle']
        self.inv_kpts_name = \
            [u'nose', u'right_eye', u'left_eye', u'right_ear', u'left_ear',
             u'right_shoulder', u'left_shoulder', u'right_elbow', u'left_elbow',
             u'right_wrist', u'left_wrist', u'right_hip', u'left_hip',
             u'right_knee', u'left_knee', u'right_ankle', u'left_ankle']
        self.num_kpts = len(self.kpts_name)
        self.inv_idx  = [self.inv_kpts_name.index(self.kpts_name[i]) for i in xrange(self.num_kpts)]
        self.sigmas   = np.array([.026,.025,.025, .035,.035, .079,.079, .072,.072,
                                     .062,.062, .107,.107, .087,.087, .089,.089])
        self.oksThrs  = np.array([.5 ,.55, .6, .65, .7, .75, .8, .85, .9, .95])
        # the threshold that determines the limit for localization error
        self.oksLocThrs = .1
        # oks thresholds that define a jitter error
        self.jitterKsThrs = [.5,.85]
        self.maxDets      = [20]
        self.teamMaxDets  = []
        self.areaRng      = [[32 ** 2, 1e5 ** 2],[32 ** 2, 96 ** 2],[96 ** 2, 1e5 ** 2]]
        self.areaRngLbl   = ['all','medium','large']
        self.err_types    = ['miss','swap','inversion','jitter']
        self.check_kpts   = True; self.check_scores = True; self.check_bckgd  = True

    def __init__(self, iouType='keypoints'):
        if iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType *%s* not supported'%iouType)
        self.iouType = iouType
