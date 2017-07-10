## imports
import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from scipy.misc import imresize
import skimage.io as io
import utilities

def backgroundFalseNegErrors( coco_analyze, imgs_info, saveDir ):
    loc_dir = saveDir + '/background_errors/false_negatives'
    if not os.path.exists(loc_dir):
        os.makedirs(loc_dir)
    f = open('%s/std_out.txt'%loc_dir, 'w')
    f.write("Running Analysis: [False Negatives]\n\n")
    tic = time.time()
    paths = {}

    oksThrs     = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
    areaRngs    = [[32**2,1e5**2]]
    areaRngLbls = ['all']

    coco_analyze.params.areaRng    = areaRngs
    coco_analyze.params.areaRngLbl = areaRngLbls
    coco_analyze.params.oksThrs    = oksThrs
    coco_analyze.cocoEval.params.useGtIgnore = 0
    coco_analyze.cocoEval.params.gtIgnoreIds = []
    coco_analyze.analyze(check_kpts=False, check_scores=False, check_bckgd=True)

    badFalseNeg = coco_analyze.false_neg_gts['all',str(.5)]
    for tind, t in enumerate(coco_analyze.params.oksThrs):
        badFalseNeg = badFalseNeg & coco_analyze.false_neg_gts['all',str(t)]
    # bad false negatives are those that are false negatives at all oks thresholds
    fn_gts = [coco_analyze.cocoGt.loadAnns(b)[0] for b in badFalseNeg]

    f.write("Num. annotations: [%d]\n"%len(coco_analyze.cocoGt.loadAnns(coco_analyze.cocoGt.getAnnIds())))
    for oks in oksThrs:
        f.write("OKS thresh: [%f]\n"%oks)
        f.write(" - Matches:      [%d]\n"%len(coco_analyze.bckgd_err_matches[areaRngLbls[0], str(oks), 'gts']))
        f.write(" - Bckgd. FN:    [%d]\n"%len(coco_analyze.false_neg_gts[areaRngLbls[0],str(oks)]))

    sorted_fns = sorted(fn_gts, key=lambda k: -k['num_keypoints'])
    sorted_fns = [fff for fff in sorted_fns if fff['num_keypoints']>0]
    show_fn = sorted_fns[0:4] + sorted_fns[-4:]
    f.write("\nBackground False Negative Errors:\n")
    for tind, t in enumerate(show_fn):
        name    = 'bckd_false_neg_%d'%tind
        paths[name] = "%s/%s.pdf"%(loc_dir,name)
        f.write("Image_id, ground_truth id, num_keypoints: [%d][%d][%d]\n"%(t['image_id'],t['id'],t['num_keypoints']))
        utilities.show_dets([],[t],imgs_info[t['image_id']],paths[name])

    max_height = max([d['bbox'][3] for d in fn_gts])
    min_height = min([d['bbox'][3] for d in fn_gts])
    max_width = max([d['bbox'][2] for d in fn_gts])
    min_width = min([d['bbox'][2] for d in fn_gts])

    f.write("\nBackground False Negatives Bounding Box Dimenstions:\n")
    f.write(" - Min width:  [%d]\n"%min_width)
    f.write(" - Max width:  [%d]\n"%max_width)
    f.write(" - Min height: [%d]\n"%min_height)
    f.write(" - Max height: [%d]\n"%max_height)

    ar_pic = np.zeros((int(max_height)+1,int(max_width)+1))
    ar_pic_2 = np.zeros((30,30))
    ar_bins = range(10)+range(10,100,10)+range(100,1000,100)+[1000]
    ar_pic_3 = np.zeros((10,10))
    ar_bins_3 = [np.power(2,x) for x in xrange(11)]

    num_fn_keypoints = {}

    areaRngs    = [[0, 32 ** 2],[32 ** 2, 64 ** 2],[64 ** 2, 96 ** 2],[96 ** 2, 128 ** 2],[128 ** 2, 1e5 ** 2]]
    areaRngLbls = ['small','medium','large','xlarge','xxlarge']
    small = 0; medium = 0; large = 0; xlarge = 0; xxlarge = 0

    num_people_ranges = [[0,0],[1,1],[2,4],[5,8],[9,100]]
    num_people_labels = ['none','one','small grp.','large grp.', 'crowd']
    no_people = 0; one = 0; small_grp = 0; large_grp = 0; crowd = 0

    segm_heatmap = np.zeros((128,128))
    for i,b in enumerate(fn_gts):
        if b['num_keypoints'] in num_fn_keypoints:
            num_fn_keypoints[b['num_keypoints']] += 1
        else:
            num_fn_keypoints[b['num_keypoints']] = 1

        if b['num_keypoints']==0: continue

        b_width  = int(b['bbox'][2])
        b_height = int(b['bbox'][3])
        ar_pic[0:b_height,0:b_width] += 1
        if b_width < 1024 and b_height < 1024:
            col = [i for i in xrange(len(ar_bins)-1) if ar_bins[i]<b_width<ar_bins[i+1]]
            row = [i for i in xrange(len(ar_bins)-1) if ar_bins[i]<b_height<ar_bins[i+1]]
            ar_pic_2[row,col] += 1

            col = [i for i in xrange(len(ar_bins_3)-1) if ar_bins_3[i]<b_width<ar_bins_3[i+1]]
            row = [i for i in xrange(len(ar_bins_3)-1) if ar_bins_3[i]<b_height<ar_bins_3[i+1]]
            ar_pic_3[row,col] += 1
        else:
            print "False Positive bbox has a side larger than 1024 pixels."
            print "Change lists ar_bins_2 and ar_bins_3 to include larger bins."
            assert(False)

        area = b_width * b_height * .5
        if areaRngs[0][0] <= area < areaRngs[0][1]:
            small += 1
        elif areaRngs[1][0] <= area < areaRngs[1][1]:
            medium += 1
        elif areaRngs[2][0] <= area < areaRngs[2][1]:
            large += 1
        elif areaRngs[3][0] <= area < areaRngs[3][1]:
            xlarge += 1
        elif areaRngs[4][0] <= area < areaRngs[4][1]:
            xxlarge += 1

        anns = coco_analyze.cocoGt.loadAnns(coco_analyze.cocoGt.getAnnIds(b['image_id']))
        iscrowd = [ann['iscrowd'] for ann in anns]
        num_people = len(anns) if sum(iscrowd)==0 else 100
        if num_people_ranges[0][0] <= num_people <= num_people_ranges[0][1]:
            no_people += 1
        elif num_people_ranges[1][0] <= num_people <= num_people_ranges[1][1]:
            one += 1
        elif num_people_ranges[2][0] <= num_people <= num_people_ranges[2][1]:
            small_grp += 1
        elif num_people_ranges[3][0] <= num_people <= num_people_ranges[3][1]:
            large_grp += 1
        elif num_people_ranges[4][0] <= num_people <= num_people_ranges[4][1]:
            crowd += 1

        if b['iscrowd']==1: continue
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
    path = "%s/bckd_false_neg_heatmaps.pdf"%(loc_dir)
    paths['false_neg_hm'] = path
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    f.write("\nNumber of people in images with Background False Negatives:\n")
    f.write(" - No people:         [%d]\n"%no_people)
    f.write(" - One person:        [%d]\n"%one)
    f.write(" - Small group (2-4): [%d]\n"%small_grp)
    f.write(" - Large Group (5-8): [%d]\n"%large_grp)
    f.write(" - Crowd       (>=9): [%d]\n"%crowd)

    f.write("\nArea size (in pixels) of Background False Negatives:\n")
    f.write(" - Small    (%d,%d):            [%d]\n"%(areaRngs[0][0],areaRngs[0][1],small))
    f.write(" - Medium   (%d,%d):         [%d]\n"%(areaRngs[1][0],areaRngs[1][1],medium))
    f.write(" - Large    (%d,%d):         [%d]\n"%(areaRngs[2][0],areaRngs[2][1],large))
    f.write(" - X-Large  (%d,%d):        [%d]\n"%(areaRngs[3][0],areaRngs[3][1],xlarge))
    f.write(" - XX-Large (%d,%d): [%d]\n"%(areaRngs[4][0],areaRngs[4][1],xxlarge))

    f.write("\nNumber of visible keypoints for Background False Negatives:\n")
    for k in num_fn_keypoints.keys():
        if k == 0: continue
        f.write(" - [%d] kpts: [%d] False Neg.\n"%(k,num_fn_keypoints[k]))

    plt.figure(figsize=(10,10))
    plt.imshow(ar_pic,origin='lower')
    plt.colorbar()
    plt.title('BBox Aspect Ratio',fontsize=20)
    plt.xlabel('Width (px)',fontsize=20)
    plt.ylabel('Height (px)',fontsize=20)
    path = "%s/bckd_false_neg_bbox_aspect_ratio.pdf"%(loc_dir)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(ar_pic_2,origin='lower')
    plt.xticks(xrange(1,len(ar_bins)+1),["%d"%(x) for x in ar_bins],rotation='vertical')
    plt.yticks(xrange(1,len(ar_bins)+1),["%d"%(x) for x in ar_bins])
    plt.colorbar()
    plt.grid()
    plt.title('BBox Aspect Ratio',fontsize=20)
    plt.xlabel('Width (px)',fontsize=20)
    plt.ylabel('Height (px)',fontsize=20)
    path = "%s/bckd_false_neg_bbox_aspect_ratio_2.pdf"%(loc_dir)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.imshow(ar_pic_3,origin='lower')
    plt.xticks([-.5 + x for x in range(11)],["%d"%(x) for x in ar_bins_3])
    plt.yticks([-.5 + x for x in range(11)],["%d"%(x) for x in ar_bins_3])
    plt.colorbar()
    plt.grid()
    plt.title('BBox Aspect Ratio',fontsize=20)
    plt.xlabel('Width (px)',fontsize=20)
    plt.ylabel('Height (px)',fontsize=20)
    path = "%s/bckd_false_neg_bbox_aspect_ratio_3.pdf"%(loc_dir)
    paths['false_neg_bbox_ar'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('lightgray')
    plt.bar(xrange(5),[small,medium,large,xlarge,xxlarge],color='g',align='center')
    plt.xticks(xrange(5),areaRngLbls)
    plt.grid()
    plt.title('Histogram of Area Size',fontsize=20)
    path = "%s/bckd_false_neg_bbox_area_hist.pdf"%(loc_dir)
    paths['false_neg_bbox_area_hist'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_bgcolor('lightgray')
    plt.bar(xrange(5),[no_people,one,small_grp,large_grp,crowd],color='g',align='center')
    plt.xticks(xrange(5),num_people_labels)
    plt.grid()
    plt.title('Histogram of Num. of People in Images',fontsize=20)
    path = "%s/bckd_false_neg_num_people_histogram.pdf"%(loc_dir)
    paths['false_neg_num_ppl_hist'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.bar([k for k in num_fn_keypoints.keys() if k!=0],[num_fn_keypoints[k] for k in num_fn_keypoints.keys() if k!= 0],align='center')
    plt.title("Histogram of Number of Keypoints",fontsize=20)
    path = "%s/bckd_false_neg_num_keypoints_histogram.pdf"%(loc_dir)
    paths['false_neg_num_kpts_hist'] = path
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    f.write("\nDone, (t=%.2fs)."%(time.time()-tic))
    f.close()

    return paths
