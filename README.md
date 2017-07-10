# coco-analyze

Contains:
 - The COCOanalyze class, a wrapper of the COCOeval class for keypoint error estimation analysis.
 - analysisAPI, an API that uses COCOanalyze to generate
 - run_analysis.py, script generating a pdf summary of the extended analysis.

Notes:
 - COCOeval class contained in pycocotools directory is a modified version of the mscoco repository class. modified code prints stuff...
 - COCOanalyze class contained in pycocotools is a new class.
 - run_analysis.py will reproduce the plots contained in the paper "Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation".
 - Full analysis will take approximately () depending on number of detections your algorithm

To run the extended keypoint error estimation analysis:
 - [annFile]  -> ./annotations/keypoints_val2014.json
 - [resFile]  -> ./results/fakekeypoints100_keypoints_val2014_results.json
 - [saveDir]  -> ./analysis
 - [teamName] -> fakekeypoints100
 - $ python run_analysis.py [annFile] [resFile] [saveDir] [teamName]
