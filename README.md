# coco-analyze

Contains:
 - The COCOanalyze class, a wrapper of the COCOeval class.
 - run_analysis.py, a script for extended keypoint error estimation analysis using COCOanalyze.
 
Notes:
 - COCOeval class contained in pycocotools directory is a modified version of the mscoco repository class.
 - COCOanalyze class is newly written.
 - run_analysis.py will reproduce the plots from the paper "Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation", for a different analysis use the variables returned by COCOanalyze for your specific needs.

To run the extended keypoint error estimation analysis:
 - [annFile]  -> ./annotations/keypoints_val2014.json
 - [resFile]  -> ./results/fakekeypoints100_keypoints_val2014_results.json
 - [saveDir]  -> ./analysis
 - [teamName] -> fakekeypoints100
 - $ python run_analysis.py [annFile] [resFile] [saveDir] [teamName]
