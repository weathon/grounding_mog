import os
import pandas as pd
import argparse
root_path = "../sim"
videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(os.path.join(root_path, "in"))]))

for morph in [0, 10, 20, 30, 40, 50]:
    for video_id in videos:
        print("Running for video_id: ", video_id)    
        os.system("python3 bgs_baseline.py --video_id {} --root_path {} --log_file results_bgs_only.csv --bgs_method knn --morph {}".format(video_id, root_path, morph))
        os.system("python3 bgs_baseline.py --video_id {} --root_path {} --log_file results_bgs_only.csv --bgs_method mog --morph {}".format(video_id, root_path, morph))
        os.system("python3 bgs_baseline.py --video_id {} --root_path {} --log_file results_bgs_only.csv --bgs_method median --morph {}".format(video_id, root_path, morph))