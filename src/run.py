import os
import pandas as pd
import argparse
root_path = "../sim"
videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(os.path.join(root_path, "in"))]))
videos.sort()
args = argparse.ArgumentParser()
args.add_argument("--skip_past", action="store_true")
args.add_argument("--method", type=str, default="owl")
args = args.parse_args()



if args.skip_past:
    past_log = pd.read_csv("results.csv")
    past_videos = past_log["video_id"].unique().tolist()
    print(videos)
for video_id in videos:
    if args.skip_past:
        if video_id in past_videos:
            print("Skipping video_id: ", video_id)
            continue
    display = ""
    print("Running for video_id: ", video_id)
    
    if args.method == "owl":
        os.system("python3 owl_notracking.py --video_id {} --root_path {} --log_file results.csv".format(video_id, root_path))
    elif args.method == "mog":
        os.system("python3 mog_baseline.py --video_id {} --root_path {} --log_file results_mog.csv".format(video_id, root_path))
    elif args.method == "mog_morph":
        os.system("python3 mog_baseline.py --video_id {} --root_path {} --log_file results_mog_morph.csv --morph 10".format(video_id, root_path))