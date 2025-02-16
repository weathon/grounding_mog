import os
import pandas as pd
root_path = "../sim"
videos = list(set(["_".join(i.split("_")[:-1]) for i in os.listdir(os.path.join(root_path, "in"))]))

past_log = pd.read_csv("results.csv")
past_videos = past_log["video_id"].unique().tolist()
# print(videos)
for video_id in videos:
    if video_id in past_videos:
        print("Skipping video_id: ", video_id)
        continue
    display = ""
    print("Running for video_id: ", video_id)
    os.system("python3 owl_notracking.py --video_id {} --root_path {} --display '{}' --log_file results.csv".format(video_id, root_path, display))