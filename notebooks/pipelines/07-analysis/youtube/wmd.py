import gensim.downloader as api
import glob
import sys
import os
import pandas as pd
import numpy as np

from tqdm import tqdm

sys.path.append("../../../../utils")
from absolute_path_builder import AbsolutePathBuilder

def main():
    input_path = AbsolutePathBuilder.get_path(
        f"04_youtube_scored",
        filepaths="../../../../config/filepaths.yaml"
    )

    df_data_desc = pd.read_csv(
        AbsolutePathBuilder.get_path(
            "00_youtube_data_description",
            filepaths="../../../../config/filepaths.yaml"
        )
    )

    df_data_desc["group"] = df_data_desc.group.apply(lambda s: s.split()[0])

    group_urls = df_data_desc[df_data_desc.group == "White"]    
    filenames = [row["url"].split("v=")[1].split("&")[0] for _, row in group_urls.iterrows()]

    dfs = []
    count = 0
    for file in tqdm(filenames):
        try:
            dfs.append(pd.read_csv(os.path.join(input_path, file)))
        except:
            count += 1

    df_white = pd.concat(dfs, ignore_index=True)
    print(f"There was {count} reading errors")

    group_urls = df_data_desc[df_data_desc.group == "Black"]    
    filenames = [row["url"].split("v=")[1].split("&")[0] for _, row in group_urls.iterrows()]

    dfs = []
    count = 0
    for file in tqdm(filenames):
        try:
            dfs.append(pd.read_csv(os.path.join(input_path, file)))
        except:
            count += 1

    df_black = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    print(f"There was {count} reading errors")

    embedding_model = api.load("glove-wiki-gigaword-50")

    output_path = AbsolutePathBuilder.get_path(
        f"07_youtube_wmd",
        filepaths="../../../../config/filepaths.yaml"
    )

    file_idx = 0
    data = []
    distances = []
    for idx, row_white in tqdm(df_white.iterrows(), total=df_white.shape[0]):
        closest_row_black = None
        closest_dist = np.inf
        for _, row_black in df_black.iterrows():
            cur_distance = embedding_model.wmdistance(row_white.text, row_black.text)

            if(cur_distance < closest_dist):
                closest_dist = cur_distance
                closest_row_black = row_black

        closest_row_black.index = [f"{name}_black" for name in closest_row_black.index]
        concatenated_row = pd.concat([closest_row_black, row_white])
        
        data.append(concatenated_row)
        distances.append(closest_dist)
        
        if ((idx+1) % 250 == 0) or (idx == df_white.index[-1]):
            df_wmd = pd.DataFrame(data)
            df_wmd = df_wmd.reset_index(drop=True)

            df_wmd["wmd"] = distances

            data.clear()
            distances.clear()
            
            df_wmd.to_csv(os.path.join(output_path, f"youtube_wmd_p{file_idx}.csv"), index=False)
            del df_wmd
            
            file_idx += 1


if __name__=="__main__":
    main()
