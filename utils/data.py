import collections
import os

from urllib.parse import urlparse
from urllib.parse import parse_qs

import pandas as pd
import glob

import yaml

filepaths = yaml.safe_load(open("/home/luiznery/locus/dissertation/config/filepaths.yaml"))

def get_subtitles_df(path_00_raw = '../data/00_raw/'):
    if path_00_raw[-1] != '/':
        path_00_raw += '/'

    df = pd.DataFrame(columns=['id','text'])
    for _id in os.listdir(path_00_raw):
        text = open(path_00_raw+_id).read()
        d = {
            'id': _id,
            'text': text
        }
        df = df.append(d,ignore_index=True)
    return df

def clean_sub(s, punctuations = ['!','.',':',';','<','=','>','?']):
    while s.find('[') >= 0:
        tmp = s[:s.find('[')]
        s = tmp + s[s.find(']')+1:]
    for p in punctuations:
        s = s.replace(p,'')
    return s

def count_tags(s):
    count = collections.Counter()

    while s.find('[') >= 0:
        tag = s[s.find('[')+1:s.find(']')]
        count[tag] = count[tag] + 1
        tmp = s[:s.find('[')]
        s = tmp + s[s.find(']')+1:]   
    return count

def get_clean_subtitles_df(path_00_raw = '../data/00_raw/'):
    df = get_subtitles_df(path_00_raw)
    df['text_clean'] = df['text'].apply(clean_sub)
    return df

def get_video_id(url):
    parsed_url = urlparse(url)
    captured_value = parse_qs(parsed_url.query)['v'][0]
    return captured_value

def get_movies_dataset(csv_path='../data/download_descriptions.csv', path_00_raw='../data/00_raw/'):
    subs_df = get_clean_subtitles_df(path_00_raw)
    df = pd.read_csv(csv_path)

    df['id'] = df.url.apply(get_video_id)

    df = df.merge(subs_df,how='outer')
    df = df.drop_duplicates()
    df = df[df.text.notna()].reset_index(drop=True)
    df['tags_count'] = df['text'].apply(count_tags)
    return df


# #reads data - specific to each dataset
# def read_interview():
#     f_or_m = lambda x: 'Woman' if x == 'f' else 'Man'

#     data = pd.concat(
#         [
#             pd.read_csv(file) \
#                 .assign(
#                     file=file.split('/')[-1],
#                     group=' '.join([file.split('/')[-1].split('-')[0],f_or_m(file.split('/')[-1].split('-')[1])]).title()
                    
#                 )
#             for file in glob.glob(filepaths['interview_scored']+'*')
#         ]
#     )

#     return data

def read_coraal_buckeye():
    buckeye_meta = pd.read_csv(filepaths['buckeye_data_description'])
    buckeye_meta.SPEAKER = buckeye_meta.SPEAKER.str.lower()

    extract_coral_gender = lambda file: file.split('_')[-3]
    f_or_m = lambda x: 'Woman' if x == 'f' else 'Man' if x == 'm' else np.nan

    get_buckeye_gender = lambda speaker: buckeye_meta[buckeye_meta.SPEAKER==speaker].iloc[0]["SPEAKER'S GENDER"]
    
    data = pd.concat(
        [ #reading coraal
            pd.read_csv(file) \
                .assign(
                    file=file.split('/')[-1],
                    group= 'Black ' + f_or_m(extract_coral_gender(file))
                )
            for file in glob.glob(filepaths['04_coraal_scored']+'*')
        ] + [ #reading buckeye
            pd.read_csv(file) \
                .assign(
                    file=file.split('/')[-1],
                    group= 'White ' + f_or_m(get_buckeye_gender(file.split('/')[-1]))
                )
            for file in glob.glob(filepaths['04_buckeye_scored']+'*')
        ]
    )
    return data

def read_youtube():
    desc = pd.read_csv(filepaths['youtube_data_description'])
    desc['id'] = desc.url.apply(lambda x: x.split('/')[-1].split('&')[0].split('=')[-1])

    data = pd.concat([
        pd.read_csv(file) \
            .assign(
                file=file.split('/')[-1],
            )
        for file in glob.glob(filepaths['04_youtube_scored']+'*')
    ]) 
    data = data.merge(desc[['group','id']], right_on='id', left_on='file').drop(columns=['id'])
    return data

def read_twitter():
    data = pd.concat([
        pd.read_csv(file) \
            .assign(
                file=file.split('/')[-1],
                group = file.split('/')[-1].split('_')[0]
            )
        for file in glob.glob(filepaths['04_twitter_scored']+'*')
    ]) 
    return data

def read_interview():
    process_race_gender = lambda x: "{} {}".format(x[0].title(), process_gender(x[1]))
    process_gender = lambda x: 'Woman' if x == 'f' else 'Man' if x == 'm' else np.nan

    data = pd.concat([
        pd.read_csv(file) \
            .assign(
                file=file.split('/')[-1],
                group = process_race_gender(file.split('/')[-1].split('-')[0:2]),
            )
        for file in glob.glob(filepaths['04_interview_scored']+'*')
    ]) 
    return data

def load_dataset(dataset):
    if dataset == 'youtube':
        data = read_youtube()
    elif dataset == 'coraal-buckeye':
        data = read_coraal_buckeye()
    elif dataset == 'twitter':
        data = read_twitter()
    elif dataset == 'interview':
        data = read_interview()
    else:
        raise Exception('Unknown dataset')
    data = data.reset_index(drop=True)
    return data