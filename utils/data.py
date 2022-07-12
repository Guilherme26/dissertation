import collections
import os

from urllib.parse import urlparse
from urllib.parse import parse_qs

import pandas as pd

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
