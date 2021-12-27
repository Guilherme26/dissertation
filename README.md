# Dissertation

## Overview

This project explores Machine Learning (ML) biases against demographic groups. Historically speaking, predominating demographics have been perpetrating discriminating behavior against some peoples/races (e.g. Black, Gays, Latinos, Muslims, etc.) for no reason. Since these benefited groups ruled most of the business and relevant parts of society, their actions manifested their cognitive process which was further incorporated in the data generated on the decision processes controlled by them.

As we know, a necessary component to build high-quality ML models is reliable data. Sometimes the available datasets may seem taintless at first, but still comprise elaborated biases that are only brought up in the lights of quantitative analysis on those groups.

To better explore possible biases towards some demographic groups, I've followed a well-defined framework with the steps listed below:

- Manually list single-speaker videos of movie reviews from the top 50 titles of Rotten Tomatoes' Best Movies of All Times ranking;
- Manually classify each review video in one of the following demographic groups, Black Men/Women, White Men/Women;
- Extract the automatically generated subtitles of each video;
- Pre-process the sentences to build a corpus for each video. Note: the automatically generated subtitles from YouTube videos are not punctuated;
	- Remove special tokens;
	- Remove the entire subtitle from videos with censored curse words;
- Segment the sentences based on semantics:
	- Extract subtitles from TED Talk videos;
	- Pre-process these subtitles to indentify punctuations, and remove unnecessary characters;
	- Download and incorporate GloVe embeddings on a more complex model in order to allow fine-tuning on the task of punctuating sentences;
	- Fine-tune the complex model with the TED Talk sentences.
- Present each sentence to Perspective API, and save their respective toxicity scores;
- Analyze the data;

## Directories Description:

### `data/`
Stores all data from YouTube, TEDTalks, as well as their download descriptions (e.g. video name, link, etc.).

### `notebooks/`
Stores the pre-processing and analysis notebooks.

### `segmenter/`
Stores the code used to train and save the segmentation model.

### `ted_scraper/`
Stores all files regarding the TED Talks scraper.

### `utils/`
Stores all ad-hoc files required on the previously listed directories.
