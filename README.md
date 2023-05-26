# Open-vocabulary event extraction tool


## Overview
Here we provide an implementation of the event extractor in Python3. The repository is organized as follows:
- `extractor.py` is the core implementation of the extractor, which extracts a list of events from a set of preprocessed sentences.
- `extract_event.py` provides a pipeline to call the extractor. Specifically, given a document, this pipeline first preprocesses each sentence using dependency parsing and then extracts events from these sentences using our extractor. Noted that the path to Stanford CoreNLP, `path_to_corenlp`, needs to be set in the code. 
- `mp_extract_event.py` implements the abovementioned pipeline in a multiprocessing way, which can handle multiple documents at the same time to achieve higher efficiency. Noted that here also need the path to Stanford CoreNLP, `path_to_corenlp`. 
- `example.json` is an example input which includes one document. 
- `output/` will contain the automatically extracted events after running `extract_event.py` or `mp_extract_event.py`.


## Run the pipeline
You can choose to process data sequentially by running the following command:
```
python extract_event.py 
```

or process larger-scale data in parallel by running the following command:
```
python mp_extract_event.py 
```


## Dependencies
- Python 3.8.5
- [stanfordcorenlp](https://stanfordnlp.github.io/CoreNLP/) 3.9.2
- nltk 3.4.5
- multiprocessing
- cytoolz 
- json

## Reference
If you use this tool, please cite these two papers:
```
@inproceedings{DBLP:conf/www/JiaoZSZZ023,
  author       = {Yizhu Jiao and
                  Ming Zhong and
                  Jiaming Shen and
                  Yunyi Zhang and
                  Chao Zhang and
                  Jiawei Han},
  editor       = {Ying Ding and
                  Jie Tang and
                  Juan F. Sequeda and
                  Lora Aroyo and
                  Carlos Castillo and
                  Geert{-}Jan Houben},
  title        = {Unsupervised Event Chain Mining from Multiple Documents},
  booktitle    = {Proceedings of the {ACM} Web Conference 2023, {WWW} 2023, Austin,
                  TX, USA, 30 April 2023 - 4 May 2023},
  pages        = {1948--1959},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3543507.3583295},
  doi          = {10.1145/3543507.3583295},
  timestamp    = {Tue, 02 May 2023 14:07:23 +0200},
  biburl       = {https://dblp.org/rec/conf/www/JiaoZSZZ023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```
@inproceedings{DBLP:conf/emnlp/ZhongLGMJZXZ0H22,
  author       = {Ming Zhong and
                  Yang Liu and
                  Suyu Ge and
                  Yuning Mao and
                  Yizhu Jiao and
                  Xingxing Zhang and
                  Yichong Xu and
                  Chenguang Zhu and
                  Michael Zeng and
                  Jiawei Han},
  editor       = {Yoav Goldberg and
                  Zornitsa Kozareva and
                  Yue Zhang},
  title        = {Unsupervised Multi-Granularity Summarization},
  booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP}
                  2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022},
  pages        = {4980--4995},
  publisher    = {Association for Computational Linguistics},
  year         = {2022},
  url          = {https://aclanthology.org/2022.findings-emnlp.366},
  timestamp    = {Fri, 03 Mar 2023 20:37:32 +0100},
  biburl       = {https://dblp.org/rec/conf/emnlp/ZhongLGMJZXZ0H22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
