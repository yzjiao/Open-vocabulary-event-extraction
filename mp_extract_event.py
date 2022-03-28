import os
import json
from time import time
from datetime import timedelta
import multiprocessing as mp
from cytoolz import curry
from os.path import join
from stanfordcorenlp import StanfordCoreNLP
from extractor import Extractor
from nltk import sent_tokenize

input_file = 'example.json'
output_path = 'output'
path_to_corenlp = r'./stanford-corenlp-full-2018-10-05'



@curry
def extract_event(idx):
    if os.path.exists(join(output_path, '{}.json'.format(idx))):
        return
    sent_info = []
    src, event = [], []
    for sent in sent_tokenize(data[idx]['src']):
        if len(sent) > 1024:
            continue
        info = {}
        src.append(sent)
        info['sentence'] = sent
        info['word'] = nlp.word_tokenize(sent)
        info['pos'] = nlp.pos_tag(sent)
        info['dependency'] = nlp.dependency_parse(sent)
        sent_info.append(info)
    cur_event = extractor.extract(sent_info)
    for j in range(len(cur_event)):
        event.append(' | '.join(cur_event[j]))
    assert len(src) == len(event)
    with open(join(output_path, '{}.json'.format(idx)), 'w') as f:
        cur = {}
        cur['id'] = data[idx]['id']
        cur['src'] = src
        cur['event'] = event
        json.dump(cur, f, indent=4)


def extract_event_mp():
    global data, extractor, nlp
    nlp = StanfordCoreNLP(path_to_corenlp)
    extractor = Extractor()

    data = []
    with open(input_file) as f:
        data = json.load(f)
    n_files = len(data)


    start = time()
    print('extracting events from {} documents with multi-processing !!!'.format(n_files))
    
    with mp.Pool() as pool:
        list(pool.imap_unordered(extract_event(), range(n_files), chunksize=100))
    
    print('finished in {}'.format(timedelta(seconds=time()-start)))
    nlp.close()


if __name__ == '__main__':
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    extract_event_mp()



