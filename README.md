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



