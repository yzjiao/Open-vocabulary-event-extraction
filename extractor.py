import json
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


SUBGRAPH_PATTERNS = [
    {'pos':['V', 'N', 'NOT', 'N', 'JJ', 'V', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1, 2, 3, 4], 4:[5], 5:[6]}}, 
    
    {'pos':['V', 'N', 'NOT', 'N', 'N', 'V', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1, 2, 3, 4], 4:[5], 5:[6]}}, 
    
    {'pos':['V', 'N', 'NOT', 'N', 'V', 'N', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'obj', 'xcomp', 'obj', 'mark'], 'child':{0:[1,2, 3, 4], 4:[5, 6]}}, 
    
    {'pos':['V', 'N', 'N', 'V', 'N', 'TO'], 'dependency':['root', 'nsubj', 'obj', 'xcomp', 'obj', 'mark'], 'child':{0:[1,2, 3], 3:[4, 5]}}, 

    {'pos':['V', 'NOT', 'N', 'V', 'N', 'TO'], 'dependency':['conj', 'advmod', 'obj', 'xcomp', 'obj', 'mark'], 'child':{0:[1,2, 3], 3:[4, 5]}}, 

    {'pos':['V', 'N', 'N', 'JJ', 'V', 'TO'], 'dependency':['root', 'nsubj', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1, 2, 3], 3:[4], 4:[5]}}, 
    
    {'pos':['V', 'NOT', 'N', 'JJ', 'V', 'TO'], 'dependency':['conj', 'advmod', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1, 2, 3], 3:[4], 4:[5]}}, 

    {'pos':['V', 'N', 'NOT', 'N', 'N', 'IN'], 'dependency':['root', 'nsubj', 'advmod', 'obj', 'obl', 'case'], 'child':{0:[1, 2, 3], 3:[4], 4:[5]}}, 
    
    {'pos':['JJ', 'N', 'NOT', 'V', 'N', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'xcomp', 'obj', 'mark'], 'child':{0:[1, 2, 3], 3:[4, 5]}}, 

    {'pos':['V', 'N', 'N', 'N', 'V', 'TO'], 'dependency':['root', 'nsubj', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2, 3], 3:[4], 4:[5]}}, 

    {'pos':['V', 'NOT', 'N', 'N', 'V', 'TO'], 'dependency':['conj', 'advmod', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2, 3], 3:[4]}}, 
    
    {'pos':['V', 'N', 'NOT', 'N', 'N', 'IN'], 'dependency':['root', 'nsubj', 'advmod', 'obj', 'obl', 'case'], 'child':{0:[1,2, 3, 4], 4:[5]}}, 
    
    {'pos':['V', 'N', 'NOT', 'JJ', 'V', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2,3], 3:[4], 4:[5]}}, 
    
    {'pos':['V', 'N', 'NOT', 'N', 'V', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2,3], 3:[4], 4:[5]}}, 
    
    {'pos':['V', 'N', 'NOT', 'V', 'N', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'xcomp', 'obj', 'mark'], 'child':{0:[1,2,3], 3:[4, 5]}}, 

    {'pos':['V', 'N', 'N', 'V', 'TO'], 'dependency':['conj', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2], 2:[3], 3:[4]}}, 
 
    {'pos':['V', 'N', 'JJ', 'V', 'TO'], 'dependency':['conj', 'obj', 'xcomp', 'cop', 'mark'], 'child':{0:[1, 2], 2:[3], 3:[4]}}, 

    {'pos':['V', 'N', 'N', 'N', 'IN'], 'dependency':['root', 'nsubj', 'obj', 'obl', 'case'], 'child':{0:[1, 2], 2:[3], 3:[4]}}, 
    
    {'pos':['V', 'NOT', 'N', 'N', 'IN'], 'dependency':['conj', 'advmod', 'obj', 'obl', 'case'], 'child':{0:[1,2], 2:[3], 3:[4]}}, 
    
    {'pos':['V', 'N', 'NOT', 'N', 'N'], 'dependency':['root', 'nsubj', 'advmod', 'iobj', 'obj'], 'child':{0:[1, 2, 3, 4]}}, 
    
    {'pos':['JJ', 'N', 'V', 'N', 'TO'], 'dependency':['root', 'nsubj', 'xcomp', 'obj', 'mark'], 'child':{0:[1, 2], 2:[3, 4]}}, 
    
    {'pos':['JJ', 'NOT', 'V', 'N', 'TO'], 'dependency':['root', 'advmod', 'xcomp', 'obj', 'mark'], 'child':{0:[1, 2], 2:[3, 4]}}, 
    
    {'pos':['JJ', 'N', 'NOT', 'N', 'IN'], 'dependency':['root', 'nsubj', 'advmod', 'obl', 'case'], 'child':{0:[1, 2, 3], 3:[4]}}, 

    {'pos':['V', 'N', 'V', 'N', 'TO'], 'dependency':['root', 'nsubj', 'xcomp', 'obj', 'mark'], 'child':{0:[1,2], 2:[3, 4]}}, 

    {'pos':['V', 'NOT', 'V', 'N', 'TO'], 'dependency':['conj', 'advmod', 'xcomp', 'obj', 'mark'], 'child':{0:[1,2], 2:[3, 4]}}, 

    {'pos':['V', 'N', 'NOT', 'N', 'IN'], 'dependency':['root', 'nsubj:pass', 'advmod', 'obl', 'case'], 'child':{0:[1,2,3], 3:[4]}}, 
    
    {'pos':['V', 'N', 'NOT', 'N', 'IN'], 'dependency':['root', 'nsubj', 'advmod', 'obl', 'case'], 'child':{0:[1,2,3], 3:[4]}},
        
    {'pos':['V', 'N', 'V', 'N', 'TO'], 'dependency':['conj', 'obj', 'xcomp', 'obj', 'mark'], 'child':{0:[1,2], 2:[3, 4]}}, 
    
    {'pos':['V', 'N', 'N', 'N', 'IN'], 'dependency':['root', 'nsubj', 'obj', 'obl', 'case'], 'child':{0:[1,2, 3], 3:[4]}}, 

    {'pos':['V', 'NOT', 'N', 'N', 'IN'], 'dependency':['conj', 'advmod', 'obj', 'obl', 'case'], 'child':{0:[1,2,3], 3:[4]}}, 
    
    {'pos':['V', 'N', 'JJ', 'V', 'TO'], 'dependency':['root', 'nsubj', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2], 2:[3], 3:[4]}}, 
    
    {'pos':['V', 'NOT', 'JJ', 'V', 'TO'], 'dependency':['conj', 'advmod', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2], 2:[3], 3:[4]}}, 
    
    {'pos':['V', 'N', 'N', 'V', 'TO'], 'dependency':['root', 'nsubj', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2], 2:[3], 3:[4]}}, 

    {'pos':['V', 'NOT', 'N', 'V', 'TO'], 'dependency':['conj', 'advmod', 'xcomp', 'cop', 'mark'], 'child':{0:[1,2], 2:[3], 3:[4]}}, 

    {'pos':['V', 'N', 'NOT', 'V', 'TO'], 'dependency':['root', 'nsubj', 'advmod', 'xcomp', 'mark'], 'child':{0:[1,2,3], 3:[4]}}, 

    {'pos':['V', 'N', 'N', 'IN'], 'dependency':['conj', 'obj', 'obl', 'case'], 'child':{0:[1], 1:[2], 2:[3]}}, 
    
    {'pos':['V', 'N', 'N', 'N'], 'dependency':['root', 'nsubj', 'iobj', 'obj'], 'child':{0:[1, 2, 3]}}, 

    {'pos':['V', 'NOT', 'N', 'N'], 'dependency':['conj', 'advmod', 'iobj', 'obj'], 'child':{0:[1, 2, 3]}}, 
    
    {'pos':['JJ', 'V', 'N', 'TO'], 'dependency':['root', 'xcomp', 'obj', 'mark'], 'child':{0:[1], 1:[2, 3]}}, 
    
    {'pos':['JJ', 'N', 'N', 'IN'], 'dependency':['root', 'nsubj', 'obl', 'case'], 'child':{0:[1, 2], 2:[3]}}, 

    {'pos':['JJ', 'NOT', 'N', 'IN'], 'dependency':['conj', 'advmod', 'obl', 'case'], 'child':{0:[1,2], 2:[3]}}, 

    {'pos':['V', 'N', 'V', 'TO'], 'dependency':['root', 'nsubj', 'xcomp', 'mark'], 'child':{0:[1,2], 2:[3]}}, 

    {'pos':['V', 'JJ', 'V', 'TO'], 'dependency':['conj', 'xcomp', 'cop', 'mark'], 'child':{0:[1], 1:[2], 2:[3]}}, 

    {'pos':['V', 'N', 'N', 'IN'], 'dependency':['conj', 'obj', 'obl', 'case'], 'child':{0:[1,2], 2:[3]}},

    {'pos':['V', 'N', 'V', 'TO'], 'dependency':['conj', 'xcomp', 'cop', 'mark'], 'child':{0:[1], 1:[2], 2:[3]}}, 

    {'pos':['V', 'V', 'N', 'TO'], 'dependency':['conj', 'xcomp', 'obj', 'mark'], 'child':{0:[1], 1:[2, 3]}}, 
      
    {'pos':['V', 'N', 'N', 'IN'], 'dependency':['root', 'nsubj', 'obl', 'case'], 'child':{0:[1,2], 2:[3]}}, 

    {'pos':['V', 'N', 'N', 'IN'], 'dependency':['root', 'nsubj:pass', 'obl', 'case'], 'child':{0:[1,2], 2:[3]}}, 

    {'pos':['V', 'NOT', 'N', 'IN'], 'dependency':['conj', 'advmod', 'obl', 'case'], 'child':{0:[1,2], 2:[3]}}, 
    
    {'pos':['V', 'N', 'NOT', 'N'], 'dependency':['root', 'nsubj', 'advmod', 'obj'], 'child':{0:[1,2,3]}}, 
    
    {'pos':['V', 'NOT', 'V', 'TO'], 'dependency':['conj', 'advmod', 'xcomp', 'mark'], 'child':{0:[1,2], 2:[3]}}, 

    {'pos':['JJ', 'N', 'NOT', 'V'], 'dependency':['root', 'nsubj', 'advmod', 'cop'], 'child':{0:[1,2,3]}}, 
    
    {'pos':['V', 'N', 'NOT', 'JJ'], 'dependency':['root', 'nsubj', 'advmod', 'xcomp'], 'child':{0:[1,2,3]}}, 
    
    {'pos':['V', 'N', 'NOT', 'RP'], 'dependency':['root', 'nsubj', 'advmod', 'compound:prt'], 'child':{0:[1,2,3]}}, 
    
    {'pos':['JJ', 'N', 'IN'], 'dependency':['conj', 'obl', 'case'], 'child':{0:[1], 1:[2]}}, 
    
    {'pos':['V', 'N', 'N'], 'dependency':['conj', 'iobj', 'obj'], 'child':{0:[1, 2]}}, 

    {'pos':['V', 'N', 'RP'], 'dependency':['root', 'nsubj', 'compound:prt'], 'child':{0:[1,2]}}, 

    {'pos':['V', 'N', 'NOT'], 'dependency':['root', 'nsubj', 'advmod'], 'child':{0:[1,2]}}, 

    {'pos':['V', 'N', 'JJ'], 'dependency':['root', 'nsubj', 'xcomp'], 'child':{0:[1,2]}}, 

    {'pos':['V', 'NOT', 'JJ'], 'dependency':['conj', 'advmod', 'xcomp'], 'child':{0:[1,2]}}, 
    
    {'pos':['JJ', 'N', 'V'], 'dependency':['root', 'nsubj', 'cop'], 'child':{0:[1,2]}}, 

    {'pos':['JJ', 'NOT', 'V'], 'dependency':['conj', 'advmod', 'cop'], 'child':{0:[1,2]}}, 
    
    {'pos':['V', 'N', 'IN'], 'dependency':['conj', 'obl', 'case'], 'child':{0:[1], 1:[2]}}, 
    
    {'pos':['V', 'NOT', 'N'], 'dependency':['conj', 'advmod', 'obj'], 'child':{0:[1,2]}}, 
        
    {'pos':['V', 'N', 'N'], 'dependency':['root', 'nsubj', 'obj'], 'child':{0:[1,2]}}, 
    
    {'pos':['V', 'V', 'TO'], 'dependency':['conj', 'xcomp', 'mark'], 'child':{0:[1], 1:[2]}}, 

    {'pos':['V', 'N', 'NOT'], 'dependency':['root', 'nsubj:pass', 'advmod'], 'child':{0:[1,2]}}, 

    {'pos':['V', 'NOT', 'RP'], 'dependency':['conj', 'advmod', 'compound:prt'], 'child':{0:[1,2]}}, 
    
    {'pos':['V', 'RP'], 'dependency':['conj', 'compound:prt'], 'child':{0:[1]}}, 

    {'pos':['JJ', 'V'], 'dependency':['conj', 'cop'], 'child':{0:[1]}}, 

    {'pos':['V', 'JJ'], 'dependency':['conj', 'xcomp'], 'child':{0:[1]}}, 
    
    {'pos':['V', 'N'], 'dependency':['root', 'nsubj'], 'child':{0:[1]}}, 
    
    {'pos':['V', 'N'], 'dependency':['root', 'nsubj:pass'], 'child':{0:[1]}}, 
    
    {'pos':['V', 'NOT'], 'dependency':['conj', 'advmod'], 'child':{0:[1]}}, 

    {'pos':['V', 'N'], 'dependency':['conj', 'obj'], 'child':{0:[1]}}, 

    {'pos':['V'], 'dependency':['conj'], 'child':{0:[]}}, 
]



class Extractor():
    def __init__(self):
        self.extract_res = [] #{sentence: tuple}
        self.all_events = {} #{tuple: frequence}   
        self.sentence = None
        self.pos = None
        self.dependency = None
        self.child = None

    
    
    def match_pattern(self, idx, pattern, loc, CodeNameId):
        pat_len = len(pattern['pos'])
        
        if trans_pos(self.pos[idx][1], self.sentence[idx]) == pattern['pos'][loc] and (self.dependency[idx][0] == pattern['dependency'][loc] or loc == 0):
            if loc not in pattern['child']:
                word_loc = [idx]
                return word_loc, CodeNameId, idx, idx

            elif idx in self.child:
                word_loc = []
                i = 0
                j = 0
                min_idx = max_idx = idx
                while j < len(pattern['child'][loc]):
                    while i < len(self.child[idx]):
                        res, CodeNameId, _min_idx, _max_idx = self.match_pattern(self.child[idx][i], pattern, pattern['child'][loc][j], CodeNameId)
                        i += 1
                        if res != None:
                            word_loc += res 
                            j += 1
                            max_idx = max(_max_idx, max_idx) 
                            min_idx = min(_min_idx, min_idx) 
                            break

                    if i == len(self.child[idx]):
                        break

                if j == len(pattern['child'][loc]):
                    word_loc = [idx] + word_loc
                    return word_loc, CodeNameId, min_idx, max_idx
        return None, CodeNameId, idx, idx
        
        

    
    def find_subgraph(self, info):
        self.sentence = info['word']
        self.pos = info['pos']
        Len = len(self.sentence)
        
        #sort the dependency
        def by_head(Tuple):
            return Tuple[-1]
        self.dependency = sorted(info['dependency'], key=by_head)

        #process child nodes
        self.child = {}
        for i in range(Len):
            if self.dependency[i][1] != 0:
                if (self.dependency[i][1] - 1) not in self.child:
                    self.child[self.dependency[i][1] - 1] = []
                self.child[self.dependency[i][1] - 1].append(i)

        subgraphs = []
        i = 0
        while i < Len:
            for pattern in SUBGRAPH_PATTERNS:
                codenameid = int(pattern['dependency'][0] == 'conj')
                word_loc, _, min_idx, max_idx = self.match_pattern(i, pattern, 0, codenameid)
                if word_loc is not None:
                    words = [self.sentence[i] for i in word_loc]
                    res = pattern.copy()
                    res['words'] = words
                    res['loc'] = word_loc
                    subgraphs.append(res)
                    i = max_idx
                    break
            i += 1
        return subgraphs
    
    def subg2event(self, subgraphs):
        events = []
        for subg in subgraphs:
            event = []
            for node in subg['child']:
                for child in subg['child'][node]:
                    event.append((subg['words'][node], subg['dependency'][child], subg['words'][child]))
            events.append(tuple(event))
        return events

    def subg2str(self, subgraphs):
        strs = []
        for subg in subgraphs:
            pair = zip(subg['loc'], subg['words'])
            pair = sorted(pair, key = lambda k: k[0])
            word_in_order = [j for (i, j) in pair]
            strs.append(' '.join(word_in_order))
        return strs

    def extract(self, story_info):
        event_list = []
        Len = len(story_info)
        for sent_info in story_info:
            #print('Sentence: ' + sent_info['sentence'])
            sent_info['word'] = lemmatize_sentence(sent_info['pos'])
            subgraphs = self.find_subgraph(sent_info)
            strs = self.subg2str(subgraphs)
            event_list.append(strs)
        return event_list
            
        
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

    
def lemmatize_sentence(pos_tag):
    res = []
    lemmatizer = WordNetLemmatizer()
    for pos in pos_tag:
        word, pos = pos
        word = word.lower()
        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos:
            res.append(lemmatizer.lemmatize(word, wordnet_pos))
        else:
            res.append(word)
    return res
    

    
def trans_pos(pos, word):
    if pos.startswith('V'):
        return 'V'
    if pos.startswith('JJ'):
        return 'JJ'
    if pos.startswith('N') or pos == 'PRP':
        return 'N'
    if pos == 'RB' and (word == 'n\'t' or word == 'not'):
        return 'NOT'
    return pos


