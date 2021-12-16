import pandas as pd
def get_word_relations(word, data):
    before = {}
    after = {}
    for sub in data:
        words = sub.split(' ')
        for i in range(len(words)):
            if words[i] == word:
                if i>0:
                    if words[i-1] not in before:
                        before[words[i-1]] = 1
                    else:
                        before[words[i-1]] += 1
                if i<len(words)-1:
                    if words[i+1] not in after:
                        after[words[i+1]] = 1
                    else:
                        after[words[i+1]] += 1

    before = pd.DataFrame().from_dict({'word':before.keys(),'count':before.values()})
    after = pd.DataFrame().from_dict({'word':after.keys(),'count':after.values()})
    
    return before, after