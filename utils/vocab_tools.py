import pandas as pd
import spacy
import collections

def count_terms_in_list(l,terms):
    count = 0
    for term in terms:
        if isinstance(term,list):
            count += sequence_ocurences_in_list(term,l)
        elif isinstance(term,str):
            count += l.count(term)
        else:
            raise Exception('term must be a string or a list')
    return count

def count_word(df,word,group): 
    return df[df.group.str.contains(group)].words.apply(lambda x: x.count(word)).sum()

def group_total_words(df,group):
    return df[df.group.str.contains(group)].len.sum()

def contain_word_idx(df,word):
    return df.words.apply(lambda x: word in x)

def calc_word_freq(df,word,group):    
    return count_word(df,word,group) / group_total_words(df,group)

def sequence_ocurences_in_list(sequence,list):
    """
    count the number of times a sequence appears in a list
    """
    i = 0
    count = 0
    while i < len(list):
        if list[i:i+len(sequence)] == sequence:
            count += 1
        i += 1
    return count

def count_sequence(df,sequence,group):
    return df[df.group.str.contains(group)].words.apply(lambda x: sequence_ocurences_in_list(sequence,x)).sum()

def contain_sequence_idx(df,sequence):
    return df.words.apply(lambda x: sequence_ocurences_in_list(sequence,x)) > 0

def calc_sequence_freq(df,sequence,group):    
    return count_sequence(df,sequence,group) / group_total_words(df,group)

def calc_freq(df,x,group):
    if isinstance(x,str):
        return calc_word_freq(df,x,group)
    elif isinstance(x,list):
        return calc_sequence_freq(df,x,group)
    else:
        raise ValueError('x must be a string or a list')

def contain_term_idx(df,term):
    if isinstance(term,str):
        return contain_word_idx(df,term)
    elif isinstance(term,list):
        return contain_sequence_idx(df,term)
    else:
        raise ValueError('term must be a string or a list')

def count_pos_classes(text,nlp):
    doc = nlp(text)
    c = collections.Counter()
    for tag in doc:
        c[tag.pos_] += 1
    return dict(c)