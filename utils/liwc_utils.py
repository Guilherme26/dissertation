def liwc_sentence_parse(s,liwc):
    return liwc.parse(s.split(' '))

def get_key_counts(counter,key):
    if key in counter:
        return counter[key]
    else:
        return 0