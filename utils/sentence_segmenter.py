import torch

from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm

class GptSegmenter:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese",)
        self.model = AutoModelWithLMHead.from_pretrained("pierreguillou/gpt2-small-portuguese",)
        self.tokenizer.model_max_length=1024 

        self.model.eval() 
        print()
    
    def get_topk_next_tokens(self, text, k):
        """
        retorna os tok k tokens mais provaveis
        """
        inputs = self.tokenizer(text, return_tensors="pt")

        # model output
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss, logits = outputs[:2]

        topk = []
        for word in torch.sort(logits[0, -1, :])[1][-k:]:
            topk.append(self.tokenizer.decode(word))

        return topk

    def isEOS(self, topk, n = 1, punctuations = ['!','.',':',';','?','...','\n','<|endoftext|>']):
        count = 0
        for tok in topk:
            for p in punctuations:
                if p in tok:
                    count += 1
                    if count >= n:
                        return True
        return False
    
    def isEOSnextword(self, topk, n = 1, punctuations = ['!','.',':',';','?','...','\n','<|endoftext|>'], next_word = None):
        count = 0
        next_word_predict = False
        if next_word!=None and next_word in topk:
            next_word_predict = True

        for i in range(len(topk)):
            for p in punctuations:
                if p in topk[i]:
                    count += 1
                    if count >= n and next_word_predict == False:
                        return True
        return False

    def segment(self, sentence, k = 2, min_sent_sz = 5, max_sent_sz=30):
        sub_tokens = sentence.split(" ")
        
        last_point = 0
        response = []
        actual_sent = []
        p = 0
        for i in range(len(sub_tokens)):
            if i - last_point >= min_sent_sz:
                actual_sent.append(sub_tokens[i])

                top = self.get_topk_next_tokens(" ".join(sub_tokens[last_point+1:i+1]), k+p)
                # topk = get_topk_next_tokens(" ".join(sub_tokens[last_point:i+1]), model, tokenizer, k)
                if self.isEOS(top) : #and " "+sub_tokens[i+1] not in top:  
                    response.append(actual_sent)
                    actual_sent = []
                    last_point = i
                    p = 0
                else:
                    if i - last_point >= max_sent_sz:
                        if p==0:
                            p = 1
                        else:
                            p = p*2
            else:
                actual_sent.append(sub_tokens[i])
        
        if actual_sent != []:
            response.append(actual_sent)

        return response