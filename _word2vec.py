import sys
import numpy as np
import json

class Word2Vec:
    
    def __init__(self, embedding_file):
        self._loadEmbedding(embedding_file)
        
    def _loadEmbedding(self, embedding_file):
        print('Loading processed embedding...', file=sys.stderr)
        with open(embedding_file.format('word2int'), 'r') as f:
            self.word2idx = json.load(f)
        with open(embedding_file.format('int2word'), 'r') as f:
            self.idx2word = {int(k): v for k, v in json.load(f).items()}
        print('done', file=sys.stderr)
        
    def _pad_sent(self, sent, sent_len):        
        if len(sent) > sent_len:
            sent = sent[:sent_len]
        else:
            for _ in range(sent_len - len(sent)):
                sent.append(self.word2idx['<PAD>'])
        return sent
    
    def sent2idx(self, sents, sent_len):
        sent_list = []
        unk_cnt = 0
        for i, sent in enumerate(sents):
            word_idx = []
            
            #word_idx.append(self.word2idx['<BOS>'])
            for word in sent:
                if word in self.word2idx.keys():
                    word_idx.append(self.word2idx[word])
                else:
                    word_idx.append(self.word2idx['<UNK>'])
                    unk_cnt += 1
            word_idx.append(self.word2idx['<EOS>'])
            
            word_idx = self._pad_sent(word_idx, sent_len)
            sent_list.append(word_idx)
        print('#{} of sents processed'.format(len(sent_list)), file=sys.stderr)
        print('#{} of unknown words'.format(unk_cnt), file=sys.stderr)
        return np.vstack(sent_list)
    
    def idx2sent(self, sents):
        sent_list = []
        for i, sent in enumerate(sents):
            word = []
            for idx in sent:
                word.append(self.idx2word[idx])
            sent_list.append(word)
        return sent_list
