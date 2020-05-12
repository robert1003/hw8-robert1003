# setup environment
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# import packages
import torch
import os
import sys
import time
import argparse
import random
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from _word2vec import Word2Vec
from _model import Seq2seq, EncoderRNN, DecoderRNN, Attention
from _utils import read_data, bleu, schedule_sampling, timeSince, train, evaluate, cut_token

# hyperparams
args = {
    'device': 'cuda',
    'dir': sys.argv[1],
    'output_name': sys.argv[2],
    'len_X': 40,
    'len_Y': 30,
    'BATCH_SIZE': 64,
    'model_name': sys.argv[3],
}
args = argparse.Namespace(**args)

# set random seed
random.seed(1003)
np.random.seed(1003)
torch.manual_seed(1003)
torch.cuda.manual_seed_all(1003)
torch.backends.cudnn.deterministic = True

# word2vec
en, cn = Word2Vec(os.path.join(args.dir, '{}_en.json')), Word2Vec(os.path.join(args.dir, '{}_cn.json'))

en_BOS_token = en.word2idx['<BOS>']
en_EOS_token = en.word2idx['<EOS>']
en_PAD_token = en.word2idx['<PAD>']
en_UNK_token = en.word2idx['<UNK>']

cn_BOS_token = cn.word2idx['<BOS>']
cn_EOS_token = cn.word2idx['<EOS>']
cn_PAD_token = cn.word2idx['<PAD>']
cn_UNK_token = cn.word2idx['<UNK>']

# train, valid, test datas
#train_X, train_Y = read_data(os.path.join(args.dir, 'training.txt'))
#valid_X, valid_Y = read_data(os.path.join(args.dir, 'validation.txt'))
test_X, test_Y = read_data(os.path.join(args.dir, 'testing.txt'))

#train_X = en.sent2idx(train_X, args.len_X)
#valid_X = en.sent2idx(valid_X, args.len_X)
test_X = en.sent2idx(test_X, args.len_X)

#train_Y = cn.sent2idx(train_Y, args.len_Y)
#valid_Y = cn.sent2idx(valid_Y, args.len_Y)
test_Y = cn.sent2idx(test_Y, args.len_Y)

# create dataset
#train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
#valid_dataset = TensorDataset(torch.from_numpy(valid_X), torch.from_numpy(valid_Y))
test_dataset = TensorDataset(torch.from_numpy(test_X), torch.from_numpy(test_Y))

#train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

# model params
en_embedding = (len(en.word2idx), 512)
encoder_input_size = en_embedding[1]
encoder_hidden_size = 256
encoder_n_layers = 1
encoder_direction = 2
encoder_dropout = 0.0

attn_size = encoder_direction * encoder_hidden_size
cn_embedding = (len(cn.word2idx), 512)
decoder_input_size = cn_embedding[1]
decoder_hidden_size = 256 * 2
decoder_output_size = cn_embedding[0]
decoder_n_layers = 1
decoder_direction = 1
decoder_dropout = 0.0

# create model
model = Seq2seq(
    in_embedding=en_embedding,
    out_embedding=cn_embedding,
    encoder=EncoderRNN(encoder_input_size, encoder_hidden_size, encoder_n_layers, encoder_direction, encoder_dropout),
    decoder=DecoderRNN(decoder_input_size, decoder_hidden_size, decoder_output_size, attn_size, decoder_n_layers, decoder_direction, decoder_dropout),
    attention = Attention(encoder_direction * encoder_hidden_size, decoder_n_layers * decoder_direction * decoder_hidden_size),
    dropout=0.5
).to(args.device)
optimizer = optim.Adam(model.parameters(), amsgrad=True)
criterion = nn.CrossEntropyLoss(ignore_index=cn_PAD_token)

# load best model
checkpoint = torch.load(args.model_name)
model.load_state_dict(checkpoint['model_state_dict'])

# beam test loss
beam_size = 2
total_loss = 0
tot = len(test_loader)
predict, true = [], []
for i, (x, y) in enumerate(test_loader):
    pred = evaluate(model, criterion, x, y, cn_BOS_token, EOS_token=cn_EOS_token, PAD_token=cn_PAD_token, TARGET_LEN=args.len_Y, beam_size=beam_size, beam_search=True)
    #total_loss += loss
    predict.append(pred)
    true.append(y.detach().cpu().numpy())
    print('predicting: {}/{}'.format(i + 1, tot), end='\r', file=sys.stderr)
print('', file=sys.stderr)
predict = [' '.join(cut_token(pred)) for pred in cn.idx2sent(np.vstack(predict))]

# print to file
with open(args.output_name, 'w') as f:
    for pred in predict:
        print(pred, file=f)
