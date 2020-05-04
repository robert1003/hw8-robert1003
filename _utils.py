import random
import nltk
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from queue import PriorityQueue
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def read_data(fname):
    X, Y = [], []
    with open(fname, 'r') as f:
        for line in f:
            raw_x, raw_y = line.strip().split('\t')
            X.append(raw_x.split())
            Y.append(raw_y.split())
    return X, Y

def schedule_sampling(step, steps, schedule_type='abc'):
    if schedule_type == 'linear':
        return 1.0 - 1.0 / steps * step
    elif schedule_type == 'exponential':
        return 0.999 ** step
    elif schedule_type == 'inverse_sigmoid':
        return 1000 / (1000 + np.exp(step / 1000))
    elif schedule_type == 'none':
        return 0
    else:
        return 1

def bleu(sentences, targets):
    score = 0 
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<EOS>' or token == '<PAD>':
                break
            elif token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp 
    
    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))                                                                                          

    return score / len(sentences)

def _asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, epoch, epochs):
    now = time.time()
    s = now - since
    rs = s / epoch * (epochs - epoch)
    return '%s (- %s)' % (_asMinutes(s), _asMinutes(rs))

def train(model, optimizer, criterion, input_tensor, target_tensor, BOS_token, grad_max, teacher_forcing_ratio, device='cuda'):
    model.train()
    
    batch_size = input_tensor.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size).to(device)
    
    optimizer.zero_grad()
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    enc_outputs, encoder_hidden = model(input_tensor, encoder_hidden, batch_size, encoding=True, enc_outputs=None)
    
    decoder_input = torch.LongTensor([BOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
    encoder_hidden = encoder_hidden.view(model.encoder.n_layers, model.encoder.direction, batch_size, model.encoder.hidden_size)
    decoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2)

    loss = 0
    for di in target_tensor:
        decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False, enc_outputs=enc_outputs)
        loss += criterion(decoder_output, di.view(-1))
        if random.random() < teacher_forcing_ratio:
            decoder_input = di
        else:
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.detach().to(device)
    loss.backward()
    
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max)
    optimizer.step()
    
    return loss.item() / target_length

class BeamNode(object):
    def __init__(self, hidden, prev, idx, score, length):
        self.hidden = hidden
        self.prev = prev
        self.idx = idx
        self.score = score
        self.length = length

    def eval(self):
        return self.score / self.length

    def __lt__(self, o):
        return self.eval() > o.eval()

def evaluate(model, criterion, input_tensor, target_tensor, BOS_token, device='cuda', EOS_token=2, PAD_token=0, TARGET_LEN=30, beam_size=3, beam_search=False):
    model.eval()
    with torch.no_grad():
        batch_size = input_tensor.size(0)
        encoder_hidden = model.encoder.initHidden(batch_size).to(device)

        input_tensor = input_tensor.transpose(0, 1).to(device)
        target_tensor = target_tensor.transpose(0, 1).to(device)

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
        enc_outputs, encoder_hidden = model(input_tensor, encoder_hidden, batch_size, encoding=True, enc_outputs=None)

        decoder_input = torch.LongTensor([BOS_token for _ in range(batch_size)]).view(-1, 1).to(device)
        encoder_hidden = encoder_hidden.view(model.encoder.n_layers, model.encoder.direction, batch_size, model.encoder.hidden_size)
        decoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2)
        
        if beam_search:
            decoder_predict = []
            for i in range(batch_size):
                start_node = BeamNode(decoder_hidden[:, i:i + 1, :].contiguous(), None, decoder_input[i, :].contiguous(), 0, 1)

                all_nodes = [start_node]
                now_nodes = [start_node]
                end_pq = PriorityQueue()

                for j in range(TARGET_LEN):
                    if len(now_nodes) == 0:
                        break

                    pq = PriorityQueue()

                    for node in now_nodes:
                        input, hidden = node.idx, node.hidden
                        output, hidden = model(input, hidden, 1, encoding=False, enc_outputs=enc_outputs[:, i:i + 1, :])
                        output = F.log_softmax(output, dim=1)
                        topv, topi = output.data.topk(beam_size)
                        for (score, idx) in zip(topv.detach().squeeze(0), topi.detach().squeeze(0)):
                            nxt_node = BeamNode(hidden, node, idx.unsqueeze(0), node.score + score, node.length + 1)
                            pq.put(nxt_node)

                    now_nodes = []
                    for _ in range(beam_size):
                        assert pq.qsize() > 0
                        node = pq.get()
                        all_nodes.append(node)
                        if node.idx == EOS_token or j == TARGET_LEN - 1:
                            end_pq.put(node)
                        else:
                            now_nodes.append(node)

                assert end_pq.qsize() > 0
                best_node = end_pq.get()

                predict = [best_node.idx.cpu().numpy()[0]]
                while best_node.prev is not None:
                    best_node = best_node.prev
                    predict.append(best_node.idx.cpu().numpy()[0])
                predict = predict[-2::-1]

                while len(predict) < TARGET_LEN:
                    predict.append(PAD_token)
                
                decoder_predict.append(np.array(predict))
                
            decoder_predict = np.stack(decoder_predict)
            return decoder_predict
        else:
            loss = 0
            decoder_predict = []
            for di in target_tensor:
                decoder_output, decoder_hidden = model(decoder_input, decoder_hidden, batch_size, encoding=False, enc_outputs=enc_outputs)
                loss += criterion(decoder_output, di.view(-1))
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.detach().to(device)

                decoder_predict.append(topi.cpu().numpy())
                
            decoder_predict = np.hstack(decoder_predict)
            return loss.item() / target_length, decoder_predict
