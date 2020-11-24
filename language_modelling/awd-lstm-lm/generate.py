###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model, _, _ = torch.load(f)
model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

lines = []
with open(args.data + 'test.txt', 'r') as test:
    lines = test.readlines()
lines = [line for line in lines if line.strip() != '']
first_line = lines[0].split() + ['<eos>']
first_word = first_line[0]

# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
input = torch.tensor([[corpus.dictionary.word2idx[first_word]]])
if args.cuda:
    input.data = input.data.cuda()
hidden = model.init_hidden(1)

success = 0
error = 0
wpa = 0

with open(args.outf, 'a') as outf:
    for line in lines:
        tokens = line.split() + ['<eos>']
        for token in tokens[1:]:
            output, hidden = model(input, hidden)
            word_weights = model.decoder(output).squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(corpus.dictionary.word2idx[token])
            try:
                word = corpus.dictionary.idx2word[word_idx]
            except:
                word = '<unk>'
            print(word, token)

            if word == token:
                success +=  1
            else:
                error += 1

        wpa = success / (success + error)
    outf.write('\n' + args.checkpoint + ': ' + str(wpa))
