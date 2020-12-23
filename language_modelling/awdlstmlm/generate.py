###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

from language_modelling.awdlstmlm import data

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
parser.add_argument('--input', type=str, default='')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

with open(args.checkpoint, 'rb') as f:
    model, _, _ = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

user_input = args.input.split(",")
user_input = torch.tensor([[corpus.dictionary.word2idx[token] for token in user_input]])

if args.cuda:
    user_input.data = user_input.data.cuda()
hidden = model.init_hidden(1)

success = 0
error = 0
wpa = 0

result = []

for i in range(args.words):
    output, hidden = model(user_input, hidden)
    word_weights = model.decoder(output).squeeze().data.div(args.temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    user_input.data.fill_(word_idx)
    word = corpus.dictionary.idx2word.get([word_idx])
    if word is None or word == '<eos>':
        break
    result.append(word)

print(','.join(result))