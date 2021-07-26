import sys
import os
import json

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from code.config import cfg, cfg_from_file
from code.mac import MACNetwork
from code.datasets import ClevrDataset, collate_fn
from code.utils import load_vocab

import argparse

import sys
import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm

def load_model(cpfile, cfg_file):
    cpdata = torch.load(cpfile, map_location=torch.device('cpu'))

    cfg_from_file(cfg_file)

    vocab = load_vocab(cfg)

    model = MACNetwork(cfg, cfg.TRAIN.MAX_STEPS, vocab)
    model.load_state_dict(cpdata['model'])

    if cfg.CUDA:
        model = model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    if cpdata['optim'] is not None:
        optimizer.load_state_dict(cpdata['optim'])        
    return model, vocab, optimizer


def evaluate(args):
    model, vocab, _ = load_model(args.checkpoint_file, args.cfg_file)
    model.eval()
    if cfg.CUDA:
        model = model.cuda()

    dataset_test = ClevrDataset(data_dir=args.data_dir, split='test')
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=64, drop_last=False,
                                    shuffle=False, num_workers=0, collate_fn=collate_fn)

    correct = {}
    total_correct = {}

    t = tqdm(dataloader_test)
    for idx, data in enumerate(t, 0):
        # load in new sample
        image, question, question_len, answer, family = data['image'], data['question'], data['question_length'], data['answer'], data['family']
        answer = answer.long()
        question = Variable(question)
        answer = Variable(answer)
        if cfg.CUDA:
            image = image.cuda()
            question = question.cuda()
            answer = answer.cuda().squeeze()
        
        # eval
        with torch.no_grad():
            scores = model(image, question, question_len)  

        # get prediction
        preds = scores.detach().argmax(1)        
        preds = preds.cpu().numpy()
        answer = answer.cpu().numpy()
        
        # create dictionary of scores ifo template type
        for i, (p, gt) in enumerate(zip(preds, answer)):
            if family[i] not in correct:
                correct[family[i]] = 0
                total_correct[family[i]] = 0
            correct[family[i]] += 1 if p==gt else 0 # if prediction == ground truth add 1
            total_correct[family[i]] += 1 # update amount of seen by 1
        
        if idx == 200:
            break

    # calculate accuracy ifo template
    result = {'template': {}}
    for family in total_correct.keys():
        result['template'][family] = (correct[family] / total_correct[family], correct[family], total_correct[family])
    result['final'] = sum(correct.values())/sum(total_correct.values()) # final accuracy
    
    # dump results
    fname = args.checkpoint_file.split('/')
    outfile = f'{fname[0]}/{fname[1]}/results.json'
    with open(outfile, 'w') as outfile:
        json.dump(result, outfile)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('cfg_file', type=str)
    args = parser.parse_args()
    # evaluate(checkpoint_file = 'models/exp1/exp1_25.pth', data_dir = 'exp1', cfgfile = 'data/clevr_train_mac_exp1.yml')
    evaluate(args)