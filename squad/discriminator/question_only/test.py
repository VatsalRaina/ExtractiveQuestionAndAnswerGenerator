#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import time
import datetime

from transformers import ElectraTokenizer, ElectraForSequenceClassification



MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the training batch size')
parser.add_argument('--test_real_data_path', type=str, help='Load path of real training data')
parser.add_argument('--test_fake_data_path', type=str, help='Load path of real training data')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--predictions_save_path', type=str, help='Load path to which predictions will be saved')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    with open(args.test_real_data_path) as f:
        real_data = json.load(f)
    with open(args.test_fake_data_path) as f:
        fake_data = json.load(f)

    test_data = real_data + fake_data
    targets = [0]*len(real_data) + [1]*len(fake_data)
    targets = np.asarray(targets)

    input_ids = []
    token_type_ids = []
    attention_masks = []

    for item in test_data:
        question = item["question"]
        combo = question
        input_encodings_dict = tokenizer(combo, truncation=True, max_length=MAXLEN, padding="max_length")
        inp_ids = input_encodings_dict['input_ids']
        inp_att_msk = input_encodings_dict['attention_mask']
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        attention_masks.append(inp_att_msk)

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)

    ds = TensorDataset(input_ids, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    preds = []

    count = 0
    for inp_id, tok_typ_id, att_msk in dl:
        print(count)
        count+=1
        inp_id, tok_typ_id, att_msk = inp_id.to(device), tok_typ_id.to(device), att_msk.to(device)
        with torch.no_grad():
            outputs = model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)
        curr_preds = outputs[0]
        preds += curr_preds.detach().cpu().numpy().tolist()
    preds = np.asarray(preds)
    np.save(args.predictions_save_path + "logits_all.npy", preds)
    np.save(args.predictions_save_path + "targets_all.npy", targets)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)