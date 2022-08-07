"""
This file performs the following steps:
- Take a set of generated extracted questions and corresponding contexts and prepare them in the form of a json
- Take the SQuAD 1.0 dataset and setup in an equivalent json where there is only 1 question taken per context
- Confirm same number of questions in the generated and fake sets
"""

import argparse
import os
import sys
import json
from transformers import T5Tokenizer
from datasets import load_dataset


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--gen_questions_path', type=str,  help='Specify path to generated questions on training set')
parser.add_argument('--gen_contexts_path', type=str,  help='Specify path to contexts corresponding to the generated questions')
parser.add_argument('--split', type=str,  help='train or validation')
parser.add_argument('--save_dir', type=str,  help='Specify path to save generated jsons')

# Everything will be tokenized and de-tokenized to make consistent
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def organise_data(questions, contexts):
    organised_data = []
    count = 0
    for question, context in zip(questions, contexts):
        count += 1
        print(count, len(questions))
        first_sep_pos = question.find("[SEP]")
        if first_sep_pos == -1:
            qu = question
            answer = None
        else:
            qu = question[:first_sep_pos]
            answer = question[first_sep_pos+6:]

        curr_point = {'question': tokenizer.decode(tokenizer.encode(qu), skip_special_tokens=True, clean_up_tokenization_spaces=True), 'context': context, 'answer':answer}
        # print(curr_point)
        organised_data.append(curr_point)
    return organised_data

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Let's prepare the real data first

    train_data = load_dataset('squad', split=args.split)

    real_data = []
    prev_context = ""

    count = 0
    for ex in train_data:
        if len(ex["answers"]["text"])==0:
            continue
        question, context, answer = ex["question"], ex["context"], ex["answers"]["text"][0]
        if prev_context == context:
            continue
        count+=1
        print(count)
        prev_context = context
        curr_item = {"question": tokenizer.decode(tokenizer.encode(question), skip_special_tokens=True, clean_up_tokenization_spaces=True), "context": context, "answer": tokenizer.decode(tokenizer.encode(answer), skip_special_tokens=True, clean_up_tokenization_spaces=True)}
        real_data.append(curr_item)

    # Now let's prepare the fake data

    with open(args.gen_questions_path, 'r') as f:
        all_gen_questions = [a.rstrip() for a in f.readlines()]
    with open(args.gen_contexts_path, 'r') as f:
        all_contexts = [a.rstrip() for a in f.readlines()]

    fake_data = organise_data(all_gen_questions, all_contexts)

    # make sure formatting of contexts of real and fake match exactly
    real_data_corrected = []
    for real_item, fake_item in zip(real_data, fake_data):
        curr_item = real_item
        curr_item["context"] = fake_item["context"]
        real_data_corrected.append(curr_item)
    real_data = real_data_corrected


    # Check that there is the same amount of data in both files
    if len(fake_data) == len(real_data):
        print("Lengths match :)")
    else:
        print("Something went wrong :(")
        print(len(fake_data))
        print(len(real_data))

    with open(args.save_dir+'real.json', 'w') as f:
        json.dump(real_data, f)
        
    with open(args.save_dir+'fake.json', 'w') as f:
        json.dump(fake_data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)