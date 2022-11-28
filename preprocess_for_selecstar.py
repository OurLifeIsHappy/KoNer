import re
import json
from tqdm import tqdm
from glob import glob
from transformers import BertTokenizerFast


def get_labels_dict():
    labels_dict = dict()
    with open('labels_selectstar.txt', 'r') as f:
        for i, label in enumerate(f.readlines()):
            labels_dict[label.replace('\n', '')] = i
    return labels_dict


def main(prefix: str = '셀렉트스타_1차_결과_1107_2cycle_18580', train_ratio=0.7):
    joined_lines = list()
    labels_dict = get_labels_dict()
    tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
    for file_path in tqdm(glob(f'{prefix}/*'), leave=True):
        with open(file_path, 'r') as f:
            json_dict = json.loads(''.join(f.readlines()))
            joined_line = preprocess(tokenizer, json_dict, labels_dict)
            joined_lines.append(joined_line)
    num_train_samples = int(len(joined_lines) * train_ratio)
    with open('select_star_preprocess_train.txt', 'w') as f:
        f.write('\n'.join(joined_lines[:num_train_samples]))
    with open('select_star_preprocess_test.txt', 'w') as f:
        f.write('\n'.join(joined_lines[num_train_samples:]))


def preprocess(tokenizer: BertTokenizerFast, json_dict: dict, labels_dict: dict):
    sentence = json_dict.get('source').get('text')
    units = json_dict.get('units')
    tags = list()
    for unit in units:
        tagged_label = unit.get('classClientId')
        coordinate = unit.get('coordinates')
        start_index = coordinate.get('startIndex')
        end_index = coordinate.get('endIndex')
        tags.append([start_index, end_index, tagged_label])

    tokenized_sentence = tokenizer.encode_plus(
        sentence,
        padding='max_length',
        return_offsets_mapping=True
    )

    labels = list()
    tokens = tokenized_sentence.tokens()
    offsets = tokenized_sentence.get('offset_mapping')
    for token, offset in zip(tokens, offsets):
        label = 'O'
        offset_start = offset[0]
        for tag in tags:
            start_index, end_index, tagged_label = tag
            if start_index <= offset_start <= end_index:
                label = tagged_label
        label = labels_dict.get(label)
        if token == "[PAD]":
            label = -100
        labels.append(str(label))
    joined_line = f'{sentence}\t{" ".join(labels)}'
    return joined_line


if __name__ == "__main__":
    main()
