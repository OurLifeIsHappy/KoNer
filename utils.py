import json
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizerFast

# model_checkpoint = 'monologg/kobert'
# model_checkpoint = 'bert-base-multilingual-cased'
model_checkpoint = 'klue/bert-base'


def get_labels_dict():
    labels_dict = dict()
    with open('labels.txt', 'r') as f:
        labels = list(map(lambda label: label.replace('\n', ''), f.readlines()))
        for i, label in enumerate(labels):
            labels_dict[label] = i
    return labels_dict


def convert_to_cache(file_name: str):
    result = read_raw_data(file_name)
    content = ''
    for sentence_label in result:
        line = f'{sentence_label.get("sentence")}\t{sentence_label.get("label")}\n'
        print(f"line : {line}")
        content += line
    with open(f'{file_name}_preprocess.txt', 'a',encoding = 'utf-8') as f:
        f.write(content)


def read_raw_data(file_name: str):
    """
    raw data json 으로부터 각 문장과 라벨 반환
    :param file_name:
    :return:
    """
    labels_dict = get_labels_dict()
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
    with open(file_name, 'r',encoding = 'utf-8') as f:
        json_val = ''.join(f.readlines())
        json_dict = json.loads(json_val)

    result = []
    documents = json_dict.get("document")
    # print(f"documents: {documents}")
    split_index_start = 25
    split_index_end = 100 # 2379244
    for document in tqdm(documents[split_index_start:split_index_end], leave=True):
        sentences = document.get("sentence")
        for sentence in sentences:

            sentence_label = []  # 한 문장 내의 토큰들의 라벨
            sentence_form = sentence.get("form")  # 온전한 문장
            named_entities = sentence.get("NE")  # 라벨링된 토큰들

            inputs = tokenizer.encode_plus(sentence_form, return_tensors='pt', return_offsets_mapping=True, padding='max_length')
            offset_mappings = inputs.get("offset_mapping")[0]
            tokens = inputs.tokens()

            for i, offset_mapping in enumerate(offset_mappings):
                token = tokens[i]
                token_label = 'O'
                start_index, end_index = offset_mapping
                for named_entity in named_entities:
                    ne_start_index, ne_end_index = named_entity.get("begin"), named_entity.get("end")
                    if ne_start_index <= start_index < ne_end_index:
                        token_label = named_entity.get("label")
                sentence_label.append(CrossEntropyLoss().ignore_index if token == "[PAD]" else labels_dict.get(token_label))

            result.append({
                "sentence": sentence_form,
                "token": inputs.tokens(),
                "label": ' '.join(map(str, sentence_label))
            })

    return result


if __name__ == "__main__":
    # convert_to_cache('data/NXNE1902008030.json')
    convert_to_cache('data/SXNE1902007240.json')

