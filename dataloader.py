import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


# 국립국어원 데이터셋
dataset_file_names = {
    True: "data/NXNE1902008030.json_preprocess.txt",
    False: "data/SXNE1902007240.json_preprocess.txt"
}
# dataset_file_names = {
#     True: "select_star_preprocess_train.txt",
#     False: "select_star_preprocess_test.txt"
# }


class NerDataset(Dataset):
    def __init__(self, is_train, device):
        self.device = device
        self.contents = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": []
        }

        tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
        dataset_file_name = dataset_file_names[is_train]

        with open(dataset_file_name, 'r',encoding = 'utf-8') as f:
            for line in tqdm(f.readlines()):
                line = line.replace('\n', '')
                sentence, label = line.split('\t')
                label = list(map(int, label.split(' ')))
                label = torch.tensor(label, dtype=torch.long)
                inputs = tokenizer.encode_plus(sentence,
                                               return_tensors='pt',
                                               padding='max_length')
                for key, value in inputs.items():
                    value = torch.squeeze(value)
                    self.contents[key].append(value)
                self.contents["labels"].append(label)

    def __len__(self):
        return len(self.contents.get("labels"))

    def __getitem__(self, index):
        return {key: value[index].to(self.device) for key, value in self.contents.items()}


def dataloader(is_train, device, batch_size=16):
    dataset = NerDataset(is_train=is_train, device=device)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True)
