import gc
import torch
import numpy as np
from model import model_checkpoints
from dataloader import dataloader
from transformers import AdamW
from torch.cuda import is_available
from tqdm import tqdm
from sklearn.metrics import classification_report
from utils import get_labels_dict  # 국립국어원 라벨
# from preprocess_for_selecstar import get_labels_dict


labels_dict = get_labels_dict()                   # 라벨 목록 텍스트파일 만들기
device = 'cuda' if is_available() else 'cpu'      # cuda 사용 가능여부 확인
special_tokens = [                                ##################################BERT 모델   Input format이 이래야하나?? 어디서 확인하지??
    "[UNK]",
    "[SEP]",
    "[PAD]",
    "[CLS]",
    "[MASK]"
]


def train_eval_ko_ner_model(model_checkpoint, num_epochs=5):   # ko ner 모델 학습 #######epoch, batch, 학습 돌릴 때 신경써야하는게 뭐야?

    def filter_special_tokens(b_input_ids, tensors):           # special token 제거
        tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
        for input_ids, tensor in zip(b_input_ids, tensors):
            input_tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids))
            filtered_tensor = tensor[
                (input_tokens != special_tokens[0]) &
                (input_tokens != special_tokens[1]) &
                (input_tokens != special_tokens[2]) &
                (input_tokens != special_tokens[3]) &
                (input_tokens != special_tokens[4])
                ]
            yield filtered_tensor

    train_dataloader = dataloader(is_train=True, device=device)   # trainset 불러오기
    eval_dataloader = dataloader(is_train=False, device=device)   # test set 불러오기

    model_class, tokenizer_class = model_checkpoints.get(model_checkpoint)      #json model check point에서 (klue/bert-base)선택
    model = model_class.from_pretrained(model_checkpoint, num_labels=len(labels_dict))   # BertForTokenClassification pretrained 모델 불러와 , input, label 이 input
    model.to(device) #########이건 뭘까 gpu 쓰려고하는거 같은데
    optim = AdamW(model.parameters(), lr=2e-5)  #AdamW가 최적화, local optimum 방지 lr = learning rate

    # Training
    for epoch in range(num_epochs):  ######정해놓은 epoch 수만큼 epoch 돌려
        train_loop = tqdm(train_dataloader, leave=True, desc=f'Epoch : {epoch}')  #tqdm 사용법
        total_loss = 0
        for batch in train_loop: #dataloader 쪼개서 batch를 뿌려
            optim.zero_grad() #gradinant 초기화
            outputs = model(**batch)  # 모델에 1 batch 입력
            loss = outputs.loss  #loss 계산
            loss.backward() #backpropagation
            optim.step()#parameter 업데이트 
            loss_val = round(loss.item(), 3)#반올림 + 소숫점 3째자리 반올림 - 1000의 자리 반올림
            total_loss += loss_val#loss 합
            train_loop.set_postfix(loss=loss_val)# loop 다 돌고 나오는 tqdm 메세지
        avg_loss = round(total_loss / len(train_dataloader), 3)# 1 epoch로스평균
        checkpoint = f'{model.__class__.__name__}_epoch_{epoch}_avg_loss_{avg_loss}.pt'
        model.save_pretrained(checkpoint)

    # Evaluation
    with torch.no_grad():
        y_true, y_pred = [], []
        eval_loop = tqdm(eval_dataloader, leave=True, desc=f'Evaluation')
        for batch in eval_loop:
            b_input_ids = batch.get("input_ids")

            # (batch_size, sequence_length)
            labels = batch.get("labels")

            # logits : (batch_size, sequence_length, config.num_labels)
            # probs : (batch_size, sequence_length, config.num_labels)
            # result : (batch_size, sequence_length)
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            preds = filter_special_tokens(b_input_ids, preds)
            labels = filter_special_tokens(b_input_ids, labels)

            for pred, label in zip(preds, labels):
                y_true.extend(label.detach().cpu().numpy()) #tensor 후처리, gpu에 올라가 있는 tensor를 떼서, cpu에 넣고, numpy로 바꿔줌
                y_pred.extend(pred.detach().cpu().numpy())

    report = classification_report(y_true=np.array(y_true),
                                   y_pred=np.array(y_pred))

    with open('report.txt', 'w') as f:
        print(report)
        f.write(report)


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    clear_gpu_memory()
    train_eval_ko_ner_model("klue/bert-base")
