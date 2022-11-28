from transformers import BertForTokenClassification, BertTokenizerFast


model_checkpoints = {
    'klue/bert-base': [BertForTokenClassification, BertTokenizerFast],
    'bert-base-multilingual-cased': [BertForTokenClassification, BertTokenizerFast],
    'monologg/kobert': [BertForTokenClassification, BertTokenizerFast]
}


######이건 왜 이렇게 했을까? 똑같은 모델 인데 이름을 다르게 해서 써놨어