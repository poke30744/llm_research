from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

classifier = pipeline("zero-shot-classification", model="Formzu/bert-base-japanese-jsnli")
sequence_to_classify = "この商品は肌の潤いを保ち、シワを減らします。"
candidate_labels = ['化粧品', '食品', '金融サービス', 'テレビ番組']
out = classifier(sequence_to_classify, candidate_labels, hypothesis_template="この例は{}です。")
print(out)