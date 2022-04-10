# Import Library
print(f"{'=' * 20} Import Library {'=' * 20}")
import datasets
from transformers import pipeline
from tqdm.auto import tqdm
from time import time

# Load Model
print(f"{'=' * 20} Load Model {'=' * 20}")
start = time()
pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="tf", device=0)     # devices -1 : CPU, 0 : GPU
end = time()
model_load_time = end - start

# Load Dataset
print(f"{'=' * 20} Load Dataset {'=' * 20}")
start = time()
dataset = datasets.load_dataset("glue", "sst2", split='validation')
end = time()
dataset_load_time = end - start

# Inference Result
inference_time = []
total_pred = []

# Data Generator
def data():
    for sentence in dataset[:len(dataset)]["sentence"]:
        yield sentence

# Inference one sentence at a time
print(f"{'=' * 20} Inference Start {'=' * 20}")
for sentence in tqdm(data(), total=len(dataset)):
    start = time()
    total_pred.append(*pipe(sentence))
    end = time()
    inference_time.append(end - start)

# Calculating Metrics
print(f"{'=' * 20} Calculating Metrics {'=' * 20}")
labeling = {'POSITIVE': 1, 'NEGATIVE': 0}
accuracy = len([1 for pred, real in zip(total_pred, dataset[:len(dataset)]["label"]) if labeling[pred['label']] == real ]) / len(dataset)

# Metric Result
print(f"{'=' * 20} Metric Result {'=' * 20}")
print(f"Accuracy = {accuracy}")
print(f"Model Load Time = {model_load_time:.6f}s")
print(f"Dataset Load Time = {dataset_load_time:.6f}s")
print(f"Total Inference Time = {sum(inference_time):.6f}s")
print(f"Inference Time(avg) = {sum(inference_time)/ len(dataset):.6f}s")
print(f"IPS = { len(dataset)/(model_load_time + dataset_load_time + sum(inference_time)) :.6f}")
print(f"IPS(inf) = { len(dataset)/(sum(inference_time)) :.6f}")
