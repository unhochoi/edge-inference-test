# Import Library
print(f"{'=' * 20} Import Library {'=' * 20}")
import datasets
from transformers import pipeline
from tqdm.auto import tqdm
from time import time
import numpy as np
import pandas as pd

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

# Data extraction in batches
def batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

inputs = dataset[:len(dataset)]["sentence"]

# Create dataframe for metric result
result = pd.DataFrame(columns=['Batch Size', 'Accuracy', 'Model Load Time', 'Data Load Time', 'Total Inference Time', 'Inference Time(avg)', 'IPS', 'IPS(inf)'])

# Inference input data in batches
for batch_size in [1, 2]:
	print(f"{'=' * 20} Batch Size {batch_size} Inference Start {'=' * 20}")
	for sentences in batch(inputs, batch_size):
		start = time()		
		total_pred.extend([*pipe(sentences)])
		end = time()
		inference_time.append(end - start)

	# Calculating Metrics
	print(f"{'=' * 20} Calculating Metrics {'=' * 20}")
	labeling = {'POSITIVE': 1, 'NEGATIVE': 0}
	accuracy = len([1 for pred, real in zip(total_pred, dataset[:len(dataset)]["label"]) if labeling[pred['label']] == real ]) / len(dataset)

	# Append Metric Result to DataFrame
	new_row = {'Batch Size' : batch_size, 'Accuracy' : accuracy, 'Model Load Time' : round(model_load_time, 6), 'Data Load Time' : round(dataset_load_time, 6), 'Total Inference Time' : round(sum(inference_time), 6), 'Inference Time(avg)' : round(sum(inference_time)/ len(dataset), 6), 'IPS' : round(len(dataset)/(model_load_time + dataset_load_time + sum(inference_time)), 6), 'IPS(inf)' : round(len(dataset)/(sum(inference_time)), 6)}
	print(new_row)
	result = result.append(new_row, ignore_index=True)

# Save result csv 
result.to_csv('./result_distilbert_sst2_batch_size_n.csv', index=False)
