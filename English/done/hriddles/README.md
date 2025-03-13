---
dataset_info:
  features:
  - name: answer
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 105908
    num_examples: 500
  download_size: 71065
  dataset_size: 105908
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
