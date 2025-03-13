---
language:
- en
dataset_info:
  features:
  - name: riddle
    dtype: string
  - name: answer
    dtype: string
  - name: hash
    dtype: string
  splits:
  - name: train
    num_bytes: 73002.886035313
    num_examples: 621
  download_size: 50162
  dataset_size: 73002.886035313
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
