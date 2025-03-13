https://huggingface.co/datasets/HWGuncoverAI/riddles
---
dataset_info:
  features:
  - name: query
    dtype: string
  - name: label
    sequence: string
  - name: hint
    dtype: string
  splits:
  - name: train
    num_bytes: 79807
    num_examples: 800
  - name: test
    num_bytes: 20015
    num_examples: 200
  download_size: 30612
  dataset_size: 99822
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
