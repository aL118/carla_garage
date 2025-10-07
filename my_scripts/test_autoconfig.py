import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
from transformers import AutoConfig

auto_config = AutoConfig.from_pretrained('/fs/nexus-scratch/aliu1237/carla_garage/bert-medium-config.json')
print(auto_config)
print(f"Hidden size: {auto_config.hidden_size}")
print(f"Number of layers: {auto_config.num_hidden_layers}")
print(f"Number of attention heads: {auto_config.num_attention_heads}")