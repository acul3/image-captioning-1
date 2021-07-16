import csv
import json
import os

import datasets
import pandas as pd
import numpy as np
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from PIL import Image
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import Dataset, load_dataset, load_metric
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    HfArgumentParser,
    TrainingArguments,
    is_tensorboard_available,
)
from transformers.file_utils import is_offline_mode

from transformers import ViTFeatureExtractor, GPT2Tokenizer, GPT2Config
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration

logger = logging.getLogger(__name__)

ds = datasets.load_dataset('wit_dataset_script.py', data_dir='data')
train_dataset = ds['train']
image_file_column = 'image_file'
caption_column = 'caption'
pixels_file_column = 'pixels_file'



vit_name_path = 'google/vit-base-patch16-224-in21k'
gpt2_name_path = 'flax-community/gpt2-small-indonesian'

gpt2_config = GPT2Config.from_pretrained(gpt2_name_path)
gpt2_config.add_cross_attention = True


vit_gpt2_name_path = ''

feature_extractor = ViTFeatureExtractor.from_pretrained(vit_name_path)

tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name_path,pad_token="<PAD>")

if not vit_gpt2_name_path:
    assert vit_name_path
    assert gpt2_name_path
    vit_gpt2_model = FlaxViTGPT2LMForConditionalGeneration.from_vit_gpt2_pretrained(
        vit_name_path, gpt2_name_path
    )
else:
    vit_gpt2_model = FlaxViTGPT2LMForConditionalGeneration.from_pretrained(
        vit_gpt2_name_path
    )

model = vit_gpt2_model
model.config.is_encoder_decoder = True
model.config.decoder_start_token_id = gpt2_config.bos_token_id
model.config.bos_token_id = gpt2_config.bos_token_id
model.config.eos_token_id = gpt2_config.eos_token_id
model.config.pad_token_id = 1

model_module = __import__(vit_gpt2_model.__module__, fromlist=["shift_tokens_right"])
shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")


def preprocess_function(examples):

    _pixel_values = []
    for y in examples[image_file_column]:
        with Image.open(y) as image:
            encoder_inputs = feature_extractor(images=image, return_tensors="np")
            x = encoder_inputs.pixel_values
            _pixel_values.append(x)
    pixel_values = np.concatenate(_pixel_values)

    targets = examples[caption_column]

    # Add eos_token!!
    targets = [x + ' ' + tokenizer.eos_token for x in targets]

    model_inputs = {}
    model_inputs['pixel_values'] = pixel_values

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=256, padding="max_length", truncation=True, return_tensors="np"
        )

    model_inputs["labels"] = labels["input_ids"]

    #print(labels["input_ids"])
    #print(gpt2_config.pad_token_id)
    #rint(gpt2_config.bos_token_id)

    decoder_input_ids = shift_tokens_right_fn(
        jnp.array(labels["input_ids"]), 1, gpt2_config.bos_token_id
    )
    model_inputs["input_ids"] = np.asarray(decoder_input_ids)

    # We need decoder_attention_mask so we can ignore pad tokens from loss
    model_inputs["attention_mask"] = labels["attention_mask"]

    return model_inputs
rng = jax.random.PRNGKey(42)
rng, dropout_rng = jax.random.split(rng)

train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            desc="Running tokenizer on train dataset",
)