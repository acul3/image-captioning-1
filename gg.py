from functools import partial
import gc
import logging
import nltk
import numpy as np
import pandas as pd
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import math
import json
from flax.serialization import to_bytes, from_bytes

import shutil
import torch
from transformers.file_utils import PushToHubMixin
from datasets import load_metric
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, GaussianBlur
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, shard, shard_prng_key, get_metrics, onehot
from flax_clip_vision_marian.modeling_clip_vision_marian import FlaxCLIPVisionMarianForConditionalGeneration
from transformers import MarianTokenizer,MBart50TokenizerFast, HfArgumentParser, TrainingArguments, is_tensorboard_available, set_seed

class Transform(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.transforms = torch.nn.Sequential(
                    Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_size),
                    ConvertImageDtype(torch.float),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x

class ImageTextDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        file_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        max_samples: int = None
    ):
        super().__init__(root, transforms, transform, target_transform)

        # self.captions = []
        # self.image_paths = []
        # self.lang = []

        examples = pd.read_csv(file_path, sep='\t')
        gc.collect()

        self.map_lang_code = {
            "en": "en_XX",
            "de": "de_DE",
            "fr": "fr_XX",
            "es": "es_XX",
        }

        # for idx,img_file in enumerate(examples["image_file"].values):
        #     if os.path.exists(os.path.join(self.root,img_file)):
        #     self.image_paths.append(img_file)
        #     self.captions.append(examples["caption"].values[idx])
        #     self.lang.append(examples["lang_id"].values[idx])

        self.image_paths = examples["image_path"].values
        self.captions = examples["captions"].values

        if max_samples is None:
            max_samples = len(self.image_paths)

        self.image_paths = self.image_paths[:max_samples]
        self.captions = self.captions[:max_samples]

        # with open(file_path, encoding="utf-8") as fd:
        #     examples = csv.DictReader(fd, delimiter="\t", quotechar='"')
        #     for row in examples:
        #         self.image_paths.append(os.path.join(self.root,row["image_file"]))
        #         self.captions.append(row["caption"])
        #         self.lang.append(row["lang_id"])


    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        return read_image(os.path.join(self.root,path), mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target,

    def __len__(self) -> int:
        return len(self.captions)
model = FlaxCLIPVisionMarianForConditionalGeneration.from_pretrained('munggok/image-captioning-marian')
config = model.config
preprocess = Transform(config.clip_vision_config.image_size)
preprocess = torch.jit.script(preprocess)
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-id')

# Initialize the image-text dataset
train_dataset = ImageTextDataset(
    'data',
    'dev.tsv',
    transform=preprocess
)