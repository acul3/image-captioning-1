from operator import mod
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration
from transformers import ViTFeatureExtractor, GPT2Tokenizer,CLIPFeatureExtractor
from torchvision.io import ImageReadMode, read_image
from PIL import Image
import numpy as np
import requests
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration

# Vit - as encoder
from transformers import ViTFeatureExtractor
from PIL import Image
import requests
import torch
import numpy as np
import jax
# GPT2 / GPT2LM - as decoder
from transformers import ViTFeatureExtractor, GPT2Tokenizer, MarianTokenizer
from flax_clip_vision_marian.modeling_clip_vision_marian import FlaxCLIPVisionMarianForConditionalGeneration
#flax_vit_gpt2_lm = FlaxViTGPT2LMForConditionalGeneration.from_pretrained('munggok/image-captioning')
flax_marian_clip = FlaxCLIPVisionMarianForConditionalGeneration.from_pretrained('munggok/image-captioning-marian',)
from torchvision.io import ImageReadMode, read_image
gpt2_model_name = 'Helsinki-NLP/opus-mt-en-id'
tokenizer = MarianTokenizer.from_pretrained(gpt2_model_name)

max_length = 64
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def generate_step(batch):
    output_ids = flax_marian_clip.generate(batch["pixel_values"], **gen_kwargs)
    return output_ids.sequences

p_generate_step = jax.pmap(generate_step, "batch")
image = read_image('000000039769.jpg', mode=ImageReadMode.RGB)
    # batch dim is added automatically

#encoder_inputs = feature_extractor(images=image, return_tensors="jax")
pixel_values = torch.stack([image]).permute(0, 2, 3, 1).numpy()

# generation
batch = {'pixel_values': pixel_values}
flax_marian_clip(pixel_values=pixel_values)