from operator import mod
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration
from transformers import ViTFeatureExtractor, GPT2Tokenizer
from torchvision.io import ImageReadMode, read_image
from PIL import Image
import numpy as np
import requests
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration

# Vit - as encoder
from transformers import ViTFeatureExtractor
from PIL import Image
import requests
import numpy as np

# GPT2 / GPT2LM - as decoder
from transformers import ViTFeatureExtractor, GPT2Tokenizer

flax_vit_gpt2_lm = FlaxViTGPT2LMForConditionalGeneration.from_pretrained('munggok/image-captioning')

vit_model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

gpt2_model_name = 'flax-community/gpt2-small-indonesian'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict(image):

    image = Image.open(requests.get(url, stream=True).raw)
    # batch dim is added automatically
    encoder_inputs = feature_extractor(images=image, return_tensors="jax")
    pixel_values = encoder_inputs.pixel_values

    # generation
    batch = {'pixel_values': pixel_values}
    generation = flax_vit_gpt2_lm.generate(batch['pixel_values'], **gen_kwargs)

    token_ids = np.array(generation.sequences)[0]
    caption = tokenizer.decode(token_ids)

    return caption, token_ids


if __name__ == '__main__':


    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    caption, token_ids = predict(image)

    print(f'token_ids: {token_ids}')
    print(f'caption: {caption}')
