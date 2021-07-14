from operator import mod
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration
from transformers import ViTFeatureExtractor, GPT2Tokenizer
from torchvision.io import ImageReadMode, read_image
from PIL import Image
import numpy as np
import requests
image = read_image('gambar.jpg', mode=ImageReadMode.RGB)
model = FlaxViTGPT2LMForConditionalGeneration.from_pretrained('flax-community/Image-captioning-Indonesia')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
encoder_inputs = feature_extractor(images=image, return_tensors="jax")
pixel_values = encoder_inputs.pixel_values
model(pixel_values=pixel_values,input_ids=pixel_values)
#print(pixel_values)
#a = model.generate(pixel_values,max_length=128)
#name = 'flax-community/gpt2-small-indonesian'
#tokenizer = GPT2Tokenizer.from_pretrained(name)
#decoder_inputs = tokenizer("seorang pria melewati persimpangan mengendarai sepeda", return_tensors="jax")
#inputs = pixel_values
#logits = model(**inputs)[0]
#preds = np.argmax(logits, axis=-1)
#print('=' * 60)
#print('Flax: Vit + modified GPT2 + LM')
#print(preds)