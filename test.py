from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, GaussianBlur
from torchvision.transforms.functional import InterpolationMode
import torch
import numpy as np
from transformers import MarianTokenizer
from flax_clip_vision_marian.modeling_clip_vision_marian import FlaxCLIPVisionMarianForConditionalGeneration
model = FlaxCLIPVisionMarianForConditionalGeneration.from_pretrained('munggok/image-captioning-marian')
from torchvision.io import ImageReadMode, read_image
marian_model_name = 'Helsinki-NLP/opus-mt-en-id'
tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
config = model.config
image_size = config.clip_vision_config.image_size
transforms = torch.nn.Sequential(
                    Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_size),
                    ConvertImageDtype(torch.float),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                )

max_length = 8
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def generate_step(batch):
    output_ids = model.generate(pixel_values, **gen_kwargs)
    token_ids = np.array(output_ids.sequences)[0]
    caption = tokenizer.decode(token_ids)
    return caption

image = read_image('000000039769.jpg', mode=ImageReadMode.RGB)
image = transforms(image)
pixel_values = torch.stack([image]).permute(0, 2, 3, 1).numpy()
#pixel_values = torch.stack([image]).permute(0, 2, 3, 1).numpy()

generated_ids = generate_step(pixel_values)
print(generated_ids)