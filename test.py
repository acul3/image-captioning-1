from torchvision.io import ImageReadMode, read_image
import torch
from transformers import MarianTokenizer,ViTFeatureExtractor,CLIPProcessor
from flax_clip_vision_marian.modeling_clip_vision_marian import FlaxCLIPVisionMarianForConditionalGeneration
flax_marian_clip = FlaxCLIPVisionMarianForConditionalGeneration.from_pretrained('munggok/image-captioning-marian',)
from torchvision.io import ImageReadMode, read_image
marian_model_name = 'Helsinki-NLP/opus-mt-en-id'
tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
clip_model_name = 'openai/clip-vit-base-patch32'
feature_extractor = CLIPProcessor.from_pretrained(clip_model_name)


max_length = 64
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def generate_step(batch):
    output_ids = flax_marian_clip.generate(batch["pixel_values"], **gen_kwargs)
    return output_ids.sequences

image = read_image('000000039769.jpg', mode=ImageReadMode.RGB)

encoder_inputs = feature_extractor(images=image, return_tensors="jax")
pixel_values = encoder_inputs.pixel_values

#pixel_values = torch.stack([image]).permute(0, 2, 3, 1).numpy()


batch = {'pixel_values': pixel_values}
generated_ids = generate_step(batch)