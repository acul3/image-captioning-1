import argparse
import csv
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import gc
import itertools
import jax
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import FlaxMarianMTModel, MarianTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--tsv_path", type=str, default="images-list-clean.tsv", help="path of directory where the dataset is stored")
parser.add_argument("--val_split", type=int, default=0.1, help="Size of validation Subset")
parser.add_argument("--lang_list", nargs="+", default=["id"], help="Language list (apart from English)")
parser.add_argument("--save_location_train", type=str, default=".", help="path of directory where the train dataset will be stored")
parser.add_argument("--save_location_val", type=str, default=".", help="path of directory where the validation dataset will be stored")
parser.add_argument("--is_train", type=int, default=0, help="train or validate")

args = parser.parse_args()




DATASET_PATH = args.tsv_path
VAL_SPLIT = args.val_split
LANG_LIST = args.lang_list
if args.save_location_train != None:
    SAVE_TRAIN = args.save_location_train
    SAVE_VAL = args.save_location_val

BATCH_SIZE = 24
IS_TRAIN = args.is_train
num_devices = 8
lang_dict = {
    "id" : "id_XX"
}

model_id = FlaxMarianMTModel.from_pretrained("Wikidepia/marian-nmt-enid", from_pt=True)

tokenizer = MarianTokenizer.from_pretrained("Wikidepia/marian-nmt-enid", source_lang="en")

def generateid_XX(params, batch):
      output_ids = model_id.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, num_beams=4, max_length=64).sequences
      return output_ids

# def generateru_RU(params, batch, rng):
#       output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], prng_key=rng, params=params, forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"]).sequences
#       return output_ids

p_generate_id_XX = jax.pmap(generateid_XX, "batch")
# p_generate_ru_RU = jax.pmap(generateru_RU, "batch")

map_name = {
    "id_XX": p_generate_id_XX,
    # "ru_RU": p_generate_ru_RU,
}

map_model_params = {
    "id": replicate(model_id.params),
    # "ru_RU": p_generate_ru_RU,
}

def run_generate(input_str, p_generate, p_params):
    inputs = tokenizer(input_str, return_tensors="jax", padding="max_length", truncation=True, max_length=64)
    #inputs = tokenizer([input_str for i in range(num_devices)], return_tensors="jax", padding="max_length", truncation=True, max_length=64)
    p_inputs = shard(inputs.data)
    #output_ids = p_generate(p_params, p_inputs)
    #output_strings = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True, max_length=64)
    output_ids = p_generate(p_params, p_inputs)
    output_strings = tokenizer.batch_decode(output_ids.reshape(-1,64), skip_special_tokens=True, max_length=64)
    return output_strings

def read_tsv_file(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t", index_col=False,names=["caption", "url"])
    print("Number of Examples:", df.shape[0], "for", tsv_path)
    return df

def arrange_data(captions, image_urls):  # iterates through all the captions and save there translations
      lis_ = []
      for caption, image_url in zip(captions, image_urls):  # add english caption first
          p_params = replicate(model_id.params)
          p_generate = p_generate_id_XX
          output = run_generate(caption, p_generate, p_params)
          lis_.append({"caption":output[0], "url":image_url, "lang_id": "id"})
          gc.collect()
      return lis_


_df = read_tsv_file(DATASET_PATH)
train_df, val_df = train_test_split(_df, test_size=VAL_SPLIT, random_state=1234)

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print("\n train/val dataset created. beginning translation")


df = train_df
output_file_name = os.path.join(SAVE_VAL, "train_file.tsv")
with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
    writer = csv.writer(outtsv, delimiter='\t')
    writer.writerow(["caption", "url", "lang_id"])

df = val_df
output_file_name = os.path.join(SAVE_VAL, "val_file.tsv")
with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
    writer = csv.writer(outtsv, delimiter='\t')
    writer.writerow(["caption", "url", "lang_id"])

for i in tqdm(range(0,len(df),BATCH_SIZE)):
    output_batch = arrange_data(list(df["caption"])[i:i+BATCH_SIZE], list(df["url"])[i:i+BATCH_SIZE])
    with open(output_file_name, "a", newline='') as f:
      writer = csv.DictWriter(f, fieldnames=["caption", "url", "lang_id"], delimiter='\t')
      for batch in output_batch:
          writer.writerow(batch)