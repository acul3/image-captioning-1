# vit-gpt2

Sample run:

```
python run_vit_gpt2.py \
    --max_train_samples 5000 \
    --preprocessing_num_workers 0 \
    --predict_with_generate \
    --output_dir="testing_decoder" \
    --data_dir="./" \
    --train_file="data/train.tsv" \
    --validation_file="data/val.tsv" \
    --do_train --do_eval \
    --num_train_epochs="8" --max_seq_length 256 \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
```
python test_eval.py \
    --preprocessing_num_workers 0 \
    --predict_with_generate \
    --output_dir="testing_1023" \
    --data_dir="./" \
    --train_file="data/train.tsv" \
    --validation_file="data/val.tsv" \
    --do_train --do_eval \
    --num_train_epochs="8" --max_seq_length 256 \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \

python run_summarization.py \
    --preprocessing_num_workers 0 \
    --predict_with_generate \
    --output_dir="testing_1023" \
    --dataset_name "adads" \
    --do_train --do_eval \
    --num_train_epochs="8" --max_source_length 256 \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \

python3 main_backup.py \
    --output_dir="test-mic" \
    --logging_steps 400 \
    --eval_steps 400 \
    --save_steps 400 \
    --data_dir="data" \
    --train_file="data/train.tsv" \
    --validation_file="data/val.tsv" \
    --save_total_limit 2 \
    --predict_with_generate \
    --do_eval \
    --do_train \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 512

[Model code](https://github.com/ydshieh/vit-gpt2)