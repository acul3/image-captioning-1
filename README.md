# vit-gpt2

Sample run:

```
python run_vit_gpt2.py \
    --output_dir="testing" \
    --data_dir="./" \
    --train_file="data/train.tsv" \
    --validation_file="data/val.tsv" \
    --do_train --do_eval \
    --num_train_epochs="8" --max_seq_length 256 \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="32" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 0 \
```

[Model code](https://github.com/ydshieh/vit-gpt2)