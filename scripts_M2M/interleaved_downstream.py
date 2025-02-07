import os
import transformers 
import numpy as np
#from transformers import M2M100Tokenizer, M2M100Model
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, M2M100Model, M2M100Config
import wandb
import torch

from torch.utils.checkpoint import checkpoint

wandb.login(key="8b579a9be261e9cc153188122605c106eda8322e")


raw_datasets = load_dataset("ted_hrlr", "az_to_en")
metric = load_metric("sacrebleu")
second_datasets = load_dataset("ted_hrlr", "tr_to_en") 

model_checkpoint = "facebook/m2m100_418M"

# get hierarchichal checkpoint trained on intermediate language

checkpoint_dir = "checkpoints/M2M_intermediate_tr_en/"

hierarchichal_checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
source_lang = "az"
target_lang = "en"
second_lang = "tr"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang=source_lang, tgt_lang=target_lang)
model = AutoModelForSeq2SeqLM.from_pretrained(hierarchichal_checkpoint)


# We have to set the model decoding to force the target language as the bos token. 
model.config.forced_bos_token_id = tokenizer.get_lang_id(target_lang)


max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=True)

# Tokenize second (intermediate dataset) 

source_lang = second_lang
target_lang = target_lang

tokenizer.src_lang = source_lang
tokenizer.tgt_lang = target_lang
model.config.forced_bos_token_id = tokenizer.get_lang_id(target_lang) 

second_tokenized_datasets = second_datasets.map(preprocess_function, batched=True, load_from_cache_file=True) 


# Training setup
batch_size = 4
num_epochs = 20
model_name = "M2M_interleaved_20_az_en"
args = Seq2SeqTrainingArguments(
    "checkpoints/"+model_name,
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    optim="adamw_bnb_8bit",
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    log_level="info",
    logging_dir="logging/"+model_name,
    logging_strategy="steps",
    logging_steps=46,
    save_strategy="steps",
    save_steps=46, 
    load_best_model_at_end=True,
    load_last_model_at_end=False,
    metric_for_best_model="eval_downstream_lang_bleu",
    ddp_find_unused_parameters=True,
    fp16=True,
    num_interleaves = 1487
)

# Data collator for dynamic padding

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# setup BLEU evaluation metric

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Initializing Trainer

eval_datasets = {"downstream_lang": tokenized_datasets["validation"], "intermediate_lang": second_tokenized_datasets["validation"]}
#eval_datasets = {"downstream_lang": tokenized_datasets["validation"]}

main_trainset = tokenized_datasets["train"].remove_columns("translation") 
other_trainset = second_tokenized_datasets["train"].remove_columns("translation")
num_interleaves = 1487
#combined_trainset = concatenate_datasets([main_trainset, other_trainset.shuffle(seed=42).select(range(num_interleaves))])

# set dropout

dropout_rate = 0.1
model.config.dropout = dropout_rate
model.config.attention_dropout = dropout_rate


class StopCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            control.should_training_stop = True

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=main_trainset,
    second_train_dataset=other_trainset,
    eval_dataset=eval_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=6)]
)

# start training

trainer.train()
"""
for epoch in range(num_epochs -1):
    if not trainer.args.load_last_model_at_end:  
        break
    print("in training epoch ", epoch +1)

    trainer.train_dataset = concatenate_datasets([main_trainset, other_trainset.shuffle(seed=42).select(range(num_interleaves))])  
    trainer.train()
"""

# Evaluate on dev/test set


print("Evaluating")
results_greedy = trainer.predict(tokenized_datasets["test"], num_beams=1)
dev_results_greedy = trainer.predict(tokenized_datasets["validation"], num_beams=1)
print("greedy")
print("dev BLEU: ", dev_results_greedy[-1]["test_bleu"])
print("test BLEU: ", results_greedy[-1]["test_bleu"])

dev_results_beam_8 = trainer.predict(tokenized_datasets["validation"], num_beams=5)
results_beam_8 = trainer.predict(tokenized_datasets["test"], num_beams=5)
print("beam = 8)")
print("dev BLEU: ", dev_results_beam_8[-1]["test_bleu"])
print("test BLEU: ", results_beam_8[-1]["test_bleu"])


print("\n\nNow on 'tr' datasets")


print("Evaluating")
results_greedy = trainer.predict(second_tokenized_datasets["test"], num_beams=1)
dev_results_greedy = trainer.predict(second_tokenized_datasets["validation"], num_beams=1)
print("greedy")
print("dev BLEU: ", dev_results_greedy[-1]["test_bleu"])
print("test BLEU: ", results_greedy[-1]["test_bleu"])

dev_results_beam_8 = trainer.predict(second_tokenized_datasets["validation"], num_beams=5)
results_beam_8 = trainer.predict(second_tokenized_datasets["test"], num_beams=5)
print("beam = 8)")
print("dev BLEU: ", dev_results_beam_8[-1]["test_bleu"])
print("test BLEU: ", results_beam_8[-1]["test_bleu"])

