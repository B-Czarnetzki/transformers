import os
import transformers 
import numpy as np
#from transformers import M2M100Tokenizer, M2M100Model
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, M2MSeq2SeqTrainer, TrainerCallback, M2M100Model, M2M100Config
import wandb
import torch

from torch.utils.checkpoint import checkpoint

wandb.login(key="8b579a9be261e9cc153188122605c106eda8322e")


raw_datasets = load_dataset("ted_hrlr", "az_to_en")
metric = load_metric("sacrebleu")
second_datasets = load_dataset("ted_hrlr", "tr_to_en") 

model_checkpoint = "facebook/m2m100_418M"

# get hierarchichal checkpoint trained on intermediate language

checkpoint_dir = "checkpoints/M2M_downstream_az_en/"
checkpoint_dir_2 = "checkpoints/M2M_finetune_az_en/"

hierarchichal_checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
finetune_baseline_checkpoint = os.path.join(checkpoint_dir_2, os.listdir(checkpoint_dir_2)[0])
source_lang = "az"
target_lang = "en"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang=source_lang, tgt_lang=target_lang)
model_hierarchical = AutoModelForSeq2SeqLM.from_pretrained(hierarchichal_checkpoint)
model_baseline = AutoModelForSeq2SeqLM.from_pretrained(finetune_baseline_checkpoint)

# We have to set the model decoding to force the target language as the bos token. 
model_baseline.config.forced_bos_token_id = tokenizer.get_lang_id(target_lang)
model_hierarchical.config.forced_bos_token_id = tokenizer.get_lang_id(target_lang)


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


def translate(sent, model):
    encoding = tokenizer(sent["az"], return_tensors="pt")
    generated_tokens = model.generate(**encoding, forced_bos_token_id=tokenizer.get_lang_id("en"))

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# index list of sentences to inspect
sents_to_inspect = [555, 121, 188, 200, 244, 391, 433, 575, 582, 745]

test_sentences = tokenized_datasets["test"]["translation"]
inspect_list = [test_sentences[i] for i in sents_to_inspect]

li_preds_baseline = []
li_preds_hierarchical = []
li_targets = []

for sent in inspect_list:
     pred_baseline = translate(sent, model_baseline)
     pred_hierarchical = translate(sent, model_hierarchical)
     target = sent["en"]
     li_preds_baseline.append(pred_baseline)
     li_preds_hierarchical.append(pred_hierarchical)
     li_targets.append([target])
     print(f"\nNew sentence: \n")
     print("source: ", sent["az"])
     print("Actual sentence: ", target)
     print("prediction baseline: ", pred_baseline)
     print("prediction hierarchical: ", pred_hierarchical)

print("\n\nDone, Bleu results:\n\n")
result_baseline = metric.compute(predictions=li_preds_baseline, references=li_targets)
result_hierarchical = metric.compute(predictions=li_preds_hierarchical, references=li_targets)
print("BLEU baseline: ", result_baseline)
print("BLEU hierarchical: ", result_hierarchical)