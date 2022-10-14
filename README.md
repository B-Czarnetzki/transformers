# NMT for low-resource languages: fighting catastrophic forgetting in hierarchical transfer learning using data interleaving
This repository contains code for reproducing the experiments in the our report for the Deep Learning for NLP course.
This is a fork of the [Transformers](https://github.com/huggingface/transformers) library implementing some custom training behaviour necessary for our experiments.

# Additions in Fork

## Interleaving

To activate data interleaving pass the Trainer a dataset to the argument 'second_train_dataset'
Additionally you have to set 'num_interleaves' in the Seq2SeqTrainingArguments. This will determine how many sentences
get randomly sampled from the 'second_train_dataset' to be interleaved.

## Training on interleaved data with multilingual target sentences

The M2M100 model by facebook uses the 'forced_bos_token_id' to prior the model decoding on the language it is supposed to be decoded to.
in the Vanilla transformers library one sets this before training. This is not problem for fine-tuning on a singular language.
When performing data interleaving one might have different language target sentences however.
For this reason we sublass the Seq2SeqTrainer to the M2MSeq2SeqTrainer.
This trainer expects the data_input to have an additional column specificing the 'forced_bos_token_id' for every sentence in the data.
You also have to remove the translation column commonly present in hugginface Datasets.
For more details check the processing_function in 'scripts/interleaved_en_az.py'

# Requirements
The code was tested using the environment we provide in 'M2M_env.yml'.
It is highly recommended to work in that environment by installing it using Anaconda.
To do this install Anaconda and simply run:

```bash
conda env create -f M2M_env.yml
```

```bash
conda activate M2M
```

# Running Instructions

To reproduce the experiments in our report you can run the scripts provided in 'scripts_M2M'

To also use the sharded distributed training we utilized you need to have at least 2 GPUs in you enviroment and run:

```bash
python -m torch.distributed.launch --nproc_per_node=2 'your_script.py' --sharded_ddp zero_dp_2
```

To reproduce specific experiments we provide the following scripts.

For inference on the base M2M100 model 'baseline.py'
For fine-tuning the M2M100 model on a single language (in our case Azerbaijani) 'fintune_baseline.py'
For fine-tuning on turkish (training the intermediate model) we provide 'intermediate_hierachical.py'
For fine-tuning the intermediate model on the downstream task (using interleaving) 'downstream_interleaving.py'
For using the M2MSeq2SeqTrainer with interleaving. downstream_interleaving_en_az.py'

Note that you have to change the wandb login and specify the correct checkpoint directories in the training scripts for downstream finetuning.

For comparing translations of different models on some selected sentences we provide 'inspect_sentence_translation.py'