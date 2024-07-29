import transformers
from composer.utils import reproducibility

# Create an Electra masked language modeling model using Hugging Face transformers
# Note: this is just loading the model architecture, and is using randomly initialized weights, so it is important to set
# the random seed here
reproducibility.seed_all(17)
config = transformers.AutoConfig.from_pretrained('google/electra-small-discriminator')
model = transformers.AutoModelForMaskedLM.from_config(config)
tokenizer = transformers.AutoTokenizer.from_pretrained('google/electra-small-discriminator')

import datasets
from torch.utils.data import DataLoader

# Load the AG News dataset from Hugging Face
agnews_dataset = datasets.load_dataset('ag_news')

# Split the dataset randomly into a train and eval set
split_dict = agnews_dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=17)
train_dataset = split_dict['train']
eval_dataset = split_dict['test']

text_column_name = 'text'

# Tokenize the datasets
def tokenize_function(examples):
    # Remove empty lines
    examples[text_column_name] = [
        line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples[text_column_name],
        padding='max_length',
        truncation=True,
        max_length=256,
        return_special_tokens_mask=True,
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[text_column_name, 'label'],
    load_from_cache_file=False,
)
tokenized_eval = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=[text_column_name, 'label'],
    load_from_cache_file=False,
)

# We use the language modeling data collator from Hugging Face which will handle preparing the inputs correctly
# for masked language modeling
collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Create the dataloaders
train_dataloader = DataLoader(tokenized_train, batch_size=64, collate_fn=collator)
eval_dataloader = DataLoader(tokenized_eval, batch_size=64, collate_fn=collator)

from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel

metrics = [
    LanguageCrossEntropy(ignore_index=-100),
    MaskedAccuracy(ignore_index=-100)
]
# Package as a trainer-friendly Composer model
composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler

optimizer = DecoupledAdamW(composer_model.parameters(), lr=1.0e-4, betas=[0.9, 0.98], eps=1.0e-06, weight_decay=1.0e-5)
lr_scheduler = LinearWithWarmupScheduler(t_warmup='250ba', alpha_f=0.02)

import torch
from composer import Trainer

# Create Trainer Object
print("creating Trainer object")
trainer = Trainer(
    model=composer_model, # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep', # train for more epochs to get better performance
    save_folder='checkpoints/pretraining/',
    optimizers=optimizer,
    schedulers=[lr_scheduler],
    device='gpu' if torch.cuda.is_available() else 'cpu',
    # train_subset_num_batches=100, # uncomment this line to only run part of training, which will be faster
    precision='fp32',
    seed=17,
)

# Start training
print("start training")
trainer.fit()
trainer.close()