# -*- coding: utf-8 -*-
!pip install transformers sentencepiece

pip install -U datasets

from datasets import load_dataset
ds = load_dataset("rahular/itihasa")

import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
import torch

def extract_translation_fields(example):
    return {
        "translation": example["translation"]["sn"],
        "target": example["translation"]["en"]
    }

ds = ds.map(extract_translation_fields)
# removing null values
ds = ds.filter(lambda example: example['target'] is not None and example['target'] != '')

# training mbart model
# model_name = "facebook/mbart-large-50-many-to-many-mmt"
# tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
# model = MBartForConditionalGeneration.from_pretrained(model_name)

# # Set source and target languages
# tokenizer.src_lang = "hi_IN"  # Sanskrit is not directly supported; using Hindi as a proxy
# target_lang = "en_XX"

# linking with hugging face
!pip install -q huggingface_hub

from huggingface_hub import notebook_login
notebook_login()
# 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

# Loading the finetuned model from hugging face
model = MBartForConditionalGeneration.from_pretrained("arshdeepawar/mbart-sanskrit-en")
tokenizer = MBart50TokenizerFast.from_pretrained("arshdeepawar/mbart-sanskrit-en")
tokenizer.src_lang = "hi_IN"  # Use as Sanskrit proxy

def preprocess_function(examples):
    inputs = examples["translation"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# tokenizing the dataset
tokenized_datasets = ds.map(preprocess_function, batched=True)

pip install sacrebleu

import numpy as np
import sacrebleu

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replacing -100 in the labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Post-process
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Computing sacreBLEU
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])

    return {
        "bleu": round(bleu.score, 4),
        "prediction_len": np.mean([len(pred.split()) for pred in decoded_preds])
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart_sanskrit_en",
    resume_from_checkpoint=True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.001,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir='./logs',
    logging_steps=20,
    report_to=None,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # Use validation set
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # <- Added
)

trainer.train()

results = trainer.evaluate()
print(results)

from transformers import MBart50TokenizerFast

def translate_sanskrit_to_english(text, model, tokenizer):
    tokenizer.src_lang = "hi_IN"  # Source is Sanskrit (use Hindi as proxy)
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generating output
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],  # Target is English
        max_length=128
    )

    # Decoding the tokens
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text[0]

# comparing predicted values and ground truth
import pandas as pd

sample_eval =tokenized_datasets["validation"].select(range(5))

results = []
for i in range(5):
    inp = sample_eval['translation'][i]  # Sanskrit
    true = sample_eval['target'][i]  # English (ground-truth)
    pred = translate_sanskrit_to_english(inp, model, tokenizer)
    results.append({"Sanskrit": inp, "Ground Truth": true, "Model Prediction": pred})
df = pd.DataFrame(results)
print(df.to_markdown(index=False))  # or use df.head()

sanskrit_input = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"

translated_output = translate_sanskrit_to_english(sanskrit_input, model, tokenizer)
print("Translation:", translated_output)

sanskrit_input = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि"

translated_output = translate_sanskrit_to_english(sanskrit_input, model, tokenizer)
print("Translation:", translated_output)

sanskrit_input = "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत अभ्युत्थानम् अधर्मस्य तदात्मानं सृजाम्यहम्"
translation = translate_sanskrit_to_english(sanskrit_input, model, tokenizer)

print("Translation:", translation)

sanskrit_input = "शांताकारं भुजगशयनं पद्मनाभं सुरेशं विश्वाधारं गगनसदृशं मेघवर्णं शुभाङ्गम्॥ लक्ष्मीकान्तं कमलनयनं योगिभिर्ध्यानगम्यम्। वन्दे विष्णुं भवभयहरं सर्वलोकैकनाथम्॥"
translation = translate_sanskrit_to_english(sanskrit_input, model, tokenizer)

print("Translation:", translation)

sanskrit_input = "गुरुर्ब्रह्मा गुरुर्विष्णुः गुरुर्देवो महेश्वरः गुरुः साक्षात् परं ब्रह्म तस्मै श्रीगुरवे नमः"
translation = translate_sanskrit_to_english(sanskrit_input, model, tokenizer)

print("Translation:", translation)

#saving on colab
model_path="/content/my_mbart_model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

from huggingface_hub import create_repo, upload_folder
# saving on hugging face
repo_id = "arshdeepawar/mbart-sanskrit-en-2"
create_repo(repo_id, private=True)

upload_folder(
    repo_id=repo_id,
    folder_path=model_path,
    commit_message="update"
)

"""**Model Inference**"""

!pip install gradio

!pip install transformers sentencepiece
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
import torch

!pip install -q huggingface_hub

from huggingface_hub import notebook_login
notebook_login()
# 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

model = MBartForConditionalGeneration.from_pretrained("arshdeepawar/mbart-sanskrit-en")
tokenizer = MBart50TokenizerFast.from_pretrained("arshdeepawar/mbart-sanskrit-en")
tokenizer.src_lang = "hi_IN"  # Use as Sanskrit proxy

from transformers import MBart50TokenizerFast

def translate_sanskrit_to_english(text, model, tokenizer):
    tokenizer.src_lang = "hi_IN"  # Source is Sanskrit (use Hindi as proxy)
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate output
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],  # Target is English
        max_length=128
    )

    # Decode the tokens
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text[0]

import gradio as gr

def gradio_translate(text):
    return translate_sanskrit_to_english(text, model, tokenizer)

iface = gr.Interface(
    fn=gradio_translate,
    inputs=gr.Textbox(lines=3, placeholder="Enter Sanskrit shloka..."),
    outputs="text",
    title="Shloka2Sense-English explanation of Sanskrit Shloka",
    description="Enter a Sanskrit verse and get its English explanation.",
    theme="default"  # Or use "compact", "soft", etc.
)

iface.launch()
