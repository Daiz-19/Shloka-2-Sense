# Shloka-2-Sense
A Transformer-based Sanskrit-to-English Translation Tool

# Shloka2Sense 🚩
**Sanskrit Shloka to English Translator using mBART**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Model](https://img.shields.io/badge/model-mBART--sanskrit--en-ffca28?logo=huggingface)
![License](https://img.shields.io/badge/license-MIT-green)

**Shloka2Sense** is a deep learning-based machine translation project that converts classical Sanskrit shlokas into English using a fine-tuned multilingual mBART transformer. The project aims to make Indian scriptures more accessible and understandable to non-Sanskrit readers around the world.


## ✨ Features
- 🔄 Translation of Sanskrit Shlokas to English
- 🤖 Fine-tuned multilingual mBART model
- 🧠 Hugging Face integration
- ⚡ Real-time translation via Gradio interface
- 📊 Evaluation with BLEU score
- 📚 Trained on 93,000+ Sanskrit-English sentence pairs

---

## 📊 BLEU Score
**2.7262**  
> Note: While the BLEU score is relatively low due to domain complexity and scarcity of parallel Sanskrit corpora, the translations often preserve core semantic meaning.

---

## 📁 Dataset
- Dataset: [`rahular/itihasa`](https://huggingface.co/datasets/rahular/itihasa)
- Size: ~93,000 Sanskrit-English sentence pairs
- Preprocessing steps:
  - Cleaning & normalization
  - Sentence alignment
  - Tokenization using `MBart50TokenizerFast`

---

## 🧠 Model

- **Fine-tuned model originally used**: `arshdeepawar/mbart-sanskrit-en` *(now unavailable on Hugging Face)*
- **Base model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Training data**: ~93,000 Sanskrit-English sentence pairs from the *Itihasa* dataset
- **Language codes used**:
  - Source (Sanskrit): `hi_IN` (used as proxy)
  - Target (English): `en_XX`

> 📝 **Note**: At the time of development, the model was available and used successfully for training and inference. However, it is currently inaccessible via the Hugging Face link. You can adapt the code to use other publicly available models such as [`Swamitucats/M2M100_Sanskrit_English`](https://huggingface.co/Swamitucats/M2M100_Sanskrit_English) if needed.

---

## 🚀 Getting Started

### 🔧 Installation
```bash
pip install transformers sentencepiece datasets gradio sacrebleu
```

## ▶️ Run Translation
```bash
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Load model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("arshdeepawar/mbart-sanskrit-en")
tokenizer = MBart50TokenizerFast.from_pretrained("arshdeepawar/mbart-sanskrit-en")
tokenizer.src_lang = "hi_IN"

# Translation function
def translate_sanskrit_to_english(text):
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_tokens = model.generate(
        **encoded_input,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        max_length=128
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Example
text = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः"
print("Translation:", translate_sanskrit_to_english(text))
```

## 🧪 Run the Gradio Interface
```bash
import gradio as gr

def gradio_translate(text):
    return translate_sanskrit_to_english(text)

iface = gr.Interface(
    fn=gradio_translate,
    inputs=gr.Textbox(lines=3, placeholder="Enter Sanskrit shloka..."),
    outputs="text",
    title="Shloka2Sense",
    description="Translate Sanskrit shlokas into English using a fine-tuned mBART model.",
    theme="default"
)

iface.launch()
```
