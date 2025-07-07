# 🩺 MedicalChatBot

A medical chatbot fine-tuned with Low-Rank Adaptation (LoRA) on top of [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct).  
This model was trained on a curated set of medical question-answer pairs for research and health education purposes.

---

## 📚 Training Details

- **Base model**: `mistralai/Mistral-7B-Instruct`
- **Fine-tuning method**: PEFT with LoRA
- **Dataset**: [kberta2014/medical-chat-dataset](https://huggingface.co/datasets/kberta2014/medical-chat-dataset)
- **Frameworks**: `transformers`, `peft`, `datasets`, `accelerate`

---

## 🧠 Prompt Format

This model expects prompts in the following format:

```
### Instruction:
What are common symptoms of diabetes?

### Input:


### Response:
```

---

## 💬 Example

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="kberta2014/MedicalChatBot", tokenizer="kberta2014/MedicalChatBot")

prompt = '''### Instruction:
What are the symptoms of high blood pressure?

### Input:


### Response:
'''

output = pipe(prompt, max_new_tokens=200, temperature=0.7)
print(output[0]["generated_text"])
```

---

## 🔗 Gradio Demo

You can also launch a simple chatbot using Gradio:

```python
import gradio as gr
from transformers import pipeline

pipe = pipeline("text-generation", model="kberta2014/MedicalChatBot", tokenizer="kberta2014/MedicalChatBot")

def chat(instruction, input_text=""):
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return pipe(prompt, max_new_tokens=200, temperature=0.7)[0]["generated_text"]

gr.Interface(fn=chat,
             inputs=["text", "text"],
             outputs="text",
             title="🩺 MedicalChatBot",
             description="Ask medical questions and get responses from a fine-tuned LLM"
).launch()
```

---

## ⚠️ Disclaimer

This model is for **educational and research use only**.  
It is **not intended for clinical use** or as a substitute for professional medical advice.

---

## 📤 License

Apache 2.0 (Same as base model)
