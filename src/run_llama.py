import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "huggingface/llama"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=100, do_sample=True, top_k=50)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text)

