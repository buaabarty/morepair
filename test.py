from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer

model_path = 'output_model/codellama_merged'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "[INST] This is an incorrect code(TEST.java):\n```java\npublic Main {\n}\n```\nYou are a software engineer. Can you repair the incorrect code?\n[/INST]\n```java\n"

output = pipe(prompt, min_length=128, max_length=2048, temperature=1.0, do_sample=True)

print(output[0]['generated_text'])