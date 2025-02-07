
# Load model directly
from transformers import AutoTokenizer, Qwen2Tokenizer, Qwen2ForCausalLM

base_model_name_or_path = "V:\\language_models\\DeepSeek-R1-Distill-Qwen-1.5B\\"
tokenizer_deepseek = AutoTokenizer.from_pretrained(r"V:\language_models\DeepSeek-R1")
tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
model = Qwen2ForCausalLM.from_pretrained(base_model_name_or_path, device_map="cuda", load_in_8bit=True)

prompt = "你是不是陈小慧？"
input_str = tokenizer_deepseek.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
print(input_str)
inputs = tokenizer(input_str, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
