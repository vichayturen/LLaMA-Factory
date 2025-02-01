
# Load model directly
from transformers import AutoTokenizer, Qwen2ForCausalLM

base_model_name_or_path = "V:\\language_models\\DeepSeek-R1-Distill-Qwen-1.5B\\"
tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
model = Qwen2ForCausalLM.from_pretrained(base_model_name_or_path, device_map="cuda", load_in_8bit=True)

print(model)
