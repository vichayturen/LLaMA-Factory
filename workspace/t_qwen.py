from transformers import Qwen2ForCausalLM, Qwen2Config, Qwen2Tokenizer

config = Qwen2Config(
    hidden_size=1024,
    num_hidden_layers=2
)
model_path = "/media/user/备份/models/qwen2.5-0.5b-instruct"
model = Qwen2ForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
print(model)
print(tokenizer.bos_token)
