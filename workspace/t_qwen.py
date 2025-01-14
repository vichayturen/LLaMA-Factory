from transformers import Qwen2ForCausalLM, Qwen2Config

config = Qwen2Config(
    hidden_size=1024,
    num_hidden_layers=2
)
model = Qwen2ForCausalLM(config)
print(model)
