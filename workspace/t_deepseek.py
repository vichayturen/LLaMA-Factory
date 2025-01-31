
from vutils import io
from transformers import AutoTokenizer
from model_deepseek.modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3Config


base_model_dir_path = r"V:\code\202401\LLaMA-Factory\workspace\model_deepseek"
tokenizer = AutoTokenizer.from_pretrained(base_model_dir_path)
config_dict = io.jsonload(r"V:\code\202401\LLaMA-Factory\workspace\model_deepseek\config.json")
config = DeepseekV3Config(
    **config_dict
)

config.num_hidden_layers = 10
config.hidden_size = 526
print(config)
# exit(0)
model = DeepseekV3ForCausalLM(config=config)
print(model)
print("num of parameters:", sum([param.numel() for param in model.parameters()]))
