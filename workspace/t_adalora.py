from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, AdaLoraConfig, TaskType
model_name_or_path = "/media/user/备份/models/qwen2.5-0.5b-instruct"
tokenizer_name_or_path = "/media/user/备份/models/qwen2.5-0.5b-instruct"

peft_config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules=["q_proj", "k_proj",  "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,
    init_r=8,
    target_r=9,
    lora_alpha=32,
    lora_dropout=0.1
)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
