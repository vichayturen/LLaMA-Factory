
from transformers import Qwen2ForCausalLM, Qwen2Config, Qwen2Tokenizer


model_path = r"V:\language_models\Qwen2.5-7B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    # torch_dtype="auto",
    load_in_8bit=True
)
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
print(model)
# print(tokenizer.bos_token)

messages = [
    {
        "role": "user",
        "content": "你好"
    }
]
inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(inputs, return_tensors="pt")
inputs = inputs.to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

