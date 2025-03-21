
from transformers import Qwen2ForCausalLM, Qwen2Config, Qwen2Tokenizer


# model_path = r"V:\language_models\Qwen2.5-7B-Instruct"
model_path = r"/media/user/备份/models/Qwen2.5-7B-Instruct"
model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    # load_in_8bit=True
)
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
print(model)
# print(tokenizer.bos_token)

messages = [
    {
        "role": "user",
        "content": "2024年英雄联盟全球总决赛的冠军上路是谁？"
    }
]
tools = [
    {
        "name": "web_search",
        "description": "从网络中搜索答案",
        "arguments": [
            {
                "name": "keyword",
                "description": "搜索的关键词",
                "type": "string",
                "required": True
            }
        ],
    },
]
inputs = tokenizer.apply_chat_template(messages, tools, tokenize=False, add_generation_prompt=True)
print(inputs)
inputs = tokenizer(inputs, return_tensors="pt")
inputs = inputs.to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

