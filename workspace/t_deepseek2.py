
# Load model directly
from transformers import AutoTokenizer, Qwen2Tokenizer, Qwen2ForCausalLM


# base_model_name_or_path = "V:\\language_models\\DeepSeek-R1-Distill-Qwen-1.5B\\"
# base_model_name_or_path = "/media/user/备份/models/DeepSeek-R1-Distill-Qwen-1.5B"
base_model_name_or_path = "/media/user/备份/models/DeepSeek-R1-Distill-Qwen-7B"
tokenizer_deepseek = AutoTokenizer.from_pretrained("./model_deepseek")
tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
model: Qwen2ForCausalLM = Qwen2ForCausalLM.from_pretrained(
    base_model_name_or_path,
    device_map="auto",
    torch_dtype="auto",
    # load_in_8bit=True
)

# prompt = "“strawberry”中有几个r？"
prompt = "英雄联盟国服峡谷之巅S6第一个王者是谁？"
input_str = tokenizer_deepseek.apply_chat_template(
    [
        {"role": "user", "content": prompt}
    ],
    tools=[
        {
            "name": "get_answer",
            "description": "根据问题，获取答案。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "问题",
                    },
                },
                "required": ["question"],
            },
        }
    ],
    tokenize=False,
    add_generation_prompt=True
)
print(input_str)
inputs = tokenizer(input_str, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=4096)
print(tokenizer.decode(outputs[0]))
