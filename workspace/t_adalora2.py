from peft import AutoPeftModelForCausalLM, AdaLoraModel
from transformers import AutoTokenizer, Qwen2Tokenizer
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

font_name = "simhei"
matplotlib.rcParams['font.family'] = font_name  # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号，防止变成方框



model_name_or_path = "/media/user/备份/wkyc_article/LLaMA-Factory/saves/Qwen2.5-0.5B-Instruct/lora/train_2024-12-23-17-00-21_adalora1000e"
model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path).to("cuda")
tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model.eval()
print(model)
# prompt = tokenizer.apply_chat_template(
#     [{"role": "user", "content": "你的名字是什么？"}],
#     add_special_tokens=True,
#     tokenize=False
# )
# inputs = tokenizer(prompt, return_tensors="pt")
#
# outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])

result = {}
result2 = {}
for name, param in model.named_parameters():
    if "ranknum" in name:
        name_sp = name.split(".")
        layer_num = int(name_sp[4])
        module_name = name_sp[6]
        result[module_name] = result.get(module_name, {})
        result[module_name][layer_num] = float(param.data[0])
    elif "lora_A" in name:
        name_sp = name.split(".")
        layer_num = int(name_sp[4])
        module_name = name_sp[6]
        result2[module_name] = result2.get(module_name, {})
        result2[module_name][layer_num] = int(param.data.size(0))
df2 = pd.DataFrame(result2)
plt.figure()
for name, data in result2.items():
    plt.plot(df2[name], label=name)
# plt.title("不同层各模块AdaLoRA秩数")
plt.xlabel("层序号")
plt.ylabel("秩")
plt.legend()
plt.savefig("./ranknum.png")
plt.show()
df2.to_csv("./ranknum.csv", index=False)
print(df2.sum())
print(df2.sum().sum())
