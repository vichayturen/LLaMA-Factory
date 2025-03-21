
import re
import math
from datetime import datetime

from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig


# base_model = "V:\\language_models\\DeepSeek-R1-Distill-Qwen-1.5B"
# base_model = "V:\\language_models\\Qwen2.5-0.5B-Instruct\\"
# base_model = "/media/user/备份/models/DeepSeek-R1-Distill-Qwen-1.5B"
base_model = "/media/user/备份/models/Qwen2.5-1.5B-Instruct"
# train_dataset = "V:\\code\\202401\\LLaMA-Factory\\data\\wkyc\\test_grpo.json"
# train_dataset = "/media/user/备份/wkyc_article/LLaMA-Factory/data/wkyc/test_grpo.json"
train_dataset = "/home/user/下载/train-00000-of-00001.parquet"
output_dir = f"./saves/grpo/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
tokenizer = AutoTokenizer.from_pretrained(base_model)
train_dataset = load_dataset("parquet", data_files=train_dataset)

SYSTEM_PROMPT = """\
你说话带有口头禅“喵”。比如“谢谢喵”、“不好意思喵”。"""
def tokenize_prompt(example: dict) -> dict:
    example["prompt"] = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": example["question_zh-cn"]}],
        tokenize=False,
        add_generation_prompt=True
    )
    return example
train_dataset = train_dataset.map(tokenize_prompt)
# prompt_to_answer_dict = {train_dataset['train'][i]['prompt']: train_dataset['train']['answer_only'] for i in range(train_dataset['train'].num_rows)}


def length_reward_algo(x, alpha):
    x = x / alpha
    return x / (2 * math.exp(x - 1))


def reward_func1(prompts: list, completions: list, **kwargs):
    """猫娘奖励"""
    print("##########################################")
    print(prompts[0])
    print("##########################################")
    print("\n##########################################\n".join(completions))
    rewards = []
    for prompt, completion in zip(prompts, completions):
        if "喵" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_func2(prompts: list, completions: list, **kwargs):
    """思考奖励"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        completion = completion.strip()
        if completion.startswith("<think>") and "</think>" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_func3(prompts: list, completions: list, **kwargs):
    """长度奖励"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        len_prompt = len(prompt)
        thinking = re.match(r"<think>.*</think>", completion.strip(), re.DOTALL)
        if thinking:
            len_thinking = len(thinking.group().replace("<think>", "").replace("</think>", ""))
            len_answering = len(completion[thinking.span()[1]:].strip())
        else:
            len_thinking = 0
            len_answering = len(completion)
        reward = length_reward_algo(len_thinking, 5 * len_prompt) + length_reward_algo(len_answering, len_prompt)
        rewards.append(reward)
    return rewards

def reward_func4(prompts: list, completions: list, **kwargs):
    """格式奖励"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        if "请按照以下格式回答问题：\n```\n选项：【序号】\n```" not in prompt:
            rewards.append(1.0)
        elif re.search(r"```\n选项：[0-9]+\n```", completion, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_func5(prompts: list, completions: list, **kwargs):
    """答案奖励"""
    answer = str(kwargs['answer_only'][0])
    rewards = []
    for completion in completions:
        if answer in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


grpo_args = GRPOConfig(
    output_dir=output_dir,
    model_init_kwargs={
        "torch_dtype": "auto",
        "device_map": "auto",
    },
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    num_generations=2,
    per_device_train_batch_size=2,
    max_prompt_length=256,
    max_completion_length=256,
    save_steps=100,
    logging_steps=1,
    num_train_epochs=100,
)
peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=[reward_func1, reward_func2, reward_func5],
    args=grpo_args,
    train_dataset=train_dataset["train"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
