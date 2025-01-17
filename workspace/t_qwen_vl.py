
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


model_dir = "/media/user/备份/models/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)
print(model)
print(sum(p.numel() for p in model.parameters()))

# default processer
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(model_dir, max_pixels=max_pixels)

def get_inputs(messages):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs

def get_outputs(ids):
    return processor.batch_decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

inputs = get_inputs([
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "/home/user/图片/output.png",
            # },
            {
                "type": "image",
                "image": "/home/user/下载/20250116-151203.jpg",
            },
            # {
            #     "type": "image",
            #     "image": "/home/user/下载/20250116-212347.jpg",
            # },
            {"type": "text", "text": "图中这位是不动游星，你认识他吗？"},
        ],
    }
])

print(inputs)
print(get_outputs(inputs["input_ids"])[0])

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
