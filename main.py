import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "meta-llama/Llama-2-7b-hf"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用低精度推理
    device_map="auto"  # 自动分配设备 (CPU 或 MPS)
)

# 检查 MPS (Metal 后端) 是否可用
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# 输入测试
input_text = "What is the meaning of life?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 模型推理
with torch.no_grad():  # 禁用梯度计算
    outputs = model.generate(inputs["input_ids"], max_length=50)

# 解码输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)