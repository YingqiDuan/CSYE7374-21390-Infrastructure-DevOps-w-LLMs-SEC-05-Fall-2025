import torch
from transformers import AutoTokenizer
from assignment2.mini_gpt import MiniGPT, MiniGPTConfig

# 1. Load checkpoint
ckpt = torch.load("assignment2/runs/exp01/mini_gpt_best.pt", map_location="cpu")
config = MiniGPTConfig(**ckpt["config"])
model = MiniGPT(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 2. Prepare tokenizer/prompt 
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "Artificial intelligence is"
input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="torch")

# 3. Generate continuation (adjust args as needed)
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=40,
        temperature=1.0,
        top_k=50,
    )

# 4. Decode to text
full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(full_text)
