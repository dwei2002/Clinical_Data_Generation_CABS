from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load base model + adapter
peft_model_path = "/content/llama3-lora-final"
config = PeftConfig.from_pretrained(peft_model_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load adapter
new_model = PeftModel.from_pretrained(base_model, peft_model_path, torch_dtype=torch.float16)
new_model = new_model.to("cuda")
new_model = torch.compile(new_model)
new_model.eval()



prompt = "You are a synthetic patient data generator. Your task is to generate virtual ICU patient data.\nIMPORTANT: return only a valid JSON object, with no preamble, no python code.\n\n"


inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = new_model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=True,
        temperature=0.5,
        top_p=0.97,
        pad_token_id=tokenizer.eos_token_id,
    )

synthetic_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
print(synthetic_output)


inputs = tokenizer(prompt, return_tensors="pt").to(new_model.device)

with torch.no_grad():
    outputs = new_model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

synthetic_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(synthetic_output)

print("Prompt:\n", prompt)
print("Output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True).strip())