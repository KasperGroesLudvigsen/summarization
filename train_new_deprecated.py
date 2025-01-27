
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, Dataset
from utils import formatting_prompts_func
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from codecarbon import EmissionsTracker

RANDOM_SEED = 90210

max_seq_length = 4*2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
dataset_id = "ThatsGroes/synthetic-dialog-summaries-processed"
col_to_process = "messages" # the name of the column in the dataset containing the actual training data

#dataset = load_dataset(dataset_id, split = "train", stream=True)

### Load model and tokenizer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id, # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

dataset = load_dataset(dataset_id)

#####################################
# Convert dataset to ChatML format #
#####################################

# defaults to using chatml as per https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py  
tokenizer = get_chat_template(
    tokenizer
)

dataset = dataset.map(formatting_prompts_func, batched = True, fn_kwargs={"tokenizer": tokenizer, "col_to_process": col_to_process})

# create training and test splits
dataset = dataset.train_test_split(test_size=0.05)

#####################################
#           Initialize LoRA         #
#####################################
rank = 128
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
lora_alpha = 16
use_rslora = True

PEFT_CONFIGURATION = dict(
    r = rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 # A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method
    target_modules=target_modules,
    lora_alpha = lora_alpha,
    lora_dropout = 0,  # Supports any, but = 0 is optimized
    bias = "none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    use_rslora = use_rslora,  # Support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
    random_state = RANDOM_SEED,
)


model = FastLanguageModel.get_peft_model(model, **PEFT_CONFIGURATION)

#model = FastLanguageModel.get_peft_model(
#    model,
#    r = rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 # A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method   
#    target_modules = target_modules
#    lora_alpha = rank, # standard practice seems to be to set this to 16. Mlabonne says its usually set to 1-2x the rank
#    lora_dropout = 0, # Supports any, but = 0 is optimized
#    bias = "none",    # Supports any, but = "none" is optimized
#    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#    random_state = 3407,
#    use_rslora = use_rslora,  # We support rank stabilized LoRA
#    loftq_config = None, # And LoftQ
#)


#####################################
#           Initialize trainer      #
#####################################

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field = "text", # the key of the dict returned in formatting_prompts_func()
    max_seq_length = max_seq_length,
    dataset_num_proc = 4, # Number of processes to use for processing the dataset. Only used when packing = False
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        num_train_epochs = 1, # Set this for 1 full training run. OpenAI's default is 4 https://community.openai.com/t/how-many-epochs-for-fine-tunes/7027/5
        learning_rate = 3e-4,
        warmup_steps=100,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw",
        weight_decay = 0.01,
        lr_scheduler_type = "linear", # linear or cosine are most common
        seed = RANDOM_SEED,
        output_dir = "outputs",
    ),
)


#####################################
#           Start training          #
#####################################

# Log some GPU stats before we start the finetuning
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(
    f"You're using the {gpu_stats.name} GPU, which has {max_memory:.2f} GB of memory "
    f"in total, of which {start_gpu_memory:.2f}GB has been reserved already."
)

# Run LLM fine tuning and track emissions
tracker = EmissionsTracker()
tracker.start()
try:
    trainer_stats = trainer.train()
finally:
    tracker.stop()

#####################################
#          Save results             #
#####################################

# save adapted model weights locally
trainer.save_model()

# Save to huggingface
try:
    model.push_to_hub_merged(repo, tokenizer, save_method="merged_16bit", token = token)
except Exception as e:
    print(f"Could not push to hub due to error: {e}")


try:
    # Save locally
    save_in_local_folder = "model"
    if not os.path.exists(save_in_local_folder):
        os.makedirs(save_in_local_folder)
    model.save_pretrained_merged(save_in_local_folder, tokenizer, save_method="merged_16bit")
except Exception as e:
    print(f"Could not save locally due to error: {e}")


# Log some post-training GPU statistics
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(
    f"We ended up using {used_memory:.2f} GB GPU memory ({used_percentage:.2f}%), "
    f"of which {used_memory_for_lora:.2f} GB ({lora_percentage:.2f}%) "
    "was used for LoRa."
)

