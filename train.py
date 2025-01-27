from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, Dataset, concatenate_datasets
import os
import yaml
import utils
import pathlib
from huggingface_hub import HfApi
import os
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


def train(config):    
    
    print(f"#### Starting a fine tuning run with the following configuration:\n\n{config}")

    wandb_config = config["wandb_config"]

    model_config = config["model_config"]

    lora_config = config["lora_config"]

    sft_config = config["sft_config"]
 
    #####################
    # Load model and tokenizer
    #####################

    max_seq_length = model_config["max_seq_length"] # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
    model_id = model_config["model_id"]

    ### Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit)

    # defaults to using chatml as per https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py
    tokenizer = get_chat_template(
        tokenizer
    )

    #####################
    # Load datasets
    #####################

    # Load dataset1
    dataset_id1 = "ThatsGroes/synthetic-dialog-summaries-processed-clean-chatml" #"ThatsGroes/synthetic-dialog-summaries-processed-clean"
    dataset_id2 = ("HuggingFaceTB/smoltalk", "smol-summarize")
    col_to_process = "messages" # the name of the column in the dataset containing the actual training data

    dataset = load_dataset(dataset_id1) #dataset.map(formatting_prompts_func, batched = True, fn_kwargs={"tokenizer": tokenizer, "col_to_process": col_to_process})

    # Load dataset2
    smoltalk = load_dataset(dataset_id2[0], dataset_id2[1])

    # Convert dataset2 to chatml format using the built in method `map` and our own `formatting_prompts_func` and pass arguments using `fn_kwargs`
    smoltalk = smoltalk.map(utils.formatting_prompts_func, batched = True, fn_kwargs={"tokenizer": tokenizer, "col_to_process": col_to_process})

    # Concat datasets to training data
    training_dataset = concatenate_datasets(
        [
            dataset["train"],
            smoltalk["train"]
        ]
        )

    # Create evaluation data
    eval_dataset = concatenate_datasets(
        [
            dataset["test"],
            smoltalk["test"]
        ]
        )

    """# LoRA fine tuning

    Above, we loaded the model and tokenizer and prepared the dataset. Now, we will fine tune the model on the dataset using Low Rank Adaptation (LoRA) fine tuning. To do so, we must first configure the model for LoRA fine tuning using the `get_peft_model` method. As a side note, the `peft` part of the method's name is short for "parameter efficient fine tuning" which is an umbrella term that denotes a family of techniques for fine tuning models using fewer parameters than full fine tuning. LoRA falls under the PEFT family of fine tuning techniques.

    Maxime Labonne gives a gracefully brief description of the LoRA technique:

    > "Low-Rank Adaptation (LoRA) is a popular parameter-efficient fine-tuning technique. Instead of retraining the entire model, it freezes the weights and introduces small adapters (low-rank matrices) at each targeted layer. This allows LoRA to train a number of parameters that is drastically lower than full fine-tuning (less than 1%), reducing both memory usage and training time. This method is non-destructive since the original parameters are frozen, and adapters can then be switched or combined at will." (Source: https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html)

    Let's dive a little deeper into the matrix math behind LoRA.

    LoRA creates an "adaptor matrix" denoted `delta_W` for each layer that you want to target in the original model, denoted `W_original`. Each `W_original` has the following dimensions: 𝑚 × 𝑟 (m rows and n columns).

    For each layer, LoRA creates the adaptor matrix `delta_W` = A x B.

    - A has dimensions
    𝑚
    ×
    𝑟
    (a skinny, tall matrix).
    - 𝐵 has dimensions
    𝑟
    ×
    𝑛
    (a short, wide matrix).
    - 𝑟 is much smaller than both
    𝑚
    and
    𝑛
    , which keeps things efficient.

    The weights of the new, LoRA fine tuned model is then `W_new  = W_original x delta_W`. During fine tuning, only the weights in `A` and `B` updated.

    the "addition" between `W_orginal` and `delta_W` happens element wise
    𝑊
    original
    W
    original
    ​
    and
    Δ
    𝑊
    ΔW happens element-wise. This means that each entry in
    𝑊
    new
    W
    new
    ​
    is the sum of the corresponding entries in
    𝑊
    original
    W
    original
    ​
    and
    Δ
    𝑊
    ΔW.

    After you have trained the LoRA adaptors, you can merge them with the original weights. If you do so, the fine tuned model will have the same number of paramters as the original.

    ## LoRA arguments
    As you can see below, Unsloth's LoRA implementation takes many arguments. Here, I will briefly describe those I deem most important.

    ### `r`
    Recall the `r`from before that we used to described the size of the matrices that make up `delta_W` ? That's what the `r` - aka "rank" - argument here controls. So, the rank, `r`, denotes the size of the low-rank matrices. A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method. The rank can be thought of as a trade-off:

    - Higher Rank: More capacity to capture complex relationships and potentially better performance, but at the cost of increased parameters and training complexity.

    - Lower Rank: Fewer parameters, faster training, and less risk of overfitting, but it may limit the model’s ability to fully adapt to the new task.

    In practice, people seem to mostly use either of the following rank values:

    `[8, 16, 32, 64, 128, 256]`

    It [has been observed](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2) that increasing from 8 to 16 did not significantly improve performance , but I have also seen large values used in practice, for instance to fine tune a [Finnish LLM](https://huggingface.co/Finnish-NLP/Ahma-7B-Instruct).

    ### `target_modules`
    This argument determines which layers (aka modules) in the model to train adaptor matrices for. If this is specified, adaptors will only be made for the modules with the specified names. A widespread practice seems to be to target the following modules ([Source](https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html), [Source](https://huggingface.co/Finnish-NLP/Ahma-7B-Instruct)):

    `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

    ### `alpha`
    Remember when we said that `W_original = A X B`? Actually, it is
    ΔW=
    α
    ​
    (A×B)
    where α is a so-called scaling factor for updates. Alpha directly impacts the contribution of the adators. The larger `alpha` is, the more the LoRA adaptors will dominate the resulting model. If the original models needs a lot of adaptation for the new task that you're fine tuning for, you might want to set `alpha` >>1.  

    It seems that in practice, `alpha` is often set to 1x or 2x the rank value. (https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html). It is suggested to not treat the alpha parameter as a tunable hyperparameter (https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)

    ### `use_rslora`
    RSLoRA means Rank-Stabilized LoRA. I suggest to set this to true when using higher values for rank as per (https://mlabonne.github.io/blog/posts/2024-07-29_Finetune_Llama31.html)
    """


    #####################
    # Set up LoRA
    #####################

    rank = lora_config["r"]
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
    use_rslora = True
    lora_alpha = lora_config["lora_alpha"]


    model = FastLanguageModel.get_peft_model(
        model,
        r = rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 # A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method
        target_modules = target_modules,
        lora_alpha = lora_alpha, # standard practice seems to be to set this to 16. Mlabonne says its usually set to 1-2x the rank
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = use_rslora,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    wandb.login()
    os.environ["WANDB_PROJECT"]= wandb_config["wandb_project"] #"llm_dialog_summarizer"
    os.environ["WANDB_LOG_MODEL"] = "end"

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = training_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field = sft_config["dataset_text_field"], # the key of the dict returned in formatting_prompts_func()
        max_seq_length = max_seq_length,
        dataset_num_proc = 4, # Number of processes to use for processing the dataset. Only used when packing = False
        packing = False, # Can make training 5x faster for short sequences.
        eval_strategy = "steps",
        eval_steps = 100,
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 2,
            num_train_epochs = sft_config["epochs"], # Set this for 1 full training run. OpenAI's default is 4 https://community.openai.com/t/how-many-epochs-for-fine-tunes/7027/5
            learning_rate = sft_config["learning_rate"],
            warmup_steps=sft_config["warmup_steps"],
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 100,
            optim = sft_config["optimizer"],
            weight_decay = sft_config["weight_decay"],
            lr_scheduler_type = sft_config["lr_scheduler_type"], # linear or cosine are most common
            seed = sft_config["seed"],
            output_dir = "outputs",
            report_to="wandb",
            run_name = f"Baseline-{model_id}",
            save_steps=10000
        ),
    )

    trainer_stats = trainer.train()

    print(f"\nTrainer stats:\n {trainer_stats}\n")

    save_suffix = "-summarizer"
    hf_user = "ThatsGroes"

    new_model_id = f"{hf_user}/{model_id.split('/')[-1]}{save_suffix}"
    model.push_to_hub_merged(new_model_id, tokenizer, save_method="merged_16bit")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=config["config_path"],
        path_in_repo="fine_tuning_configuration.yml",
        repo_id=new_model_id,
        repo_type="model",
    )
