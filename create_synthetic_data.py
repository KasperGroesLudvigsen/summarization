"""
To DO: Calculate and saved tokens per second according to: https://github.com/vllm-project/vllm/issues/4968
"""

from datasets import load_dataset, Dataset
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker
import argparse
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from copy import copy
import torch 
import os
#from get_topics import get_topics
import random 

SAMPLES = 20

models = ["google/gemma-2-27b-it"]

def get_topics():

    data = load_dataset("knkarthick/dialogsum")

    extra = ["Ordinary life", "School life", "Culture and education", "Attitude and emotion", "Relationship", "Tourism", "Health", "Work", "Politics", "Finance", "The economy"]
    topics = list(set(data["train"]["topic"]))
    topics.extend(extra)

    return topics


topics = get_topics()

topics = ["weather forecasts"]

extra_topics = ["transporting goods over long distances", "getting something from the basement", "financial transactions",
                "debt collection", "arranging to meet someone", "arranging a meeting", "setting up a meeting", "setting up a trade", 
                "putting someone in contact with the boss", "introducing someone to your friend", "collecting books", "espionage", "arranging a surprise party",
                "setting up a business meeting", "introducing someone to a business contact", "offering to supply a product", "organized crime", "ilicit activities",
                "weapons manufacturing", "smuggling", "weapon embargo", "chemicals", "homemade explosive devices", "traitors", "the weapon industry",
                "facilitating a business meeting", "potential partnerships", "prostitution", "hiding things", "nuclear weapons", "narcotics", "uncontrolled substances", 
                "making money in dubious ways", "hustling", "russia", "china", "arabic", "arabic names", "activities in the middle east", "geopolitics in west asia",
                "the war in Afghanistan", "the war in Iraq", "the war on terror", "the Bush administration", "Chinese culture", "US and China relations",
                "arabic traditions", "chinese traditions", "russian traditions", "Arabic culture", "Russian culture", "Capitols in middle eastern countries",
                "drug abuse", "Chinese cities", "Russian oligarks", "islam", "christianity", "the bible", "the quran"]



# upsample extra topics
for i in range(15):
    topics.extend(extra_topics)

len(topics)

cs_topics_base = ["getting a refund", "delivery options", "delayed delivery", "return policy", "discounts", "how to return a product", "make an exchange", "cancel a subscription", "renew a subscription", "sign up", "become a member", "opening hours", "making a reservation", "warranty", "buying a giftcard"]
cs_topics = []
for i in range(20):
    cs_topics.extend(cs_topics_base)

lengths = ["100-150 words", "250-300 words", "350-400 words", "600-700 words"]

topics = ["arabic traditions"]

# TODO: Add to prompt that conversation must be realistic with stopped sentences etc. 
def make_prompt() -> dict:
    topic = random.choice(topics)
    #prompt = f"""f{example["instruction"]}: **TEXT:** {example["text"]}"""

    cs_topic = random.choice(cs_topics)

    length = random.choice(lengths)

    prompt_options = [
        f"Please write a text of {length} that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as on long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation.", # Indicate speaker turns like this: '**Speaker1**', '**Speaker2**' and so forth.
        #f"Please write a text of {length} that could pass as a transcription of a telephone conversation between a customer and a customer service representative on the topic of: {cs_topic}. Do not indicate speaker turns. Do not use quotation marks. Just write the transcription as on long text. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation.",
        f"Imagine you walked into a room where a group of people were in the middle of having a conversation on the topic of: {topic}. Write a verbatim transcript of {length} of what they said. Do not indicate speaker turns. Do not use quotation marks. Then, under the headline '**Summary**' write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation."
    ]

    #prompt = f"Please write a text that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns and do not use quotation marks. Just write the transcription as on long text. Then, write one sentence that summarizes the transcription, emphasizing any meetings, persons or places mentioned in the conversation"
    prompt = random.choice(prompt_options)

    return {"prompt": [{"role": "user", "content": prompt}]}

#di = {"prompt": [{"role": "user", "content": prompt}], "topic" : "cs"}

#prompts = [di for i in range(10)]


token = os.getenv("HF_TOKEN") 

prompts = [make_prompt() for i in range(SAMPLES)]

#dataset = dataset.map(make_prompt)

#dataset = dataset.select(20)

dataset = Dataset.from_list(prompts)

all_results = []

energy_use = []

for model in models:

    results = copy(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model, token=token)

    sampling_params = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=2048*2)

    llm = LLM(model=model, max_seq_len_to_capture=8000)

    # Log some GPU stats before we start inference
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(
        f"You're using the {gpu_stats.name} GPU, which has {max_memory:.2f} GB of memory "
        f"in total, of which {start_gpu_memory:.2f}GB has been reserved already."
    )

    tracker = EmissionsTracker()
    tracker.start()
    outputs = llm.chat(dataset["prompt"], sampling_params)
    emissions = tracker.stop()
    print(f"Emissions from generating queries with {model}:\n {emissions}")
    energy_consumption_kwh = tracker._total_energy.kWh  # Total energy in kWh
    print(f"Energy consumption from generating queries with {model}:\n {emissions}")

    responses = [output.outputs[0].text for output in outputs]

    results = results.add_column("response", responses)

    results = results.add_column("model", [model for _ in range(len(results))])
    
    # number of tokens in the prompt and response. Used for calculcating kwh/token
    results = results.add_column("num_tokens_query", [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]) # [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]

    # each element in results["prompt"] is a list with a dictionary with two keys: "content" and "role"
    results = results.add_column("num_tokens_prompt", [len(tokenizer.encode(text[0]["content"], add_special_tokens=False)) for text in results["prompt"]])

    all_results.append(results)

    # Print some post inference GPU stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_inference = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    inference_percentage = round(used_memory_inference / max_memory * 100, 3)

    energy_use.append({
        "model" : model, 
        "energy_use_kwh" : energy_consumption_kwh, 
        "num_tokens_query" : sum(results["num_tokens_query"]), 
        "num_tokens_prompt" : sum(results["num_tokens_prompt"]),
        "num_tokens_total" : sum(results["num_tokens_query"]) + sum(results["num_tokens_prompt"]),
        "used_memory_inference": used_memory_inference
        })

    print(
        f"We ended up using {used_memory:.2f} GB GPU memory ({used_percentage:.2f}%), "
        f"of which {used_memory_inference:.2f} GB ({inference_percentage:.2f}%) "
        "was used for inference."
    )

    torch.cuda.empty_cache()

    # torch.cuda.empty_cache does not properly free up memory
    del llm 

energy_use = pd.DataFrame.from_records(energy_use)

energy_use["energy_per_token_total"] = energy_use["energy_use_kwh"] / energy_use["num_tokens_total"]

energy_use.to_csv("energy_use_per_model.csv", index=False)

final_dataset = concatenate_datasets(all_results)

print(f"Final dataset: \n {final_dataset}")

final_dataset.to_csv("summaries.csv")

final_dataset.train_test_split(test_size=0.05)

final_dataset.push_to_hub("ThatsGroes/synthetic-dialog-summaries-raw")






# time it took to generate: print(outputs[0].metrics.finished_time - outputs[0].metrics.arrival_time)
