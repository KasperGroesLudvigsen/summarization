from datasets import load_dataset, concatenate_datasets

def get_topics():

    data = load_dataset("knkarthick/dialogsum")

    return data["train"]["topic"]


