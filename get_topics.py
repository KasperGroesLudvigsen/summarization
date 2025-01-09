from datasets import load_dataset, concatenate_datasets
import pandas as pd


def get_topics():
    sum = load_dataset("knkarthick/dialogsum")
    sum = sum["train"]["topic"]
    sum = list(set(sum))
    wiki = pd.read_csv("wiki_views/all_wiki_views.csv")
    wiki = wiki["article"].to_list()
    sum.extend(wiki)

    extra_topics = pd.read_csv("extra_topics.csv")["topic"].to_list()
    
    # upsample extra topics
    for i in range(15):
        sum.extend(extra_topics)

    return sum

