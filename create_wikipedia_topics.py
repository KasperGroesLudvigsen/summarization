import pandas as pd
from datasets import load_dataset

data_files = {"train": "en/*.json.gz"}
c4_validation = load_dataset("Cohere/wikipedia-22-12", data_files=data_files, split="train")

from datasets import load_dataset
lang = 'en'
data = load_dataset(f"Cohere/wikipedia-22-12", lang, split='train', trust_remote_code=True, streaming=True)

views = []
for row in data:
   views.append(row["views"])

top_n = 1000

views = sorted(views)

threshold = views[-1000]

titles = []
for row in data:
   if row["views"] > threshold:
      titles.append(row["title"])

df = pd.DataFrame({"title" : titles})
df.to_csv("wiki_titles.csv", index=False)

