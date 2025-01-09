# How to fine tune - a guide

## Chat templates

ShareGPT style datasets look like this:
```python
{"from": "system", "value": "You are an assistant"}
{"from": "human", "value": "What is 2+2?"}
{"from": "gpt", "value": "It's 4."}
```

Here is in example of what a Huggingface style dataset looks like. When read into a Huggingface dataset using `Dataset.from_dict()`, the below example will have a single column called "messages" with two entries: 
```python
data = {
    "messages": [

        [ { "content": "You are an expert in summarizing texts. Extract and present the main key point of the input text in one short sentence, including essential details like dates, locations, persons and organizations if necessary.", "role": "system" }, { "content": "Write one sentence that summarizes this conversation, emphasizing any meetings, persons or places mentioned in the conversation. \n\n **Conversation:** \n\n So, I'm thinking about trying this new coffee shop downtown. They apparently roast their own beans, and they have a crazy amount of different brewing methods. I was wondering if you'd be interested in going with me sometime this week.I might be free on Thursday afternoon, maybe around 3 p.m.Thursday works for me. I've been wanting to check out that place too.Perfect. How about we meet at the coffee shop directly?Yeah, that sounds good. What do you think about trying something different, like the siphon brewing method?I'm down for anything.Awesome. I heard they also have some really good pastries, so maybe we can grab a treat while we're there.Definitely.Okay, so Thursday at 3 p.m. at the new coffee shop downtown. Sounds good.Oh, and by the way, I was talking to Sarah, and she mentioned she's looking for a good graphic designer for a project she's working on.Really?Yeah, she said it's for a website redesign. That's interesting. I might be able to help her out. I've been doing some freelance design work lately.That's great. Sarah's got a good budget for this project, and she's really easy to work with. I'll put you two in touch.Thanks, I'd love to hear more about it.No problem. Maybe we can talk about it more at the coffee shop on Thursday.Sounds like a plan.Oh, and while we're on the topic of trades, I was thinking maybe you could help me with something too.Sure, what do you need?Remember how I was telling you about that website I'm building for my photography? I could really use some help with the coding. I'm not the best coder, but I know the basics. What kind of help do you need specifically?Well, I'm having trouble with the navigation menu. It's not displaying correctly on mobile devices.Let me take a look at it. Maybe I can figure it out. That would be amazing.Okay, so it's a deal then. You'll connect me with Sarah, and I'll help you with your website.Perfect.Great. I'm looking forward to our coffee date.Me too.", "role": "user" }, { "content": "The conversation discussed a planned meeting at a new coffee shop downtown on Thursday afternoon to discuss potential trade arrangements, including connecting one party with a graphic design project and providing coding assistance for a website.", "role": "assistant" } ],

        [ { "content": "You are an expert in summarizing texts. Extract and present the main key point of the input text in one short sentence, including essential details like dates, locations, persons and organizations if necessary.", "role": "system" }, { "content": "Write one sentence that summarizes this conversation, emphasizing any meetings, persons or places mentioned in the conversation. \n\n **Conversation:** \n\n I saw a documentary on Bohemian Grove the other day, you know that place in California. It's this massive redwood forest, like seriously huge, and they have this secret society that meets there every year. Apparently, it's been going on for over a hundred years, started by some newspaper guys back in the day. They call themselves the Bohemian Club, and it's supposed to be this exclusive club for powerful people, like politicians, CEOs, and celebrities. Every July, they go up to this grove and have this big retreat where they dress up in togas, build effigies, and burn them. They call it the Cremation of Care ceremony. Sounds a little creepy, right?It's supposed to be a place where they can just relax and let loose, away from the pressures of their everyday lives. But there's a lot of speculation about what really goes on up there. Some people think it's just harmless fun, a bunch of guys getting together for a guys' weekend. Others think it's something more sinister, like a place where they plan world domination or something.There are always protesters outside the grove during the meetings, you know, conspiracy theorists and activists. They accuse the club of being elitist and secretive, and they say they're up to no good. I mean, I don't know what to believe. It's all pretty hush-hush, and they don't let the public in.But the fact that it's so secret just makes people more suspicious. They say there are hidden bunkers and underground tunnels, and that they do all sorts of weird rituals. It's kind of like Area 51, you know? Everyone wants to know what's going on behind closed doors.I read somewhere that Richard Nixon used to go there, and Ronald Reagan too. Apparently, they have a giant owl statue that's supposed to be the symbol of the club. They even have a lake there, and supposedly, they have these little boats shaped like owls that they use to get around.It's just fascinating, you know? This little pocket of secrecy in the middle of the redwoods. It's like stepping back in time, a secret world that most people will never get to see.I think I'd like to see it, just to see what all the fuss is about. Maybe sneak in and take some pictures, but I doubt I could get past the security.", "role": "user" }, { "content": "The conversation discussed the Bohemian Grove, a secluded redwood forest in California where the Bohemian Club, an exclusive society of powerful individuals, meets annually for a secretive retreat featuring the Cremation of Care ceremony.", "role": "assistant" } ]
        ]
    }
```

A base model has not been instruction fine tuned and therefore it has never seen any data structured in any particular chat template. We can therefore freely choose which one to use for the fune tuning. After fine tuning, when we use the model in practice, we must stick to our chat template choice. In this guide, we will use the `chatml` template. 

We can convert the dataset to the `chatml` format using `unsloth`:
```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, Dataset

data = {"messages": [[ { "content": "You are an expert in summarizing texts. Extract and present the main key point of the input text in one short sentence, including essential details like dates, locations, persons and organizations if necessary.", "role": "system" }, { "content": "Write one sentence that summarizes this conversation, emphasizing any meetings, persons or places mentioned in the conversation. \n\n **Conversation:** \n\n So, I'm thinking about trying this new coffee shop downtown. They apparently roast their own beans, and they have a crazy amount of different brewing methods. I was wondering if you'd be interested in going with me sometime this week.I might be free on Thursday afternoon, maybe around 3 p.m.Thursday works for me. I've been wanting to check out that place too.Perfect. How about we meet at the coffee shop directly?Yeah, that sounds good. What do you think about trying something different, like the siphon brewing method?I'm down for anything.Awesome. I heard they also have some really good pastries, so maybe we can grab a treat while we're there.Definitely.Okay, so Thursday at 3 p.m. at the new coffee shop downtown. Sounds good.Oh, and by the way, I was talking to Sarah, and she mentioned she's looking for a good graphic designer for a project she's working on.Really?Yeah, she said it's for a website redesign. That's interesting. I might be able to help her out. I've been doing some freelance design work lately.That's great. Sarah's got a good budget for this project, and she's really easy to work with. I'll put you two in touch.Thanks, I'd love to hear more about it.No problem. Maybe we can talk about it more at the coffee shop on Thursday.Sounds like a plan.Oh, and while we're on the topic of trades, I was thinking maybe you could help me with something too.Sure, what do you need?Remember how I was telling you about that website I'm building for my photography? I could really use some help with the coding. I'm not the best coder, but I know the basics. What kind of help do you need specifically?Well, I'm having trouble with the navigation menu. It's not displaying correctly on mobile devices.Let me take a look at it. Maybe I can figure it out. That would be amazing.Okay, so it's a deal then. You'll connect me with Sarah, and I'll help you with your website.Perfect.Great. I'm looking forward to our coffee date.Me too.", "role": "user" }, { "content": "The conversation discussed a planned meeting at a new coffee shop downtown on Thursday afternoon to discuss potential trade arrangements, including connecting one party with a graphic design project and providing coding assistance for a website.", "role": "assistant" } ],
 [ { "content": "You are an expert in summarizing texts. Extract and present the main key point of the input text in one short sentence, including essential details like dates, locations, persons and organizations if necessary.", "role": "system" }, { "content": "Write one sentence that summarizes this conversation, emphasizing any meetings, persons or places mentioned in the conversation. \n\n **Conversation:** \n\n I saw a documentary on Bohemian Grove the other day, you know that place in California. It's this massive redwood forest, like seriously huge, and they have this secret society that meets there every year. Apparently, it's been going on for over a hundred years, started by some newspaper guys back in the day. They call themselves the Bohemian Club, and it's supposed to be this exclusive club for powerful people, like politicians, CEOs, and celebrities. Every July, they go up to this grove and have this big retreat where they dress up in togas, build effigies, and burn them. They call it the Cremation of Care ceremony. Sounds a little creepy, right?It's supposed to be a place where they can just relax and let loose, away from the pressures of their everyday lives. But there's a lot of speculation about what really goes on up there. Some people think it's just harmless fun, a bunch of guys getting together for a guys' weekend. Others think it's something more sinister, like a place where they plan world domination or something.There are always protesters outside the grove during the meetings, you know, conspiracy theorists and activists. They accuse the club of being elitist and secretive, and they say they're up to no good. I mean, I don't know what to believe. It's all pretty hush-hush, and they don't let the public in.But the fact that it's so secret just makes people more suspicious. They say there are hidden bunkers and underground tunnels, and that they do all sorts of weird rituals. It's kind of like Area 51, you know? Everyone wants to know what's going on behind closed doors.I read somewhere that Richard Nixon used to go there, and Ronald Reagan too. Apparently, they have a giant owl statue that's supposed to be the symbol of the club. They even have a lake there, and supposedly, they have these little boats shaped like owls that they use to get around.It's just fascinating, you know? This little pocket of secrecy in the middle of the redwoods. It's like stepping back in time, a secret world that most people will never get to see.I think I'd like to see it, just to see what all the fuss is about. Maybe sneak in and take some pictures, but I doubt I could get past the security.", "role": "user" }, { "content": "The conversation discussed the Bohemian Grove, a secluded redwood forest in California where the Bohemian Club, an exclusive society of powerful individuals, meets annually for a secretive retreat featuring the Cremation of Care ceremony.", "role": "assistant" } ]]
}

dataset = Dataset.from_dict(data)

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
dataset_id = "ThatsGroes/synthetic-dialog-summaries-processed"

#dataset = load_dataset(dataset_id, split = "train", stream=True)

### Load model and tokenizer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id, # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

######
# Convert dataset for ChatML format
######

# defaults to using chatml as per https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py  
tokenizer = get_chat_template(
    tokenizer
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)
```

A base model is trained on raw text data to predict the next token, while an instruct model is fine-tuned specifically to follow instructions and engage in conversations. For example, SmolLM2-135M is a base model, while SmolLM2-135M-Instruct is its instruction-tuned variant.

To make a base model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in. ChatML is one such template format that structures conversations with clear role indicators (system, user, assistant).

It's important to note that a base model could be fine-tuned on different chat templates, so when we're using an instruct model we need to make sure we're using the correct chat template.