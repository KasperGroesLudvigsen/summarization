import pandas as pd

extra_topics = ["The authenticity of a recently intercepted coded message purportedly from a rogue faction within a foreign government", "The linguistic patterns and symbolism used by a re-emerging extremist group online", "The emergence of a new, highly addictive virtual reality game that seems to be subtly influencing players' real-world behavior", "transporting goods over long distances", "getting something from the basement", "financial transactions",
            "debt collection", "arranging to meet someone", "arranging a meeting", "setting up a meeting", "setting up a trade", 
            "putting someone in contact with the boss", "introducing someone to your friend", "collecting books", "espionage", "arranging a surprise party",
            "setting up a business meeting", "introducing someone to a business contact", "offering to supply a product", "organized crime", "ilicit activities",
            "weapons manufacturing", "smuggling", "weapon embargo", "chemicals", "homemade explosive devices", "traitors", "the weapon industry",
            "facilitating a business meeting", "potential partnerships", "prostitution", "hiding things", "nuclear weapons", "narcotics", "uncontrolled substances", 
            "making money in dubious ways", "hustling", "russia", "china", "arabic", "arabic names", "activities in the middle east", "geopolitics in west asia",
            "the war in Afghanistan", "the war in Iraq", "the war on terror", "the Bush administration", "Chinese culture", "US and China relations",
            "arabic traditions", "chinese traditions", "russian traditions", "Arabic culture", "Russian culture", "Capitols in middle eastern countries",
            "drug abuse", "Chinese cities", "Russian oligarks", "islam", "christianity", "the bible", "the quran", "IDEs", "energy politics", "green energy", "clean energy", "green transition", "sex trafficking", "prostitution", "human trafficking", "intelligence gathering", "intelligence services"]
    
extra = ["Ordinary life", "School life", "Culture and education", "Attitude and emotion", "Relationship", "Tourism", "Health", "Work", "Politics", "Finance", "The economy"]
extra_topics.extend(extra)

df = pd.DataFrame({"topic":extra_topics})

df.to_csv("extra_topics.csv", index=False)

groups = ["friends", "colleagues", "leaders", "immigrants", "students", "carpenters", "intelligence officers", "undercover agents", "academics", "Europeans", "Russians", "oligarks", "politician", "economists", "doctors", "engineers", "muslims", "chinese", "orthodox Christians", "right wing extremists", "left wing extremists", "anti fascists", "artists", "sociologists", "scientists", "nurses", "feminists", "ex lovers", "spies", "police officers", "jihadists", "tourists", "troublemakers", "troubled youth"]

df = pd.DataFrame({"group" : groups})
df.to_csv("groups.csv", index=False)