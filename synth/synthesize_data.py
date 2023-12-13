# Code used to generate pre-training dataset
# The prompt is sourced from the TinyStories paper
import random
from openai import OpenAI
import json

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY,)

OUTPUT_FOLDER = "samples_pretraining"

PROMPT = "Write a short story (3-5 paragraphs) which only uses very simple words\
            that a 3 year old child would likely understand. The story should use\
            the verb \"{0}\", the noun  \"{1}\" and the adjective  \"{2}\". \
            The story should have the following features:  \"{3}\". Remember to only use simple words!"

nouns   = [n.strip() for n in open("features/nouns").readlines()]
verbs   = [n.strip() for n in open("features/verbs").readlines()]
adjectives = [n.strip() for n in open("features/adjectives").readlines()]
features = [n.strip() for n in open("features/features").readlines()]

def _gen_pretraining(format_prompt, file):
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content": format_prompt}
                ]
        )

    open(file + "_pretrain.txt", "w+").write(completion.choices[0].message.content)

def _gen_instruct(story, features):

def generate(N, output_folder, sample_mode=False, create_instruct=False):
    if sample_mode:
        N = N // 100
    

    for sample_num in range(N):
        verb = random.choice(verbs)
        noun = random.choice(nouns)
        adjective = random.choice(adjectives)
        feature = ", ".join(random.sample(features, 2))
        
        format_prompt = PROMPT.format(verb, noun, adjective, feature)
        
        file = output_folder + "/" + str(sample_num)
        _gen_pretraining(format_prompt, file)

       
        
        # generate instruct variants
        #instructions = ["contain words", "summary", "features", "contain_sentence"]
        #chosen_instructions = random.sample(instructions, 2)
        #with open(output_folder + "/" + str(sample_num) + "instruct.txt", "w+") as o:
        #    for inst in chosen_instructions:
            

if __name__ == "__main__":
    GEN_SIZE = 1000
    generate(GEN_SIZE, OUTPUT_FOLDER, sample_mode=True, create_instruct=True)
