# Code used to generate pre-training dataset
# The prompt is sourced from the TinyStories paper
import random
from openai import OpenAI
import json
import re

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY,)

OUTPUT_FOLDER = "samples"

PROMPT = "Write a short story (3-5 paragraphs) which only uses very simple words\
            that a 3 year old child would likely understand. The story should use\
            the verb \"{0}\", the noun  \"{1}\" and the adjective  \"{2}\". \
            The story should have the following features:  \"{3}\". Remember to only use simple words!"

nouns   = [n.strip() for n in open("features/nouns").readlines()]
verbs   = [n.strip() for n in open("features/verbs").readlines()]
adjectives = [n.strip() for n in open("features/adjectives").readlines()]
features = [n.strip() for n in open("features/features").readlines()]

import re

def _split_and_select_random_sentence(story):
    # Splitting the story into sentences using regular expressions to account for ".", "?", "!"
    sentences = re.split(r'(?<=[.!?]) +', story)

    # Removing the first sentence if there are multiple sentences
    if len(sentences) > 1:
        sentences.pop(0)

    # Selecting a random sentence
    random_sentence = random.choice(sentences) if sentences else ""

    return random_sentence

def _gen_summary(story):
    prompt = "Write a short summary of the following story (1 paragraph) which only uses very simple words\
            that a 3 year old child would likely understand. Story: \"{0}\""

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt.format(story)}
            ]
    )
    return completion.choices[0].message.content

def _gen_pretraining(format_prompt, file):
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content": format_prompt}
                ]
        )

    open(file + "_pretrain.txt", "w+").write(completion.choices[0].message.content)
    return completion.choices[0].message.content

def _gen_instruct(story, file, elements):
    instructions = ["contain_words", "summary", "features", "contain_sentence"]
    chosen_instructions = random.sample(instructions, 2)
    with open(file + "_instruct.txt", "w+") as o:
        for inst in chosen_instructions:
            if inst == "contain_words":
                o.write("Words: " + elements["words"] + "\n")
            elif inst == "summary":
                o.write("Summary: " + _gen_summary(story) + "\n")
            elif inst == "features":
                o.write("Features: " + elements["features"]  + "\n")
            elif inst == "contain_sentence":
                o.write("Sentence: " + _split_and_select_random_sentence(story) + "\n")
        o.write("Story: " + story)

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
        print(file)
        output = _gen_pretraining(format_prompt, file)
        _gen_instruct(output, file, dict(words = ", ".join([verb, noun, adjective]), features=feature))

if __name__ == "__main__":
    GEN_SIZE = 1000
    generate(GEN_SIZE, OUTPUT_FOLDER, sample_mode=True, create_instruct=True)