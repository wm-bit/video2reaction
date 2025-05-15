from pydantic import BaseModel
from enum import Enum
import json
import os
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from enum import Enum
import json
import torch
from typing import List
from tqdm import tqdm
import re

#TODO: implement changes for v2 pipeline

# Output definition
class ReactionL1Class(str, Enum):
    joy = "joy"
    sadness = "sadness"
    anger = "anger"
    surprise = "surprise"
    disgust = "disgust"
    fear = "fear"
    neutral = "neutral"
    # TODO: need to add irrelevant

class ReactionL2Class(str, Enum):
    admiration = "admiration"
    amusement = "amusement"
    anger = "anger"
    annoyance = "annoyance"
    approval = "approval"
    caring = "caring"
    confusion = "confusion"
    curiosity = "curiosity"
    desire = "desire"
    disappointment = "disappointment"
    disapproval = "disapproval"
    disgust = "disgust"
    embarrassment = "embarrassment"
    excitement = "excitement"
    fear = "fear"
    gratitude = "gratitude"
    grief = "grief"
    joy = "joy"
    love = "love"
    nervousness = "nervousness"
    optimism = "optimism"
    pride = "pride"
    realization = "realization"
    relief = "relief"
    remorse = "remorse"
    sadness = "sadness"
    surprise = "surprise"
    neutral = "neutral"

#TODO: need to do some literature review to find out what are the common reasons for each reaction
class ReactionReasonType(str, Enum):
    acting = "acting"
    plot = "plot"
    music = "music"
    cinematography = "cinematography"
    character = "character"
    dialogue = "dialogue"
    pacing = "pacing"
    setting = "setting"
    relatable = "relatable"
    nostalgia = "nostalgia"
    humor = "humor"
    unrelated_to_clip = "unrelated_to_clip"
    none = "none"
    other = "other"

class ReactionParsingOutput(BaseModel):
    high_level_reaction: List[ReactionL1Class]
    finer_grained_reaction: List[ReactionL2Class]
    reaction_reason: str
    reaction_reason_type: List[str]

class ReactionParsingOutputNoRestriction(BaseModel):
    high_level_reaction: List[ReactionL1Class]
    finer_grained_reaction: List[str]
    reaction_reason: str
    reaction_reason_type: List[str]

FEW_SHOT_EXAMPLES = json.load(open(os.path.join(os.path.dirname(__file__), "example_v1.json"), "r"))

def extract_tag_text(text, tag, random=False):
    """
    Extract text between two tags
    random=True: Extracts text between the first tag and the next tag
    random=False: Extracts text between the first tag and the closing tag
    """
    # if the first line is None, return None
    if random:
        pattern = re.compile(rf"<{tag}>(.*?)<(.*?)>", re.DOTALL)
    else:
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)

    if pattern.findall(text):
        return pattern.findall(text)[0][0]
    else:
        # check if there is a line with None
        if "\nNone\n" in text:
            return None
        print(f"Tag {tag} not found in the text: {text}")
        return False

def rephrase_comments(my_llm, comment_dict, version="v2", include_examples=True):
    # open a file from the same directory as the current file
    if include_examples:
        prompt_template = open(os.path.join(os.path.dirname(__file__), f'{version}_comment_rephrase_prompt_few_shot.md'), 'r').read()
    else:
        prompt_template = open(os.path.join(os.path.dirname(__file__), f'{version}_comment_rephrase_prompt.md'), 'r').read()
    prompts = [prompt_template.format(summary=data['clip_summary'], comment=data['comment']) for id, data in comment_dict.items()]

    tokenizer = my_llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, 
                                    max_tokens=200,
                                    stop_token_ids=[tokenizer.eos_token_id, 
                                                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                                                tokenizer.convert_tokens_to_ids("</rephrased>")])
    # Run the LLM with the prompt

    result = my_llm.generate(prompts, sampling_params=sampling_params,
                            )
    print("After generating rephrased comments...")

    # Extract the generated output
    outputs = {}
    for idx, (comment_id, comment) in enumerate(comment_dict.items()):
        output = result[idx].outputs[0].text
        output = extract_tag_text(output, "rephrased", random=True)
        outputs[comment_id] = {
            "clip_summary": comment["clip_summary"],
            "comment": comment["comment"],
            "rephrased_comment": output,
            "raw_output" : result[idx].outputs[0].text}
    # How many None and False comments
    none_count = len([v for v in outputs.values() if v["rephrased_comment"] is None])
    false_count = len([v for v in outputs.values() if v["rephrased_comment"] is False])
    print(f"With reaction: {len(comment_dict.keys()) - none_count - false_count};  Without reaction: {none_count}; Cannot process: {false_count}")
    return outputs

def analyze_comment(my_llm, comment_dict, version="v2", include_examples=False):
    # Prepare the prompt
    prompt_template = open(os.path.join(os.path.dirname(__file__), f"{version}_comment_reaction_extraction_prompt.md"), 'r').read()
    outputs = {}
    prompts = []
    extracted_ids = []
    for id, data in comment_dict.items():
        if data['rephrased_comment'] != "None" and data['rephrased_comment'] is not None and data['rephrased_comment'] is not False:
            prompt = prompt_template.format(clip_description=data['clip_summary'], rephrased_comment=data['rephrased_comment'])
            prompts.append(prompt)
            extracted_ids.append(id)
        else:
            outputs[id] = {
                "clip_summary": data["clip_summary"],
                "comment": data["comment"],
                           "rephrased_comment": data["rephrased_comment"],
                           "parsed_output": None,
                           "raw_output": None}

    # Set sampling parameters for generation
    json_schema = ReactionParsingOutput.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(guided_decoding=guided_decoding_params,
                                     temperature=0.0,
                                     max_tokens=200)

    # Run the LLM with the prompt
    result = my_llm.generate(prompts, sampling_params=sampling_params)

    # Extract the generated output
    for idx, comment_id in enumerate(extracted_ids):
        output = result[idx].outputs[0].text
        try:
            json_output = json.loads(output)
        except json.JSONDecodeError:
            json_output = None
        outputs[comment_id] = {
            "clip_summary": comment_dict[comment_id]["clip_summary"],
            "comment": comment_dict[comment_id]["comment"],
                                "rephrased_comment": comment_dict[comment_id]["rephrased_comment"],
                               "parsed_output": json_output,
                               "raw_output": output}
    return outputs