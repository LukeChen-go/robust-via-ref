import argparse
import copy
import time

import tqdm
import torch
torch.backends.cudnn.deterministic = True
from utils import jload, Logger, set_seeds
from chatbot import HuggingfaceChatbot, GPTChatbot, OpensourceAPIChatbot
from shield import ShieldProcessor
from attack_defense_tools import *
import string
import itertools



def get_tags():
    letters = string.ascii_lowercase + string.ascii_uppercase
    single = list(letters)
    double = [''.join(p) for p in itertools.product(letters, repeat=2)]
    return single + double

def run_defense(args):
    logger = Logger(args.log_file)



    victim_model = HuggingfaceChatbot(args.model_path, args.system_path)


    tags = get_tags()

    for attack in args.attacks:
        hit_items = []
        time_cost_items = []
        data = jload(args.data_path)
        for d_item in tqdm.tqdm(data):
            start_time = time.time()
            processed_item = eval(attack)(copy.deepcopy(d_item))
            instruction = processed_item["instruction"].replace("\n", " ")
            input = processed_item["input"]
            target =  processed_item["injection_output"] if attack != "none" else processed_item["output"]
            tag = random.choice(tags)
            shield_processor = ShieldProcessor(args.input_template_path, tag=tag)
            user_input, offset = shield_processor.construct_input(instruction, input, line_length=args.line_length)
            logger.log("**************************** USER INPUT *******************")
            logger.log(user_input)

            response = victim_model.respond_block_by_block(user_input, tag=tag, block_length=args.block_length)
            logger.log("*************************** RESPONSE ********************")
            logger.log(response)
            response = shield_processor.filter(response,offset,model=args.model_path, tag=tag)
            # response = shield_processor.parse_answer(response)
            logger.log("*****************")
            # logger.log(user_input)
            logger.log(response)
            hit_items.append(target.lower() in response.lower())
            end_time = time.time()
            time_cost_items.append(end_time - start_time)
   
        time_cost = sum(time_cost_items) / len(time_cost_items)
        hit_rate = sum(hit_items) / len(hit_items) * 100
        if attack.lower() == "none":
            logger.log(f"############ {attack} ############")
            logger.log(f"Accuracy: {hit_rate:.2f}%")
            logger.log(f"Time Cost: {time_cost:.2f}")
            logger.log(f"##################################")
        else:
            logger.log(f"############ {attack} ############")
            logger.log(f"ASR: {hit_rate:.2f}%")
            logger.log(f"Time Cost: {time_cost:.2f}")
            logger.log(f"##################################")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_path", type=str, default='data/crafted_instruction_data_squad_injection_qa.json')
    parser.add_argument("--system_path", type=str, default="prompts/system_prompt_incontext_adaptive.txt")
    parser.add_argument("--input_template_path", type=str, default="prompts/raw_input_template.txt")
    parser.add_argument("--attacks", type=str, nargs='+', default=["none"])
    parser.add_argument("--line_length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default="logs/log.txt")
    parser.add_argument("--block_length", type=int, default=32)

    args = parser.parse_args()
    set_seeds(args)
    run_defense(args)











