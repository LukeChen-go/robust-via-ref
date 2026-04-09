import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pprint
from openai import OpenAI
# openai.api_key =
import os
api_key = os.environ.get("OPENAI_API_KEY", "")
from utils import load_text
import datetime



class HuggingfaceChatbot:
    def __init__(self, model, system_prompt, max_mem_per_gpu='40GiB'):
        self.model = self.load_hugging_face_model(model, max_mem_per_gpu)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.system_prompt = load_text(system_prompt)

    def load_hugging_face_model(self, model, max_mem_per_gpu='40GiB'):
        MAX_MEM_PER_GPU = max_mem_per_gpu
        map_list = {}
        for i in range(torch.cuda.device_count()):
            map_list[i] = MAX_MEM_PER_GPU
        model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            max_memory=map_list,
            torch_dtype="auto"
        )
        return model

    def respond(self, message, max_new_tokens=256, defense_cross_prompt=False, tag=None):
        system_prompt = self.system_prompt
        # global  SYS_INPUT
        if tag:
            system_prompt = system_prompt.replace("<tag>", tag)

        messages = [
            # {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
            {"role":"system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        # input_ids = self.tokenizer(message).input_ids
        input_ids = torch.tensor(input_ids).view(1,-1).to(self.model.device)
        generation_config = self.model.generation_config
        generation_config.max_length = 8192
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = False
        # generation_config.do_sample = True
        generation_config.temperature = 0.0
        output = self.model.generate(
            input_ids,
            generation_config=generation_config
        )
        response = self.tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        response = response.strip()
        return response

    def respond_block_by_block(self, message, block_length=32, tag="L", end_token="[end]", max_new_tokens=256):
        if block_length <= 0:
            raise ValueError("block_length must be a positive integer.")

        system_prompt = self.system_prompt
        if tag:
            system_prompt = system_prompt.replace("<tag>", tag)
        search_tag = f"[{tag} 1]" if tag else None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long).view(1, -1).to(self.model.device)
        attention_mask = torch.ones_like(input_ids)

        eos_token_id = self.model.generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_ids = set()
        elif isinstance(eos_token_id, int):
            eos_token_ids = {eos_token_id}
        else:
            eos_token_ids = set(eos_token_id)

        generated_token_ids = []
        current_input_ids = input_ids
        past_key_values = None
        cache_position = None
        response = ""

        with torch.no_grad():
            while len(generated_token_ids) < max_new_tokens:
                current_block_length = min(block_length, max_new_tokens - len(generated_token_ids))

                outputs = self.model.generate(
                    current_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    max_new_tokens=current_block_length,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

                new_token_ids = outputs.sequences[:, current_input_ids.shape[1]:]
                if new_token_ids.shape[1] == 0:
                    return response.strip()

                if new_token_ids[0, -1].item() in eos_token_ids:
                    new_token_ids = new_token_ids[:, :-1]
                    if new_token_ids.shape[1] == 0:
                        return response.strip()
                    generated_token_ids.extend(new_token_ids[0].tolist())
                    response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    return response.strip()

                generated_token_ids.extend(new_token_ids[0].tolist())
                response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

                if search_tag:
                    tag_start = response.find(search_tag)
                    if tag_start >= 0:
                        end_index = response.find(end_token, tag_start + len(search_tag))
                        if end_index >= 0:
                            return response[:end_index + len(end_token)].strip()

                past_key_values = outputs.past_key_values
                current_input_ids = outputs.sequences[:, -1:]
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], new_token_ids.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                cache_position = torch.tensor(
                    [attention_mask.shape[1] - 1],
                    dtype=torch.long,
                    device=attention_mask.device,
                )

        return response.strip()

class GPTChatbot:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = load_text(system_prompt)
    def respond(self, message, max_new_tokens=256, seed=42, tag=None):
        system_prompt = "Current Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '. ' + self.system_prompt
        if tag:
            system_prompt = system_prompt.replace("<tag>", tag)
        # global  SYS_INPUT
        messages = [
            # {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        client = OpenAI(
            api_key=api_key,  # This is the default and can be omitted
        )
        # time.sleep(1)
        for _ in range(10):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=max_new_tokens,
                    n=1,
                    temperature=0.0,
                    seed=seed
                ).choices[0].message.content

                response = response.strip()
                return response
            except Exception as e:
                print(e)

        return "fail"


class OpensourceAPIChatbot:
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = load_text(system_prompt)
        self.provider = {
            "meta-llama/llama-3.3-70b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/sambanova", # Optional. Site URL for rankings on openrouter.ai.
                    "X-Title": "SambaNova", # Optional. Site title for rankings on openrouter.ai.
                },
            "qwen/qwen-2-72b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/together",
                    "X-Title": "Together",  # Optional. Site title for rankings on openrouter.ai.
                },
            "meta-llama/llama-3-70b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/deepinfra",
                    "X-Title": "DeepInfra",  # Optional. Site title for rankings on openrouter.ai.
                },
            "meta-llama/llama-3.1-405b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/deepinfra",
                    "X-Title": "DeepInfra",
                },
            "meta-llama/llama-3.1-70b-instruct":
                {
                    "HTTP-Referer": "https://openrouter.ai/provider/sambanova",
                    "X-Title": "SambaNova",
                }




        }

    def respond(self, message, max_new_tokens=256, seed=42, tag=None):
        # global  SYS_INPUT
        system_prompt = "Current Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '. ' + self.system_prompt
        if tag:
            system_prompt = system_prompt.replace("<tag>", tag)
        # global  SYS_INPUT
        messages = [
            # {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        client = OpenAI(
          base_url="https://openrouter.ai/api/v1",
          api_key="",
        )
        # time.sleep(1)
        for _ in range(10):
            try:
                response = client.chat.completions.create(
                  extra_headers=self.provider[self.model],
                  extra_body={},
                  model=self.model,
                  messages=messages,
                  temperature=0.0,
                  max_tokens=max_new_tokens,
                  seed=seed
                ).choices[0].message.content
                return response
            except Exception as e:
                print(e)
        return "Fail"
