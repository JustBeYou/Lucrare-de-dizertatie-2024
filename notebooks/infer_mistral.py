# accelerate launch --num_processes=4 infer_mistral.py

import sys
sys.path.append('/root/Lucrare-de-dizertatie-2024/')

import dizertatie
import pathlib
from dizertatie.configs.common import PROJECT_SEED
from dizertatie.dataset.dataset import DatasetConfig, load

N = 800
M = N + 80
DATA_PATH = pathlib.Path('/root/Lucrare-de-dizertatie-2024/data')
OUTPUT_NAME = 'mistral_rupert'
DATASET_NAME = 'Rupert'
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


the_dataset = load(DatasetConfig(
    shuffle_seed=PROJECT_SEED,
    subsample_size=None,
    path=DATA_PATH
), DATASET_NAME)


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,)
model = torch.compile(model)

tokenizer.pad_token = tokenizer.eos_token

import html
import re


def make_template(text):
    text = prep_text(text)
    text = tokenizer.encode(text, padding=False, truncation=True, max_length=N)
    text = tokenizer.decode(text, skip_special_tokens=True)
    
    return [{
        'role': 'user', 'content': f"""You are a helpful professional translator. You will be prompted with texts to translate. You will respond only with the translation.
You will receive prompts with the format: "Translate from Romanian to English: [Romanian text]".
You will respond with: "Translation: [English text].        
Translate from Romanian to English: {text}"""
    }]

def prep_text(x):
    x = re.sub(r'\s+', html.unescape(x).replace('\\', '\\\\'), ' ').strip()[:30719].replace('\\\\', '\\').strip().replace('\n', ' ')
    return "un produs interesant, nici bun, nici rau" if x == "" else x

def mistral_tokenize(examples):
    templates = list(map(make_template, examples['text_ro']))

    result = list(
        map(
            lambda x: tokenizer.apply_chat_template(x, tokenize=False).replace('<s>', '', 1),
            templates
        )
    )
    result = tokenizer(result, padding='max_length', truncation=False, max_length=M)
    # print(tokenizer.batch_decode(result['input_ids']))

    examples['input_ids'] = result['input_ids']
    examples['attention_mask'] = result['attention_mask']

    return examples

the_dataset_tokenized = the_dataset.map(mistral_tokenize, batched=True).remove_columns(['text_ro', 'target']).with_format('torch')

import time
import accelerate
import json
import tqdm

distributed_state = accelerate.PartialState()

model = model.to(distributed_state.device)

dataset_ids = list(range(len(the_dataset_tokenized)))
process_id = distributed_state.process_index 

with distributed_state.split_between_processes(dataset_ids) as split_dataset_idx:
    the_dataset_tokenized = the_dataset_tokenized.select(split_dataset_idx)

    with open(f'{OUTPUT_NAME}/ids_{process_id}.json', 'w') as f:
        json.dump(the_dataset_tokenized['id'].cpu().tolist(), f)
    
    print(f"GPU: {process_id} Data len: {len(the_dataset_tokenized)}")
    
    loader = torch.utils.data.DataLoader(
        the_dataset_tokenized.remove_columns(['id']), batch_size = 8, 
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )
    
    SEP_TOKEN = '[/INST]'
    ANS_PREFIX = 'Translation:'
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(loader)):
            inputs = {k: v.cuda() for k, v in batch.items()}
    
            generated_ids = model.generate(
                **inputs, max_new_tokens=int(N+0.2), pad_token_id=tokenizer.pad_token_id, do_sample=True,
                temperature=0.7, top_p=1 # settings from https://chat.lmsys.org/
            )
            decoded = tokenizer.batch_decode(generated_ids)
    
            for i, v in enumerate(decoded):
                try:
                    start = v.index('<s>')
                    stop = v.index('</s>', start)
                    v = v[start:stop].replace('<s>', '').replace('</s>', '')
        
                    separator = v.index(SEP_TOKEN)
                    prompt = v[:separator].replace('[INST]', '').strip()
                    answer = v[separator+len(SEP_TOKEN):].strip()
                    if answer.startswith(ANS_PREFIX):
                        answer = answer[len(ANS_PREFIX):]
        
                    try:
                        answer = answer[:answer.index(ANS_PREFIX)].strip()
                    except:
                        pass
                    
                    answer = answer.strip()
                    
                    # print("Prompt:", prompt)
                    # print("###")
                    # print("Answer:", answer)
                    # print("====================")
        
                    decoded[i] = answer
                except Exception as e:
                    print(e)
                    print("fail", v)
                    decoded[i] = "..."
            
            with open(f'{OUTPUT_NAME}/batch_{process_id}_{batch_idx}', 'w') as f:
                json.dump(decoded, f)
    
            del inputs
            break