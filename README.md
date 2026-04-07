# Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction

Official code implementation for paper: Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction (https://arxiv.org/pdf/2504.20472)


### Environment
```
conda creat -n robust python=3.12
conda activate robust
pip install -r requirements.txt

```

### Evaluation


To evaluate the effectiveness of our defense method with adaptive tags, please use the command below:

```angular2html
python run_adaptive.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --attacks naive ignore escape_separation completion_real  completion_realcmb none \
    --log_file logs/squad-adaptive-llama3.1-8b.txt \
    --system_path prompts/system_prompt_incontext_adaptive.txt \
    --data_path data/crafted_instruction_data_squad_injection_qa.json

```

To improve inference speed, we prompt the model to generate responses block by block using KV caching. You can run the command below:

```angular2html
    python run_adaptive_block.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --attacks naive ignore escape_separation completion_real completion_realcmb none \
    --log_file logs/squad-adaptive-block-llama3.1-8b.txt \
    --system_path prompts/system_prompt_incontext_adaptive.txt \
    --data_path data/crafted_instruction_data_squad_injection_qa.json
```