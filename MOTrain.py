import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
import sys
import torch.nn as nn

subset = sys.argv[1]

max_len = 2048 # replace with the max input length of your model, recommend no less than 2k

model_name = "CodeLlama-7b-Instruct-hf" # replace with the model name you want to train

full_dataset = load_dataset("json", data_files="trainset.json", split="train") # replace with your dataset location

output_dir = 'output_model' # replace with your output model location

# load as 4bit model, prepare for qlora
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# define multi-task data collator
class TaskPrefixDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        pred_features, expl_features = [], []
        
        # print(features)

        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            print(len(input_ids))

            # find the split points, which is the eos_token_id, in input_ids
            split_indices = [i for i, x in enumerate(input_ids) if x == eos_token_id]

            # assure at least 1 split point
            if len(split_indices) < 2:
                print('data illegal, not enough split points!')
                sys.exit(0)
                # if len(split_indices) < 2, exit the program
            # split the input_ids and attention_mask into two parts
            pred_features.append({
                'input_ids': (input_ids[:split_indices[0]] + input_ids[split_indices[0]+1:split_indices[1]])[:max_len-1] + [2],
                'attention_mask': (attention_mask[:split_indices[0]] + attention_mask[split_indices[0]+1:split_indices[1]])[:max_len-1] + [1]
            })
            expl_features.append({
                'input_ids': (input_ids[:split_indices[0]] + input_ids[split_indices[1]+1:])[:max_len-1] + [2],
                'attention_mask': (attention_mask[:split_indices[0]] + attention_mask[split_indices[1]+1:])[:max_len-1] + [1]
            })
        # use the base class's __call__ method to process the split features
        pred_features = super().__call__(pred_features, return_tensors)
        expl_features = super().__call__(expl_features, return_tensors)
        return {
            'pred': pred_features,
            'expl': expl_features,
        }

# multi-objective fine-tuning trainer
class TaskPrefixTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_beta = float(sys.argv[3])
        self.separator_token_id = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs.keys())
        pred_outputs = model(**inputs['pred'])
        loss = pred_outputs.loss
        if self.weight_beta != 0:
            expl_outputs = model(**inputs['expl'])
            loss += self.weight_beta * expl_outputs.loss
        loss /= (1 + self.weight_beta)
        return (loss, {'pred': pred_outputs, 'expl': expl_outputs}) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        print(pred_outputs[0], expl_outputs[0])
        loss = (pred_outputs[0] + self.weight_beta * expl_outputs[0]) / (1 + self.weight_beta)
        print(loss)
        return (
            loss,
            [pred_outputs[1], expl_outputs[1]],
            [pred_outputs[2], expl_outputs[2]],
        )

# QLoRA parameters selection function
def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

# QLoRA config
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=find_all_linear_names(base_model, int4=True),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# get peft model
base_model = get_peft_model(base_model, peft_config)

# trainer config
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing =True,
    prediction_loss_only=False,
    max_grad_norm= 0.3,
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=100,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="constant",
    warmup_ratio=0.05,
    remove_unused_columns = False,
    neftune_noise_alpha=5,
)

# data collator
data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, mlm=False)

# multi-objective fine-tuning trainer
trainer = TaskPrefixTrainer(
    model=base_model,
    train_dataset=full_dataset,
    max_seq_length=1048576,
    data_collator=data_collator,
    tokenizer=tokenizer,
    dataset_text_field='text',
    args=training_args,
)

# start training
trainer.train()
trainer.save_model(output_dir)

# save final checkpoint
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
codellama_merged_dir = os.path.join(output_dir, 'codellama_merged')

os.makedirs(final_checkpoint_dir, exist_ok=True)
os.makedirs(codellama_merged_dir, exist_ok=True)

trainer.model.save_pretrained(final_checkpoint_dir)

print('training process finished ...')

# merge model
del trainer
del base_model
del data_collator
del full_dataset
import gc
torch.cuda.empty_cache()
gc.collect()
gc.collect()

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

print('load model success ...')

model = PeftModel.from_pretrained(base_model, output_dir)
model = model.merge_and_unload()
print('merge model success ...')
model.save_pretrained(codellama_merged_dir, safe_serialization=True)

print('merge model saved success ...')

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(codellama_merged_dir)