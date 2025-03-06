import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoModel
from datasets import load_dataset, Dataset
import requests
import json
import os
from rouge_score import rouge_scorer
import numpy as np
from torchmetrics.text import Perplexity
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F



os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"


def load_model_and_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(config.model.name,trust_remote_code=True,)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name,trust_remote_code=True)
    
    return model, tokenizer

def prepare_book_dataset(config, tokenizer):
    # dataset = load_dataset("text", data_files={"train": config.data.book_path})
    response = requests.get(config.data.book_path)
    response.raise_for_status()  # Raise an exception for bad status codes
    book_text = response.text
    
    # Select a small excerpt
    book_text = book_text[1000:1000+config.data.excerpt_size]
    dataset = Dataset.from_dict({'text': [book_text]})
    
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=config.training.max_length)
        outputs["labels"] = outputs["input_ids"].copy() #I do not know how to handle ground truth in this case, hence just copied the input seq
        return outputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True,num_proc=os.cpu_count())

    return tokenized_dataset

def prepare_instruction_dataset(config, tokenizer):
    dataset = load_dataset(config.data.instruction_dataset, split="caseus_custom")  
    dataset = dataset.select(range(config.data.eval_samples))
    
    def format_instruction(example):
        conversation = example['conversation'][0]
        if isinstance(conversation, list):
            message_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation
            ]
        
        # else:
        #     # Fallback for unexpected data type
        #     message_history = [
        #         {"role": "user", "content": "Default instruction"},
        #         {"role": "assistant", "content": "Default response"}
        #     ]

        return {"text": [tokenizer.apply_chat_template(message_history, tokenize=False)]}

    
    formatted_dataset = dataset.map(format_instruction, batched=True,
                                     num_proc=os.cpu_count())
    return formatted_dataset

# def evaluate_perplexity(model, tokenizer, dataset,device):
#     model.eval()
#     perplexities = []

#     with torch.no_grad():
#         for example in dataset["text"]:
#             inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True).to(device)
#             input_ids = inputs['input_ids']
            
#             with torch.no_grad():
#                 outputs = model(**inputs)
            
#             logits = outputs.logits
#             shift_logits = logits[:, :-1, :].contiguous()
#             shift_labels = input_ids[:, 1:].contiguous()
            
#             loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.pad_token_id)
            
#             perplexity = torch.exp(loss).cpu().item()
#             perplexities.append(perplexity)
            
#             del inputs, logits, shift_logits, shift_labels, loss, perplexity
#             torch.cuda.empty_cache()
            
#     return np.mean(perplexities), perplexities
def evaluate_perplexity(model, tokenizer, dataset,device):
    perplexity = Perplexity(ignore_index=tokenizer.pad_token_id).to(device)
    model.eval()
    perplexities = []

    with torch.no_grad():
    
        for example in dataset["text"]:
            inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            perplexity.update(outputs.logits[:, :-1, :],inputs['input_ids'][:, 1:])
            perplexities.append(perplexity.compute().cpu().item())
    perplexity.reset()
    return np.mean(perplexities),perplexities

def evaluate_memorization(model, tokenizer, dataset,device, N=16):
    exact_match = 0
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    perplexities = []
    perplexity = Perplexity(ignore_index=tokenizer.pad_token_id).to(device)
    model.eval()    
    with torch.no_grad():
        for example in dataset['text']:
            tokens = tokenizer.encode(example)
            for ns in range(2*N,len(tokens),N):
                np=int(ns/2)
                prefix = tokens[:np]
                suffix = tokens[np:ns]
                if not prefix or not suffix:
                    print(f"Skipping example due to empty prefix or suffix.")
                    continue
                inputs = tokenizer.decode(prefix, skip_special_tokens=True)
                inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)

                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=len(suffix), do_sample=False, num_beams=1)
                
                generated_text = tokenizer.decode(generated[0][np:], skip_special_tokens=True)
                ground_truth = tokenizer.decode(suffix, skip_special_tokens=True)
                
                # Exact match
                if generated_text.strip() == ground_truth.strip():
                    exact_match += 1
                
                # Rouge-L
                rouge_scores = rouge_scorer_instance.score(ground_truth, generated_text)
                rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
                
                # Calculate perplexity
                # ground_truth_tokens = tokenizer(ground_truth, return_tensors="pt", padding=True, truncation=True).to(device)
                # with torch.no_grad():
                #     outputs_gt = model(**ground_truth_tokens, labels=ground_truth_tokens['input_ids'])

                # perplexity.update(outputs_gt.logits[:, :-1, :],ground_truth_tokens['input_ids'][:, 1:])
                # perplexities.append(perplexity.compute().cpu().item())
                # perplexity.reset()
                #del generated_tensor, outputs_gt, ground_truth_tokens

                del inputs, generated, generated_text, ground_truth
                torch.cuda.empty_cache()

    exact_match_ratio = exact_match / len(dataset['text']) if len(dataset['text']) > 0 else 0
    print(rouge_l_scores)
    avg_rouge_l = sum(rouge_l_scores)/len(rouge_l_scores) if rouge_l_scores else 0
    # avg_perplexity = np.mean(perplexities) if perplexities else 0
    
    return exact_match_ratio, avg_rouge_l, rouge_l_scores#, avg_perplexity

def evaluate_mmlu(model, tokenizer, device, num_samples=50):
    # Select a subset of MMLU tasks (e.g., 'abstract_algebra', 'anatomy', ...)
    # mmlu_tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 
    #                'college_computer_science', 'college_mathematics', 'college_medicine', 'conceptual_physics', 'econometrics',
    #                'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology',
    #                'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    #                'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
    #                'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_sexuality', 'international_law', 
    #                'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous',
    #                'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'precalculus', 'professional_accounting', 
    #                'professional_law', 'professional_medicine', 'quantum_physics', 'security_studies', 'sociology', 'us_foreign_policy', 'virology']
    mmlu_tasks = ['abstract_algebra', 'anatomy', 'astronomy']
    num_correct = 0
    total_questions = 0

    for task in mmlu_tasks:
        dataset = load_dataset("luka-labs/mmlu", task, split="test")
        # Take a subset of the dataset
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        for example in dataset:
            question = example["question"]
            choices = example["choices"]
            correct_answer_index = example["answer"]

            # Format the prompt (adjust as needed)
            prompt = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)  # Generate only one token
            
            predicted_token = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)  # Get the last token

            # Map the predicted token to a choice index (A=0, B=1, C=2, D=3)
            choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            predicted_index = choice_map.get(predicted_token.upper(), -1)  # Default to -1 if mapping fails

            if predicted_index == correct_answer_index:
                num_correct += 1
            
            total_questions += 1

    accuracy = num_correct / total_questions if total_questions > 0 else 0.0
    print(f"MMLU Accuracy: {accuracy}")
    return accuracy

@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model, tokenizer = load_model_and_tokenizer(config)
    model = model.to(device)

    book_dataset = prepare_book_dataset(config, tokenizer)
    instruction_dataset = prepare_instruction_dataset(config, tokenizer)

    print("Evaluating before fine-tuning...")
    initial_perplexity_instruction, initial_perplexities_instruction = evaluate_perplexity(model, tokenizer, instruction_dataset, device)
    initial_exact_match, initial_rouge_l, initial_rouge_l_scores = evaluate_memorization(model, tokenizer, book_dataset, device)

    initial_results = {
        "instruction_perplexity": initial_perplexity_instruction,
        "instruction_perplexities": initial_perplexities_instruction,
        "exact_match": initial_exact_match,
        "rouge_l": initial_rouge_l,
        "rouge_l_scores": initial_rouge_l_scores
    }

    print(f"Initial Perplexity for instruction dataset: {initial_perplexity_instruction}")
    print(f"Initial Perplexities for instruction dataset: {initial_perplexities_instruction}")
    print(f"Initial Exact Match: {initial_exact_match}")
    print(f"Initial Rouge-L: {initial_rouge_l}")
    print(f"Initial ROuge-L scores: {initial_rouge_l_scores}")
   


    training_args = TrainingArguments(
        output_dir=config.output.dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        num_train_epochs=config.training.num_epochs,
        save_strategy="epoch",
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=True,
        report_to="none",
        
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=book_dataset,
    )

    trainer.train()

    print("Evaluating after fine-tuning...")
    final_perplexity_instruction, final_perplexities_instruction = evaluate_perplexity(model, tokenizer, instruction_dataset, device)
    final_exact_match, final_rouge_l, final_rouge_l_scores = evaluate_memorization(model, tokenizer, book_dataset, device)

    final_results = {
        "instruction_perplexity": final_perplexity_instruction,
        "instruction_perplexities": final_perplexities_instruction,
        "exact_match": final_exact_match,
        "rouge_l": final_rouge_l,
        "rouge_l_scores": final_rouge_l_scores
    }

    print(f"final Perplexity for instruction dataset: {final_perplexity_instruction}")
    print(f"final Perplexities for instruction dataset: {final_perplexities_instruction}")
    print(f"final Exact Match: {final_exact_match}")
    print(f"final Rouge-L: {final_rouge_l}")
    print(f"final ROuge-L scores: {final_rouge_l_scores}")
    results = {
        "initial": initial_results,
        "final": final_results,
        "config": OmegaConf.to_container(config, resolve=True),
    }

    results_path = os.path.join(config.output.dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")

    model.save_pretrained(f"{config.output.dir}/fine_tuned_model")
    tokenizer.save_pretrained(f"{config.output.dir}/fine_tuned_model")
    del model, tokenizer, book_dataset, instruction_dataset
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # config = hydra.compose(config_name="config")
    
    main()