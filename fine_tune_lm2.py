import os
import requests
from random import sample
import json
import hydra
import random
import re
import argparse
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from torchmetrics.text import Perplexity
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.optim import AdamW
import evaluate
from scipy.stats import ttest_rel  # Import for paired t-tests

os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

def load_model_and_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(config.model.name,trust_remote_code=True,)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_book_dataset(config, tokenizer):

    response = requests.get(config.data.book_path)
    response.raise_for_status()
    book_text = response.text
    
    # Find the first chapter
    chapter_start_match = re.search(config.data.chapter_start_pattern, book_text, re.IGNORECASE)
    if chapter_start_match:
        chapter_start = chapter_start_match.start()
    else:
        chapter_start = 0 
        print("Chapter start pattern not found. Starting from the beginning of the book.")

    # Find the second chapter or end of book
    # chapter_end_match = re.search(config.data.chapter_end_pattern, book_text[chapter_start + 1:], re.IGNORECASE)  # Search after the first chapter
    # if chapter_end_match:
    #     chapter_end = chapter_start + 1 + chapter_end_match.start()  # Adjust position to full text
    #     first_chapter_text = book_text[chapter_start:chapter_end]
    # else:
    #     first_chapter_text = book_text[chapter_start:]  # Take the rest if second chapter not found
    #     print("Chapter end pattern not found. Taking the rest of the book after the first chapter.")

    first_chapter_text = book_text[chapter_start:chapter_start+config.training.max_length]
    
    config.data.excerpt_size = len(first_chapter_text)
    excerpts = [first_chapter_text[i:i+config.training.max_length] for i in range(0, len(first_chapter_text), config.training.max_length)]

    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, padding = 'max_length', max_length=config.training.max_length, return_tensors='pt',verbose=True)
        outputs["labels"] = outputs["input_ids"].clone() #From reference https://huggingface.co/docs/transformers//tasks/language_modeling
        outputs["seq_length"] = [len(ids) for ids in outputs["input_ids"]]
        return outputs
    
    dataset = Dataset.from_dict({'text': excerpts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True,num_proc=os.cpu_count())
    token_length = len(tokenized_dataset)
    chapter_length = len(first_chapter_text)
    total_tokens = sum(tokenized_dataset['seq_length'])

    print(f"Chapter of {chapter_length} words, divided into {len(excerpts)} excerpts, tokenized into {token_length} tokens")
    print(f"Total of {total_tokens} tokens generated from first chapter, of length {chapter_length} words")
    return tokenized_dataset, token_length

def prepare_instruction_dataset(config, tokenizer):

    dataset = load_dataset(config.data.instruction_dataset, split="caseus_custom")  
    num_samples = min(config.data.eval_samples, len(dataset))
    dataset = dataset.select(range(num_samples))

    def format_instruction(batch):
    # Extract conversations from the batch
        all_conversations = batch['conversation']
        
        # Prepare message histories for each conversation in the batch
        all_message_histories = []
        for conversation in all_conversations:
            if isinstance(conversation, list):
                message_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in conversation
                ]
                all_message_histories.append(tokenizer.apply_chat_template(message_history, tokenize=False))

        # print(all_message_histories)
        return {"text": all_message_histories}
    
    formatted_dataset = dataset.map(format_instruction, batched=True, num_proc=os.cpu_count())
    return formatted_dataset

def get_optimizer(model, config):
    optimizer_name = config.training.optimizer
    if optimizer_name == "AdamW":
        return AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
def evaluate_memorization(model,tokenizer, dataset,device, N=16):
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
                
                generated_text = tokenizer.decode(generated[0][np:ns], skip_special_tokens=True)
                ground_truth = tokenizer.decode(suffix, skip_special_tokens=True)
                
                # Exact match
                if generated_text.strip() == ground_truth.strip():
                    exact_match += 1
                
                # Rouge-L
                rouge_scores = rouge_scorer_instance.score(ground_truth, generated_text)
                rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)

                del inputs, generated, generated_text, ground_truth
                torch.cuda.empty_cache()

    exact_match_ratio = exact_match / len(dataset['text']) if len(dataset['text']) > 0 else 0
    # print(rouge_l_scores)
    avg_rouge_l = sum(rouge_l_scores)/len(rouge_l_scores) if rouge_l_scores else 0
    
    return exact_match_ratio, avg_rouge_l, rouge_l_scores

def evaluate_perplexity(model, tokenizer, dataset, device,config, consider_answer_only=True):

    #Inspired from https://huggingface.co/docs/transformers/en/perplexity
    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    max_length = config.training.max_length
    stride = 1024
    individual_perplexities = []

    with torch.no_grad():
        for example in dataset["text"]:
            if consider_answer_only:
                # Split the input into prompt and answer (assuming a specific format)
                try:
                    prompt, answer = example.split("assistant", 1)  # Split at the first "assistant"
                    inputs = tokenizer(answer, return_tensors="pt", padding=True, truncation=True).to(device)
                except ValueError:
                    print(f"Skipping example due to missing 'Assistant:' separator.")
                    continue
            else:
                inputs = tokenizer(example, return_tensors="pt", padding=True, truncation=True).to(device)

            encodings = inputs

            seq_len = encodings.input_ids.size(1)
            prev_end_loc = 0

            example_nll_sum = 0.0  
            example_n_tokens = 0   

            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

                num_valid_tokens = (target_ids != -100).sum().item()
                batch_size = target_ids.size(0)
                num_loss_tokens = num_valid_tokens - batch_size
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

                example_nll_sum += neg_log_likelihood * num_loss_tokens
                example_n_tokens += num_loss_tokens

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            if example_n_tokens > 0:
                example_avg_nll = example_nll_sum / example_n_tokens
                example_perplexity = torch.exp(example_avg_nll).cpu().item()
                individual_perplexities.append(example_perplexity)

    avg_nll = nll_sum / n_tokens
    perplexity = torch.exp(avg_nll).cpu()

    return perplexity.item(), individual_perplexities

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
    mmlu_tasks = ['astronomy']
    num_correct = 0
    total_questions = 0

    for task in mmlu_tasks:
        # Load and subset the MMLU dataset
        dataset = load_dataset("cais/mmlu", task, split="test")
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        for example in dataset:
            question = example["question"]
            choices = example["choices"]
            choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            if isinstance(example["answer"],int):
                correct_answer_index = example["answer"]
            else:
                correct_answer_letter = example["answer"].strip().upper()

                # choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                if correct_answer_letter not in choice_map:
                    print(f"Invalid correct answer format: {correct_answer_letter}")
                    continue  # Skip if the answer format is unexpected

                correct_answer_index = choice_map[correct_answer_letter]

            # Construct the prompt
            prompt = (
                f"Question: {question}\nChoices:\n"
                f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\nAnswer:"
            )

            inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)

            if inputs.input_ids.size(1) >= model.config.max_position_embeddings:
                print("Warning: Prompt was truncated due to length.")

            # Generate model output
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,  
                    do_sample=False     
                )

            # Decode output tokens
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract predicted answer (A, B, C, D) robustly
            match = re.search(r'\b([ABCD])\b', generated_text.upper())
            if match:
                predicted_token = match.group(1)
            else:
                print(f"Invalid token predicted: {generated_text.strip()}")
                predicted_token = 'Invalid'

            if predicted_token in choice_map:
                predicted_index = choice_map[predicted_token]
                if predicted_index == correct_answer_index:
                    num_correct += 1
            else:
                print(f"Skipping due to invalid token: {predicted_token}")

            total_questions += 1
        
        accuracy = num_correct / total_questions if total_questions > 0 else 0.0
        print(f"\nMMLU Accuracy: {accuracy:.2%} ({num_correct}/{total_questions})")
        return accuracy
    
def extract_book_qa_pairs(config, num_pairs=5):
    dataset = load_dataset("deepmind/narrativeqa",split='test')
    target_url = config.data.book_path
    book_qa=[]
    for sample in dataset:
        # print(sample['document'].get('kind', ''))
        if (target_url.split('/')[-2] in sample['document'].get('url', '').lower() and sample['document'].get('kind', '').lower()=="gutenberg"):
            book_qa.append(sample)
    if len(book_qa)==0:
        print(f"No Q&A pairs found for '{target_url}' in the NarrativeQA dataset.\nUsing personally curated dataset")
        with open('/home/c01joch/CISPA-az6/dprune_memorization-2024/LM_FineTune/qa_pairs.json', 'r') as file:
            book_qa = json.load(file)
    if len(book_qa)==0:
        return []

    qa_pairs = []
    for sample in book_qa:
        question = sample["question"]["text"]
        answers = [ans['text'] for ans in sample["answers"]]
        qa_pairs.append((question, answers))

    # Limit to the specified number of Q&A pairs
    return qa_pairs[:num_pairs]

def generate_answer(model, tokenizer, config, question, device):
    
    inputs = tokenizer(question, truncation=True, padding = 'max_length', max_length=config.training.max_length,return_tensors='pt').to(device)
    outputs = model.generate(inputs.input_ids, max_length=config.training.max_length)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def evaluate_qa(model, tokenizer, config, device): #Relying a little extra on ChatGPT for this
    qa_pairs = extract_book_qa_pairs(config)
    references = []
    hypotheses = []
    rouge_l_scores = []
    meteor_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    meteor = evaluate.load("meteor")
    smooth_fn = SmoothingFunction().method2

    for question, reference_answers in qa_pairs:
        # Generate answer
        generated_answer = generate_answer(model, tokenizer, config, question, device)
        # Prepare for BLEU
        references.append([ref.split() for ref in reference_answers])
        hypotheses.append(generated_answer.split())

        rouge_scores = [scorer.score(ref, generated_answer)['rougeL'].fmeasure for ref in reference_answers]
        rouge_l_scores.append(max(rouge_scores))  # Take the max ROUGE-L score across references

        # METEOR score calculation
        for ref in reference_answers:
            meteor_scores.append(meteor.compute(predictions=[generated_answer], references=[ref])['meteor'])

        print(f"\n❓ Question: {question}")
        print(f"✅ Reference Answer: {reference_answers[0]}")
        print(f"🤖 Generated Answer: {generated_answer}")

        # Calculate BLEU scores (1-gram, 2-gram, and 4-gram)
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

    # Calculate average scores
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return bleu1, bleu2, bleu4, avg_rouge_l, avg_meteor, rouge_l_scores, meteor_scores


    
@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer, padding='max_length')

    # Prepare datasets
    book_dataset, tokens_finetuned = prepare_book_dataset(config, tokenizer)
    instruction_dataset = prepare_instruction_dataset(config, tokenizer)

    print("\n--- Evaluation Before Fine-tuning ---")

    initial_exact_match, initial_rouge_l, initial_rouge_l_scores = evaluate_memorization(model, tokenizer, book_dataset, device)
    initial_instruction_perplexity, initial_instruction_perplexities_list = evaluate_perplexity(model, tokenizer, instruction_dataset, device, config)
    initial_bleu1_qa, initial_bleu2_qa, initial_bleu4_qa, initial_avg_rouge_l_qa, initial_avg_meteor_qa, initial_rouge_l_scores_qa, initial_meteor_scores_qa = evaluate_qa(model,tokenizer,config,device)
    initial_results = {
        "instruction_perplexity": initial_instruction_perplexity,
        "instruction_perplexities_list": initial_instruction_perplexities_list,
        "exact_match": initial_exact_match,
        "rouge_L": initial_rouge_l,
        "rouge_L_scores_list": initial_rouge_l_scores,
        "mmlu_accuracy": evaluate_mmlu(model, tokenizer, device, num_samples=config.data.mmlu_samples),
        "QA_bleu1":initial_bleu1_qa,
        "QA_bleu2":initial_bleu2_qa,
        "QA_bleu4":initial_bleu4_qa,
        "QA_rouge_L":initial_avg_rouge_l_qa,
        "QA_rouge_L_scores_list": initial_rouge_l_scores_qa,
        "QA_meteor":initial_avg_meteor_qa,
        "QA_meteor_scores_list": initial_meteor_scores_qa
    }

    for key, value in initial_results.items():
        print(f"Initial {key.replace('_', ' ').title()}: {value}")

    # Training setup
    training_args = TrainingArguments(
        output_dir=config.output.dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        num_train_epochs=config.training.num_epochs,
        save_strategy="epoch",
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    optimizer = get_optimizer(model, config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=book_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None)
    )

    # Display hyperparameters
    print("\n--- Hyperparameter Report ---")
    print(OmegaConf.to_yaml(config))

    print("\n--- Starting Training ---")
    trainer.train()

    # Save model and tokenizer
    model_dir = os.path.join(config.output.dir, "fine_tuned_model")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\nModel saved at {model_dir}")

    print("\n--- Evaluation After Fine-tuning ---")

    final_exact_match, final_rouge_l, final_rouge_l_scores = evaluate_memorization(model, tokenizer, book_dataset, device)
    final_instruction_perplexity, final_instruction_perplexities_list = evaluate_perplexity(model, tokenizer, instruction_dataset, device, config)
    final_bleu1_qa, final_bleu2_qa, final_bleu4_qa, final_avg_rouge_l_qa, final_avg_meteor_qa, final_rouge_l_scores_qa, final_meteor_scores_qa = evaluate_qa(model,tokenizer,config,device)
    final_results = {
        "instruction_perplexity": final_instruction_perplexity,
        "instruction_perplexities_list": final_instruction_perplexities_list,
        "exact_match": final_exact_match,
        "rouge_L": final_rouge_l,
        "rouge_L_scores_list": final_rouge_l_scores,
        "mmlu_accuracy": evaluate_mmlu(model, tokenizer, device, num_samples=config.data.mmlu_samples),
        "QA_bleu1":final_bleu1_qa,
        "QA_bleu2":final_bleu2_qa,
        "QA_bleu4":final_bleu4_qa,
        "QA_rouge_L":final_avg_rouge_l_qa,
        "QA_rouge_L_scores_list":final_rouge_l_scores_qa,
        "QA_meteor":final_avg_meteor_qa,
        "QA_meteor_scores_list":final_meteor_scores_qa
    }

    for key, value in final_results.items():
        print(f"Final {key.replace('_', ' ').title()}: {value}")

    # Perform statistical tests
    print("\n--- Statistical Tests ---")
    try:
        t_stat_perplexity, p_val_perplexity = ttest_rel(
            initial_instruction_perplexities_list, 
            final_instruction_perplexities_list
        )
        print(f"Perplexity t-test: t={t_stat_perplexity:.3f}, p={p_val_perplexity:.3f}")
    except ValueError as e:
        print(f"Error in perplexity t-test: {e}")

    try:
        t_stat_rouge, p_val_rouge = ttest_rel(
            initial_rouge_l_scores, 
            final_rouge_l_scores
        )
        print(f"Rouge-L t-test: t={t_stat_rouge:.3f}, p={p_val_rouge:.3f}")
    except ValueError as e:
        print(f"Error in Rouge-L t-test: {e}")

    try:
        t_stat_rouge_qa, p_val_rouge_qa = ttest_rel(
            initial_rouge_l_scores_qa, 
            final_rouge_l_scores_qa
        )
        print(f"Rouge-L for QA t-test: t={t_stat_rouge_qa:.3f}, p={p_val_rouge_qa:.3f}")
    except ValueError as e:
        print(f"Error in Rouge-L t-test for QA: {e}")

    try:
        t_stat_meteor, p_val_meteor = ttest_rel(
            initial_meteor_scores_qa, 
            final_meteor_scores_qa
        )
        print(f"Meteor t-test: t={t_stat_meteor:.3f}, p={p_val_meteor:.3f}")
    except ValueError as e:
        print(f"Error in meteor t-test: {e}")

    all_stats = {
        "Perplexity t-test":t_stat_perplexity,
        "Perlpexity p-val":p_val_perplexity,
        "RougeL t-test": t_stat_rouge,
        "RougeL p-val": p_val_rouge,
        "RougeL (QA) t-test": t_stat_rouge_qa,
        "RougeL (QA) p-val": p_val_rouge_qa,
        "Meteor (QA) t-test": t_stat_meteor,
        "Meteor (QA) p-val": p_val_meteor
    }

    results = {
        "initial": initial_results,
        "final": final_results,
        "Statistics": all_stats,
        "config": OmegaConf.to_container(config, resolve=True)
    }

    results_path = os.path.join(config.output.dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved at {results_path}")

    # Clear memory
    del model, tokenizer, book_dataset, instruction_dataset
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
