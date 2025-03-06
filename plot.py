import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

with open('first_3000/evaluation_results1.json', 'r') as f:
    results = json.load(f)

plt.figure(figsize=(8, 6))
perplexities = [results['initial']['instruction_perplexity'], results['final']['instruction_perplexity']]
bars = plt.bar(['Before Fine-tuning', 'After Fine-tuning'], perplexities)
plt.title('Perplexity computed on Instruction Dataset Before and After Fine-tuning')
plt.ylabel('Perplexity')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.4f}"))
# plt.savefig('Perplexity_instruction.png')
# plt.show()
plt.tight_layout()
plt.savefig('first_3000/perplexity_instruction.png', bbox_inches='tight')

metrics = ['rouge_l', 'exact_match']
before = [results['initial'][m] for m in metrics]
after = [results['final'][m] for m in metrics]

df = pd.DataFrame({
    'Metric': metrics * 2,
    'Value': before + after,
    'Stage': ['Before'] * 2 + ['After'] * 2
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Value', hue='Stage', data=df)
plt.title('RougeL and Exact Match Before and After Fine-tuning')
# plt.show()
plt.tight_layout()
plt.savefig('first_3000/Avg_RougeL_ExactMatch.png', bbox_inches='tight')
# plt.figure(figsize=(8, 6))
# sns.histplot(results['final']['rouge_l_scores'])
# plt.title('Histogram of RougeL Scores After Fine-tuning')
# plt.xlabel('RougeL Score')
# plt.ylabel('Frequency')
# plt.show()

##
# 2. Overlapping histograms for before and after
plt.figure(figsize=(8, 6))
sns.histplot(results['initial']['rouge_l_scores'], color='blue', alpha=0.2, label='Before Fine-tuning')
sns.histplot(results['final']['rouge_l_scores'], color='red', alpha=0.2, label='After Fine-tuning')
plt.title('Histogram of RougeL Scores Before and After Fine-tuning')
plt.xlabel('RougeL Score')
plt.ylabel('Frequency')
plt.legend()
# plt.show()
plt.tight_layout()
plt.savefig('first_3000/RougeL_hist.png', bbox_inches='tight')

# plt.figure(figsize=(8, 6))
# sns.histplot(results['initial']['instruction_perplexities'], color='blue', alpha=0.2, label='Before Fine-tuning')
# sns.histplot(results['final']['instruction_perplexities'], color='red', alpha=0.2, label='After Fine-tuning')
# plt.title('Histogram of instruction_perplexities Before and After Fine-tuning')
# plt.xlabel('Perlexity')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()
# plt.tight_layout()
# # plt.savefig('first_3000/RougeL_hist.png', bbox_inches='tight')