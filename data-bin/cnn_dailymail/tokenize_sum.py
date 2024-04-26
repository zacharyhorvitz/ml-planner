from datasets import load_from_disk
from transformers import BertTokenizer, GPT2Tokenizer
from multiprocessing import Pool


# load from the disk
data_path = '/mnt/swordfish-pool2/horvitz/cnn_dailymail'
dataset = load_from_disk(data_path)

# print the first 5 examples
print(dataset['train'][:5])


bert_toker = BertTokenizer.from_pretrained('bert-base-uncased')
gpt2_toker = GPT2Tokenizer.from_pretrained('gpt2')

# tokenize the highlights in the dataset

def add_tokenized_highlights(sample, max_len=256):
    # import pdb; pdb.set_trace()
    sample['bert_ids'] = bert_toker(sample['highlights'], max_length=max_len, truncation=True)['input_ids']
    sample['gpt2_ids'] = gpt2_toker(sample['highlights'], max_length=max_len, truncation=True)['input_ids']
    # import pdb; pdb.set_trace()
    return sample

MAX_LEN = 256
# apply the function to the dataset
dataset = dataset.map(lambda x: add_tokenized_highlights(x, max_len=MAX_LEN), batched=True, batch_size=128)

# save the dataset
dataset.save_to_disk(data_path+'_sum_tokenized')