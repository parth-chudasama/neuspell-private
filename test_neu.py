from neuspell import BertChecker
from neuspell.seq_modeling.helpers import load_data, train_validation_split
from neuspell.seq_modeling.helpers import get_tokens
from neuspell import BertChecker
from neuspell import ElmosclstmChecker
# Step-0: Load your train and test files, create a validation split
train_data = load_data(data_dir, clean_file, corrupt_file)
train_data, valid_data = train_validation_split(train_data, 0.8, seed=11690)

# Step-1: Create vocab file. This serves as the target vocab file and we use the defined model's default huggingface
# tokenizer to tokenize inputs appropriately.
vocab = get_tokens([i[0] for i in train_data], keep_simple=True, min_max_freq=(1, float("inf")), topk=100000)

# # Step-2: Initialize a model
checker = BertChecker(device="cuda")
checker.from_huggingface(bert_pretrained_name_or_path="distilbert-base-cased", vocab=vocab)

# Step-3: Finetune the model on your dataset
checker.finetune(clean_file=clean_file, corrupt_file=corrupt_file, data_dir=data_dir)