# experiment ID
exp = "qg-1"

# data directories
newsqa_data_dir = "/content/dataset/newsqa/newsqa-data-v1"
squad_data_dir = "/content/dataset/squad/"
out_dir = "/content/saved_models/qg/"
train_dir = squad_data_dir + "train/"
dev_dir = squad_data_dir + "dev/"

# model paths
spacy_en = "/usr/local/lib/python3.6/dist-packages/en_core_web_sm/en_core_web_sm-2.1.0"
#glove = "/content/saved_models/glove.6B/"
glove = "/content/saved_models/glove/"
squad_models = "/content/saved_models/squad/models/"

# preprocessing values
paragraph = True
min_len_context = 5
max_len_context = 100 if not paragraph else 1000
min_len_question = 5
max_len_question = 20
word_embedding_size = 300
answer_embedding_size = 2
in_vocab_size = 45000
out_vocab_size = 28000

# training hyper-parameters
num_epochs = 50
batch_size = 32
learning_rate = 0.00005
hidden_size = 1600
n_layers = 24
drop_prob = 0.3
start_decay_epoch = 8
decay_rate = 0.5
use_answer = True
cuda = False
pretrained = False

# eval hyper-parameters
eval_batch_size = 1
min_len_sentence = 5
top_k = 0.
top_p = 0.9
temperature = 0.7
decode_type = "topk"
