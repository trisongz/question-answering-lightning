import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import pytorch_lightning as pl
import layers
import config
from preprocessing import DataPreprocessor
from utils import dress_for_loss, save_checkpoint, correct_tokens, MetricReporter
from tensorboardX import SummaryWriter
from torchtext import data
import os
import json


# Preprocessing values used for training
prepro_params = {
    "word_embedding_size": config.word_embedding_size,
    "answer_embedding_size": config.answer_embedding_size,
    "max_len_context": config.max_len_context,
    "max_len_question": config.max_len_question,
}

# Hyper-parameters setup
hyper_params = {
    "num_epochs": config.num_epochs,
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "hidden_size": config.hidden_size,
    "n_layers": config.n_layers,
    "drop_prob": config.drop_prob,
    "start_decay_epoch": config.start_decay_epoch,
    "decay_rate": config.decay_rate,
    "use_answer": config.use_answer,
    "cuda": config.cuda,
    "pretrained": config.pretrained
}

experiment_params = {"preprocessing": prepro_params, "model": hyper_params}

torch.manual_seed(42)


# Define a path to save experiment logs
experiment_path = "/home/jupyter/output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# Save the preprocesisng and model parameters used for this training experiment
with open(os.path.join(experiment_path, "config_{}.json".format(config.exp)), "w") as f:
    json.dump(experiment_params, f)

# Start TensorboardX writer
writer = SummaryWriter(experiment_path)

# Create an object to report the different metrics
mc = MetricReporter()

dp = DataPreprocessor()
train_dataset, valid_dataset, vocabs = dp.load_data(os.path.join(config.out_dir, "train-dataset.pt"),
                                                    os.path.join(config.out_dir, "dev-dataset.pt"),
                                                    config.glove)

padding_idx = vocabs['trg_vocab'].stoi["<PAD>"]
criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")

class Seq2Seq(pl.LightningModule):
    def __init__(self, in_vocab, hidden_size, n_layers, trg_vocab, device, drop_prob=0., use_answer=True):
        super(Seq2Seq, self).__init__()

        self.enc = layers.Encoder(input_size=in_vocab.vectors.size(1) if not use_answer else in_vocab.vectors.size(1) +
                                  config.answer_embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=n_layers,
                                  word_vectors=in_vocab.vectors,
                                  bidirectional=True,
                                  drop_prob=drop_prob if n_layers > 1 else 0.)

        self.dec = layers.Decoder(input_size=in_vocab.vectors.size(1) + hidden_size,
                                  hidden_size=hidden_size,
                                  word_vectors=in_vocab.vectors,
                                  trg_vocab=trg_vocab,
                                  n_layers=n_layers,
                                  device=device,
                                  dropout=drop_prob if n_layers > 1 else 0.,
                                  attention=True)

    def forward(self, sentence, sentence_len, question=None, answer=None):
        enc_output, enc_hidden = self.enc(sentence, sentence_len, answer)
        outputs = self.dec(enc_output, enc_hidden, question)

        return outputs
    
    def train_dataloader(self):
        # Load the data into datasets of mini-batches
        train_dataloader = data.BucketIterator(train_dataset,
                                    batch_size=hyper_params["batch_size"],
                                    sort_key=lambda x: len(x.src),
                                    sort_within_batch=True,
                                    device=device,
                                    shuffle=False)
        
        print("Length of training data loader is:", len(train_dataloader))
        return train_dataloader

    def valid_dataloader(self):
        valid_dataloader = data.BucketIterator(valid_dataset,
                                    batch_size=hyper_params["batch_size"],
                                    sort_key=lambda x: len(x.src),
                                    sort_within_batch=True,
                                    device=device,
                                    shuffle=True)
        
        print("Length of valid data loader is:", len(valid_dataloader))
        return valid_dataloader
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), hyper_params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=list(range(hyper_params["start_decay_epoch"],
                                                                               hyper_params["num_epochs"] + 1)),
                                                         gamma=hyper_params["decay_rate"])
        
        return optimizer, scheduler
        
    def training_step(self, batch, i):
        mc.train()
        sentence, len_sentence, question = batch.src[0], batch.src[1], batch.trg[0]
        answer = batch.feat if hyper_params["use_answer"] else None
        pred = self.forward(sentence, len_sentence, question, answer)
        
        # Stack the predictions into a tensor to compute the loss
        pred = dress_for_loss(pred)
        # Calculate Loss: softmax --> negative log likelihood
        loss = criterion(pred.view(-1, pred.size(2)), question[:, 1:].contiguous().view(-1))
        # Update the metrics
        num_non_padding, num_correct = correct_tokens(pred, question, padding_idx)
        mc.update_metrics(loss.item(), num_non_padding, num_correct)
        # Truncate the gradients if the norm is greater than a threshold
        clip_grad_norm_(self.parameters(), 5.)
        
        return loss
    
    def validation_step(self, batch, i):
        mc.eval()
        sentence, len_sentence, question = batch.src[0], batch.src[1], batch.trg[0]
        answer = batch.feat if hyper_params["use_answer"] else None
        # Forward pass to get output/logits
        pred = model(sentence, len_sentence, question,  answer)
        # Stack the predictions into a tensor to compute the loss
        pred = dress_for_loss(pred)
        # Calculate Loss: softmax --> negative log likelihood
        loss = criterion(pred.view(-1, pred.size(2)), question.view(-1))

        # Update the metrics
        num_non_padding, num_correct = correct_tokens(pred, question, padding_idx)
        mc.update_metrics(loss.item(), num_non_padding, num_correct)
        
        
        return loss
        
    def training_end(self, outputs):
        mc.report_metrics()
        writer.add_scalars("valid", {"loss": mc.list_valid_loss[-1],
                                     "accuracy": mc.list_valid_accuracy[-1],
                                     "perplexity": mc.list_valid_perplexity[-1],
                                     "epoch": mc.epoch})
    
    def validation_end(self, outputs):
        mc.report_metrics()
        writer.add_scalars("valid", {"loss": mc.list_valid_loss[-1],
                                     "accuracy": mc.list_valid_accuracy[-1],
                                     "perplexity": mc.list_valid_perplexity[-1],
                                     "epoch": mc.epoch})