# external libraries
import os
import json
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torchtext import data
from tensorboardX import SummaryWriter
import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.init(name='', project='QGeneration', sync_tensorboard=True)
import pytorch_lightning as pl
# internal utilities
import config
from model import Seq2Seq, mc, vocabs

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


device = torch.device("cuda" if hyper_params["cuda"] else "cpu")

# Load the model
model = Seq2Seq(in_vocab=vocabs["src_vocab"],
                hidden_size=hyper_params["hidden_size"],
                n_layers=hyper_params["n_layers"],
                trg_vocab=vocabs['trg_vocab'],
                device=device,
                drop_prob=hyper_params["drop_prob"],
                use_answer=hyper_params["use_answer"])

# Resume training if checkpoint
if hyper_params["pretrained"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
#model.to(device)

# Get the best loss so far when resuming training
if hyper_params["pretrained"]:
    best_valid_loss = torch.load(os.path.join(experiment_path, "model.pkl"))["best_valid_loss"]
    epoch_checkpoint = torch.load(os.path.join(experiment_path, "model_last_checkpoint.pkl"))["epoch"]
    print("Best validation loss obtained after {} epochs is: {}".format(epoch_checkpoint, best_valid_loss))
else:
    best_valid_loss = 10000  # large number
    epoch_checkpoint = 1

# Train the model
print("Starting training...")
wandb_logger = WandbLogger(name='TPU-Iteration1', project='QGeneration')
wandb_logger.watch(model, log_freq=100)
trainer = pl.Trainer(num_tpu_cores=8, max_epochs=hyper_params["num_epochs"], logger=wandb_logger)
trainer.fit(model)

# Save last model weights
save_checkpoint({
    "epoch": mc.epoch + epoch_checkpoint,
    "state_dict": model.state_dict(),
    "best_valid_loss": mc.list_valid_loss[-1],
}, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))

# Save model weights with best validation error
is_best = bool(mc.list_valid_loss[-1] < best_valid_loss)
best_valid_loss = min(mc.list_valid_loss, best_valid_loss)
save_checkpoint({
    "epoch": mc.epoch + epoch_checkpoint,
    "state_dict": model.state_dict(),
    "best_valid_loss": best_valid_loss
}, is_best, os.path.join(experiment_path, "model.pkl"))

# Export scalar data to TXT file for external processing and analysis
mc.log_metrics(os.path.join(experiment_path, "train_log.txt"))
