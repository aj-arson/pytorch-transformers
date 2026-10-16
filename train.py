import os
import torch
import time
from Transformer import Transformer, config
from tokenizer.tokenizer import tokenizer
# from Logger import log, close_logger
from dataloader import get_data_loaders

def calculate_loss(loss_fn, preds, targets):
    return loss_fn(preds.reshape(-1,tokenizer.vocab_size),targets.reshape(-1))

def step(model, encoder_input, decoder_input, decoder_output, loss_fn, optimizer):
    optimizer.zero_grad()
    if config.device == 'cuda':
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            preds = model(encoder_input, decoder_input)
            step_loss = calculate_loss(loss_fn, preds, decoder_output)
    else:
        preds = model(encoder_input, decoder_input)
        step_loss = calculate_loss(loss_fn, preds, decoder_output)
    step_loss.backward()
    optimizer.step()
    return step_loss

def model_eval(epoch, iter, model, train_data_loader, val_data_loader, optimizer, max_eval_iters, best_val_loss, logger=None):
    print(f"{'='*20} Model Evaluation at {iter}: {'='*20}")
    model.eval()
    losses = {'train_loss':0., 'val_loss':0.}
    train_num_batches = 0
    val_num_batches = 0

    for train_iter, (encoder_input, decoder_input, decoder_output) in enumerate(train_data_loader):
        if train_iter <= max_eval_iters:
            with torch.no_grad():
                preds = model(encoder_input, decoder_input)
                loss = calculate_loss(loss_fn, preds, decoder_output)
                losses['train_loss'] += loss
                train_num_batches += 1
        else:
            break
    losses['train_loss'] = losses['train_loss']/max(1,train_num_batches)

    for val_iter, (encoder_input, decoder_input, decoder_output) in enumerate(val_data_loader):
            if val_iter <= max_eval_iters:
                with torch.no_grad():
                    preds = model(encoder_input, decoder_input)
                    loss = calculate_loss(loss_fn, preds, decoder_output)
                    losses['val_loss'] += loss
                    val_num_batches += 1
            else:
                break
    losses['val_loss'] = losses['val_loss']/max(1,val_num_batches)

    print("Previous best value: ", best_val_loss)
    if best_val_loss is None or losses['val_loss'] < best_val_loss:
            best_val_loss = losses['val_loss']
            save_checkpoint(model, optimizer, iter, epoch, best_val_loss)

    if not logger:
        print(f"Train Loss: {losses['train_loss'].item()} | Val Loss: {losses['val_loss'].item()}")
        print("="*len(f" Model Evaluation at {iter}: "))
    else:
        logger(tag="Loss/train", scaler_value=losses['train_loss'], global_step=iter)
        logger(tag="Loss/val", scaler_value=losses['val_loss'], global_step=iter)
    
    model.train()
    return best_val_loss

def train(model:torch.nn.Module, data_loader, num_epochs, eval_interval, max_eval_iters, loss_fn, optimizer:torch.optim.Adam, checkpoint_path="best_val_loss.pt"):
    model.train()
    
    print("Model Training Started...")
    if os.path.exists(checkpoint_path):
        print(f"Loading model and optimizer from checkpoint: {checkpoint_path}")
        loaded_checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        best_val_loss = loaded_checkpoint['best_val_loss']
    else:
        best_val_loss = None
        print("Loading a brand new model and optimizer.")
    
    for epoch in range(num_epochs):
        for iter, (encoder_input, decoder_input, decoder_output) in enumerate(data_loader['train'], start=1):
            step(model, encoder_input, decoder_input, decoder_output, loss_fn, optimizer)
            if iter%eval_interval == 0:
                best_val_loss = model_eval(epoch, iter, model, data_loader['train'], data_loader['val'], optimizer, max_eval_iters, best_val_loss)
    # close_logger()

def save_checkpoint(model, optimizer, epoch, iter_, best_val_loss, checkpoint_path="best_val_loss.pt"):
    checkpoint = {"model_state_dict":model.state_dict(),
                  "optimizer_state_dict":optimizer.state_dict(),
                  "epoch": epoch,
                  "iter":iter_,
                  "best_val_loss":best_val_loss}
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint with best_val_loss at: {checkpoint_path}")

if __name__ == "__main__":
    model = Transformer()
    model = model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    data_loaders = get_data_loaders()
    train(model, data_loaders, config.epochs, config.eval_interval, config.max_eval_iters, loss_fn, optimizer)