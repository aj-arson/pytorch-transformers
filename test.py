
# # def save_checkpoint(model, optimizer, epoch, loss):
# #     checkpoint = {
# #         "model_state":model.state_dict,
# #         "optimizer_state":optimizer.state_dict,
# #         "epoch":epoch,
# #         "loss":loss,
# #         "min_val_loss":min_val_loss
# #     }
#     # torch.save(checkpoint, 'saved_checkpoint.pt')

# # Downloading Data

# # Dataset & DataLoader



# train_dataset = NMT_DATASET('train', dataset, tokenizer, config)
# val_dataset = NMT_DATASET('val', dataset, tokenizer, config)
# test_dataset = NMT_DATASET('test', dataset, tokenizer, config)

# train_data_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
# val_data_loader = DataLoader(train_dataset, batch_size=config.val_batch_size, shuffle=False)
# test_data_loader = DataLoader(train_dataset, batch_size=config.test_batch_size, shuffle=False)

# data_loaders = {'train':train_data_loader,
#                'val':val_data_loader,
#                'test':test_data_loader}

# def train(model, data_loaders, loss_fn, optimizer, config, debug):
#     if debug:
#         print("Running in DEBUG mode.")
#     else:
#         print("Starting full on training.")
#     model.train()
#     for epoch in range(config.epochs):
#         trainer(model, data_loaders, loss_fn, optimizer, config, epoch, debug)
#     model.eval()

# model = Transformer(config)
# model.to(config.device)
# loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer._special_tokens[pad_token])
# optimizer = torch.optim.AdamW(model.parameters(), lr=config.)
# debug = False

# # Model Training

# train(model, data_loaders, loss_fn, optimizer, config, debug)

# model.state_dict