from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/transformers')

def log(tag, scaler_value, global_step):
    writer.add_scalar(tag, scaler_value, global_step)

def close_logger():
    writer.close()