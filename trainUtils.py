import math

from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from params import *

USE_TENSORBOARD = True
if(USE_TENSORBOARD):
    tensorboard_summary = SummaryWriter()

class LrStepTracker:

    def __init__(self, model_dim=512, warmup_steps=4000, init_steps=0):
        # Store Values
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        self.init_steps = init_steps

        # Begin Calculations
        self.invsqrt_dim = (1 / math.sqrt(model_dim))
        self.invsqrt_warmup = (1 / (warmup_steps * math.sqrt(warmup_steps)))

    # step
    def step(self, step):

        step += self.init_steps
        if(step <= self.warmup_steps):
            return self.invsqrt_dim * self.invsqrt_warmup * step
        else:
            invsqrt_step = (1 / math.sqrt(step))
            return self.invsqrt_dim * invsqrt_step

def train(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, save_checkpoint_steps=1000, tensorboard_steps=200, device=None):
    loss_hist = []
    save_steps = 0
    out = -1
 
    model.train()
    with tqdm(total=len(dataloader)) as bar_train:
        for batch_num, batch in enumerate(dataloader):
            #time_before = time.time()

            x   = batch[0].to(device) # (2,2048)
            tgt = batch[1].to(device) # (2, 2048)

            y= model(x) # (2, 2048, 512)

            y   = y.reshape(y.shape[0] * y.shape[1], -1) # shape(4096,512) , dictionary=512
            tgt = tgt.flatten() # shape(4096) seq_len x batch_size

            out = loss.forward(y, tgt)

            out.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()

            if(lr_scheduler is not None):
                lr_scheduler.step()

            lr = opt.param_groups[0]['lr']
            bar_train.set_description(f'Epoch: {cur_epoch} Loss: {float(out):.4} LR: {float(lr):.8}')
            bar_train.update(1)
            loss_hist.append(out.item())

            
            if save_steps % save_checkpoint_steps == 0:
                print('Saving model progress. Please wait...')
                print('gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')
                torch.save(model.state_dict(), './SaveModel/gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')

            if save_steps % tensorboard_steps == 0:
                total_steps =cur_epoch*len(dataloader)+save_steps
                if(USE_TENSORBOARD):                
                    tensorboard_summary.add_scalar("train_loss", out.item(), total_steps)
                    tensorboard_summary.add_scalar("lr", lr, total_steps)

            save_steps +=1
            
    return loss_hist


def compute_accuracy(out, tgt):
    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(torch.float32)

    acc = num_right / len(tgt)

    return acc


def train_and_eval(cur_epoch, model, dataloader, dataloader_eval, loss, loss_eval, opt, lr_scheduler=None, save_checkpoint_steps=1000, tensorboard_steps=200, device=None):
    loss_hist = []
    save_steps = 0
    out = -1
 
    model.train()
    with tqdm(total=len(dataloader)) as bar_train:
        for batch_num, batch in enumerate(dataloader):
            #time_before = time.time()

            x   = batch[0].to(device) # (2,2048)
            tgt = batch[1].to(device) # (2, 2048)

            y= model(x) # (2, 2048, 512)

            y   = y.reshape(y.shape[0] * y.shape[1], -1) # shape(4096,512) , dictionary=512
            tgt = tgt.flatten() # shape(4096) seq_len x batch_size

            out = loss.forward(y, tgt)

            out.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()

            if(lr_scheduler is not None):
                lr_scheduler.step()

            lr = opt.param_groups[0]['lr']
            bar_train.set_description(f'Epoch: {cur_epoch} Loss: {float(out):.4}')# LR: {float(lr):.8}')
            bar_train.update(1)
            #loss_hist.append(out.item())

            
            if save_steps % save_checkpoint_steps == 0:
                print('Saving model progress. Please wait...')
                print('gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')
                torch.save(model.state_dict(), './SaveModel/gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')

            if save_steps % tensorboard_steps == 0:
                total_steps =cur_epoch*len(dataloader)+save_steps
                if(USE_TENSORBOARD):                
                    tensorboard_summary.add_scalar("train_loss", out.item(), total_steps)
                    tensorboard_summary.add_scalar("lr", lr, total_steps)
               # test accuracy
            try:
                    batch = next(iter(dataloader_eval)) # extract X,y from test dataloader
            except StopIteration:
                    dataloader_eval_iter = iter(dataloader_eval)
                    batch = next(iter(dataloader_eval_iter)) # extract X,y from test dataloader
       
            x   = batch[0].to(device) # (2,2048)
            tgt = batch[1].to(device) # (2, 2048)

            model.eval()
            with torch.no_grad(): # deactivates autograd
                y= model(x) # (2, 2048, 512)

                y   = y.reshape(y.shape[0] * y.shape[1], -1) # shape(4096,512) , dictionary=512
                tgt = tgt.flatten() # shape(4096) seq_len x batch_size
                out = loss_eval.forward(y, tgt)

            if save_steps % tensorboard_steps == 0:
                if(USE_TENSORBOARD):                
                    accuracy = float(compute_accuracy(y, tgt))
                    tensorboard_summary.add_scalar("eval_loss", out.item(), total_steps)
                    tensorboard_summary.add_scalar("accuracy", accuracy, total_steps)
            model.train()
                
            save_steps +=1

            
    return loss_hist