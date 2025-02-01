import math

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from params import *

USE_TENSORBOARD = True
if(USE_TENSORBOARD):
    tensorboard_summary = SummaryWriter()

def train(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, save_checkpoint_steps=1000, tensorboard_steps=200, device=None):
    loss_hist = []
    save_steps = 0
    out = -1
 
    model.train()
    with tqdm(total=len(dataloader)) as bar_train:
        for batch_num, batch in enumerate(dataloader):
            x, tgt = batch  # Now x and tgt are dictionaries

            # Move all tensors to device
            x = {k: v.to(device) for k, v in x.items()}
            tgt = tgt.to(device)  # Move target pitches to device
            
            y,e = model(x) # (2, 2048, 512)  # Model outputs pitch logits and encoder outputs

            y   = y.reshape(y.shape[0] * y.shape[1], -1) #  # [B * T, pitch_vocab_size] shape(4096,512) , dictionary=512
            tgt = tgt.flatten() # shape(4096) # [B * T]

            # Compute reconstruction loss (cross entropy between predicted and true pitches)
            #loss_recons = loss.forward(y, tgt)
            # Compute losses and update params
            # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
            loss_recons = loss.forward(y, tgt) 
            #loss_recons = F.cross_entropy(y.view(-1, PIANO_NUM_KEYS), tgt.view(-1)) 

            # Regularize to encourage encoder to output in range [-1, 1]
            # This implements Lmargin = Σ max(|encs(x)| − 1, 0)²:
            # Penalizes encoder outputs that fall outside [-1, 1] range
            # torch.abs(pre_iq_encoding) - 1: How far values are from the [-1,1] boundary
            # torch.maximum(..., 0): Only penalize values outside the range
            # torch.square: Square the penalty (makes loss smoother)
            loss_margin = torch.square(
                torch.maximum(torch.abs(e) - 1, torch.zeros_like(e))
            ).mean()
            
            # Calculate contour penalty
            #"We also contribute a musically motivated regularization strategy which gives the model an 
            # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
            # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
            # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
            # of the input melodic contours."
            
            # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
            # Encourages button intervals to match piano note intervals in direction

            # Calculate differences between consecutive notes/latents
            # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
            # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        
            # Penalizes when the product/quotient is less than the margin
            loss_contour = torch.square(
                torch.maximum(
                    1 - torch.diff(x['pitch'], dim=1).float() * torch.diff(e, dim=1),
                    torch.zeros_like(e[:, 1:])
                )
            ).mean()

            # Deviate Penalty
            # Identifies when the same note is held (no pitch change)
            # Penalizes any change in button values during held notes
            # Helps maintain consistency in the mapping
            # Identify held notes (where consecutive pitches are the same)
            notes_held = (x['pitch'][:, 1:] == x['pitch'][:, :-1]).float()
            # Penalize button changes when notes are held
            loss_deviate = torch.square(
                torch.diff(e, dim=1) * notes_held  # button contour * notes held
            ).mean()

            # Total loss
            loss_total = torch.zeros_like(loss_recons)
            loss_total += loss_recons
            if LOSS_MARGIN_MULTIPLIER > 0:
                loss_total += LOSS_MARGIN_MULTIPLIER * loss_margin
            if LOSS_CONTOUR_MULTIPLIER > 0:
                loss_total += LOSS_CONTOUR_MULTIPLIER * loss_contour
            if LOSS_DEVIATE_MULTIPLIER > 0:
                loss_total += LOSS_DEVIATE_MULTIPLIER * loss_deviate

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()

            if(lr_scheduler is not None):
                lr_scheduler.step()

            lr = opt.param_groups[0]['lr']
            bar_train.set_description(f'Epoch: {cur_epoch} Loss: {float(loss_total):.4} LR: {float(lr):.8}')
            bar_train.update(1)
            loss_hist.append(loss_total.item())

            
            if save_steps % save_checkpoint_steps == 0:
                print('Saving model progress. Please wait...')
                print('gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')
                torch.save(model.state_dict(), './SaveModel/gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')

            if save_steps % tensorboard_steps == 0:
                total_steps =cur_epoch*len(dataloader)+save_steps
                if(USE_TENSORBOARD):                
                    tensorboard_summary.add_scalar("loss_recons", loss_recons.item(), save_steps)
                    tensorboard_summary.add_scalar("loss_margin", loss_margin.item(),save_steps)
                    tensorboard_summary.add_scalar("loss_contour", loss_contour.item(), save_steps)                   
                    tensorboard_summary.add_scalar("loss_deviate", loss_deviate.item(), save_steps)                   
                    tensorboard_summary.add_scalar("loss", loss_total.item(), save_steps) 
                    perplexity = calculate_perplexity(e, NUM_BUTTONS)
                    tensorboard_summary.add_scalar("perplexity", perplexity.item(), save_steps)

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
            x, tgt = batch  # Now x and tgt are dictionaries

            # Move all tensors to device
            x = {k: v.to(device) for k, v in x.items()}
            tgt = tgt.to(device)  # Move target pitches to device
            
            y,e = model(x) # (2, 2048, 512)  # Model outputs pitch logits and encoder outputs

            y   = y.reshape(y.shape[0] * y.shape[1], -1) #  # [B * T, pitch_vocab_size] shape(4096,512) , dictionary=512
            tgt = tgt.flatten() # shape(4096) # [B * T]

            # Compute reconstruction loss (cross entropy between predicted and true pitches)
            #loss_recons = loss.forward(y, tgt)
            # Compute losses and update params
            # loss_recons = cross entropy loss between predicted pitch sample list and true pitch sample list
            loss_recons = loss.forward(y, tgt) 
            #loss_recons = F.cross_entropy(y.view(-1, PIANO_NUM_KEYS), tgt.view(-1)) 

            # Regularize to encourage encoder to output in range [-1, 1]
            # This implements Lmargin = Σ max(|encs(x)| − 1, 0)²:
            # Penalizes encoder outputs that fall outside [-1, 1] range
            # torch.abs(pre_iq_encoding) - 1: How far values are from the [-1,1] boundary
            # torch.maximum(..., 0): Only penalize values outside the range
            # torch.square: Square the penalty (makes loss smoother)
            loss_margin = torch.square(
                torch.maximum(torch.abs(e) - 1, torch.zeros_like(e))
            ).mean()
            
            # Calculate contour penalty
            #"We also contribute a musically motivated regularization strategy which gives the model an 
            # awareness of melodic contour. By comparing the finite differences (musical intervals in semitones) 
            # of the input ∆x to the finite differences of the real-valued encoder output ∆encs(x), 
            # the Lcontour term encourages the encoder to produce "button contours" that match the shape 
            # of the input melodic contours."
            
            # This implements Lcontour = Σ max(1 − ∆x∆encs(x), 0)²:
            # Encourages button intervals to match piano note intervals in direction

            # Calculate differences between consecutive notes/latents
            # torch.diff(e, dim=1) = ∆encs(x) = e[:, 1:] - e[:, :-1]  # Button intervals
            # torch.diff(k, dim=1) = ∆x = (k[:, 1:] - k[:, :-1]).float()  # Piano note intervals
        
            # Penalizes when the product/quotient is less than the margin
            loss_contour = torch.square(
                torch.maximum(
                    1 - torch.diff(x['pitch'], dim=1).float() * torch.diff(e, dim=1),
                    torch.zeros_like(e[:, 1:])
                )
            ).mean()

            # Deviate Penalty
            # Identifies when the same note is held (no pitch change)
            # Penalizes any change in button values during held notes
            # Helps maintain consistency in the mapping
            # Identify held notes (where consecutive pitches are the same)
            notes_held = (x['pitch'][:, 1:] == x['pitch'][:, :-1]).float()
            # Penalize button changes when notes are held
            loss_deviate = torch.square(
                torch.diff(e, dim=1) * notes_held  # button contour * notes held
            ).mean()

            # Total loss
            loss_total = torch.zeros_like(loss_recons)
            loss_total += loss_recons
            if LOSS_MARGIN_MULTIPLIER > 0:
                loss_total += LOSS_MARGIN_MULTIPLIER * loss_margin
            if LOSS_CONTOUR_MULTIPLIER > 0:
                loss_total += LOSS_CONTOUR_MULTIPLIER * loss_contour
            if LOSS_DEVIATE_MULTIPLIER > 0:
                loss_total += LOSS_DEVIATE_MULTIPLIER * loss_deviate

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()

            if(lr_scheduler is not None):
                lr_scheduler.step()

            lr = opt.param_groups[0]['lr']
            bar_train.set_description(f'Epoch: {cur_epoch} Loss: {float(loss_total):.4} LR: {float(lr):.8}')
            bar_train.update(1)
            loss_hist.append(loss_total.item())
            
            if save_steps % save_checkpoint_steps == 0:
                total_steps =int(((cur_epoch-1)*len(dataloader)+save_steps )/tensorboard_steps)
                print('Saving model progress. Please wait...')
                print('gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(total_steps) + '_steps_' + str(round(float(loss_total), 4)) + '_t_loss.pth')
                torch.save(model.state_dict(), './SaveModel/gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(total_steps) + '_steps_' + str(round(float(loss_total), 4)) + '_t_loss.pth')

            if save_steps % tensorboard_steps == 0:
                total_steps =int(((cur_epoch-1)*len(dataloader)+save_steps )/tensorboard_steps)
                if(USE_TENSORBOARD):                
                    tensorboard_summary.add_scalar("loss_recons", loss_recons.item(), total_steps)
                    tensorboard_summary.add_scalar("loss_margin", loss_margin.item(),total_steps)
                    tensorboard_summary.add_scalar("loss_contour", loss_contour.item(), total_steps)                   
                    tensorboard_summary.add_scalar("loss_deviate", loss_deviate.item(), total_steps)                   
                    tensorboard_summary.add_scalar("loss", loss_total.item(), total_steps) 
                    perplexity = calculate_perplexity(e, NUM_BUTTONS)
                    tensorboard_summary.add_scalar("perplexity", perplexity.item(), total_steps)

                #### EVALUATION ####
               # Get a single batch from evaluation dataloader
                try:
                    batch = next(iter(dataloader_eval))
                except StopIteration:  # In case the iterator is exhausted
                    dataloader_eval_iter = iter(dataloader_eval)
                    batch = next(dataloader_eval_iter)
                #for batch_num, batch in enumerate(dataloader_eval):
                x, tgt = batch  # Now x and tgt are dictionaries

                # Move all tensors to device
                x = {k: v.to(device) for k, v in x.items()}
                tgt = tgt.to(device)  # Move target pitches to device


                model.eval()
                with torch.no_grad(): # deactivates autograd
                    y,e = model(x) # (2, 2048, 512)  # Model outputs pitch logits and encoder outputs

                    y   = y.reshape(y.shape[0] * y.shape[1], -1) #  # [B * T, pitch_vocab_size] shape(4096,512) , dictionary=512
                    tgt = tgt.flatten() # shape(4096) # [B * T]
    
                    loss_recons_eval = loss_eval.forward(y, tgt) 
                    loss_margin_eval = torch.square(
                        torch.maximum(torch.abs(e) - 1, torch.zeros_like(e))
                    ).mean()
                    loss_contour_eval = torch.square(
                        torch.maximum(
                            1 - torch.diff(x['pitch'], dim=1).float() * torch.diff(e, dim=1),
                            torch.zeros_like(e[:, 1:])
                        )
                    ).mean()

                    notes_held_eval = (x['pitch'][:, 1:] == x['pitch'][:, :-1]).float()
                    # Penalize button changes when notes are held
                    loss_deviate_eval = torch.square(
                        torch.diff(e, dim=1) * notes_held_eval  # button contour * notes held
                    ).mean()

                    # Total loss
                    loss_total_eval = torch.zeros_like(loss_recons_eval)
                    loss_total_eval += loss_recons_eval
                    if LOSS_MARGIN_MULTIPLIER > 0:
                        loss_total_eval += LOSS_MARGIN_MULTIPLIER * loss_margin_eval
                    if LOSS_CONTOUR_MULTIPLIER > 0:
                        loss_total_eval += LOSS_CONTOUR_MULTIPLIER * loss_contour_eval
                    if LOSS_DEVIATE_MULTIPLIER > 0:
                        loss_total_eval += LOSS_DEVIATE_MULTIPLIER * loss_deviate_eval

                    #tensorboard_summary.add_scalar("eval_loss", loss_total_eval.item(), save_steps)
                
                    if(USE_TENSORBOARD):                
                        tensorboard_summary.add_scalar("loss_recons_eval", loss_recons_eval.item(), total_steps)
                        tensorboard_summary.add_scalar("loss_margin_eval", loss_margin_eval.item(),total_steps)
                        tensorboard_summary.add_scalar("loss_contour_eval", loss_contour_eval.item(), total_steps)                   
                        tensorboard_summary.add_scalar("loss_deviate_eval", loss_deviate_eval.item(), total_steps)                   
                        tensorboard_summary.add_scalar("loss_total_eval", loss_total_eval.item(), total_steps) 

                model.train()                
                save_steps +=1

            
    return loss_hist

def calculate_perplexity(e, num_buttons):
    """Calculate perplexity of quantized encoder outputs
    Perplexity measures how uniformly the encoder uses the available buttons.
    
    Args:
        e (torch.Tensor): Encoder outputs [batch_size, seq_len]
        num_buttons (int): Number of quantization bins (usually 8)
    """
    # Check which values are in valid range
    inrange_mask = torch.logical_and(
        e >= -1,
        e <= 1
    ).float()
    
    # Convert encoder outputs to discrete indices
    e_discrete = (((e + 1) / 2) * (num_buttons - 1)).round().long()
    
    # Clamp the values to be within the valid range
    e_discrete = torch.clamp(e_discrete, min=0, max=num_buttons-1)
    # Convert to one-hot vectors
    e_onehot = F.one_hot(e_discrete, num_buttons).float()
    
    # Calculate masked average probability of each quantization level
    masked_sum = torch.sum(e_onehot * inrange_mask.unsqueeze(-1), dim=[0, 1])
    mask_sum = torch.sum(inrange_mask)
    avg_probs = masked_sum / (mask_sum + 1e-10)
    
    # Calculate perplexity as exp(entropy)
    perplexity = torch.exp(-torch.sum(
        avg_probs * torch.log(avg_probs + 1e-10)
    ))
    
    return perplexity


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