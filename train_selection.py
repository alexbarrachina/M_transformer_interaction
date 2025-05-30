import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import time
import tqdm
#from torch.utils.tensorboard import SummaryWriter
from params import *

#!set USE_FLASH_ATTENTION=1
os.environ['USE_FLASH_ATTENTION'] = '1'

from random import randint, random
import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, load_from_disk
from TMIDIX import tegridy_tokens_to_dict, Tegridy_Any_Pickle_File_Reader

from x_transformer_1_23_2 import *

class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len, is_eval=False):
        super().__init__()

        self.data = data
        self.seq_len = seq_len
        self.seq_tot_tokens = self.seq_len * 4 + 4 # 4 tokens per note + 4 for the current note


    def __len__(self):
        return int(self.data.size(0) / (self.seq_len * 4 + 4))  #  self.seq_len if you want exact training time per epoch

    def __getitem__(self, index): # TODO concatenates all data, end of files with begining of files
        seq_tot_tokens = self.seq_tot_tokens
        # We only pick starting positions that are multiples of seq_tot_tokens, // seq_tot_tokens: Integer division to get how many complete sequences of size seq_tot_tokens we can fit
        # We don't exceed the data boundaries
        #rand = secrets.randbelow((self.data.size(0)-seq_tot_tokens) // seq_tot_tokens) * seq_tot_tokens
        rand = randint(0, (self.data.size(0)-seq_tot_tokens) // seq_tot_tokens) * seq_tot_tokens

        # Extract sequences for each feature, +1 to include the current token
        x = self.data[rand: rand + seq_tot_tokens] # we take an extra token

        # convert to tensors, move to device
        dtimes = x[0::4].long()  # Every 4th token starting at index 0. observed min = 0, max = 70
        durs = (x[1::4] - OFFSET_DUR).long()  # Every 4th token starting at index 2. observed min = 1, max = 74
        pitches = (x[2::4] - OFFSET_PITCH).long()  # Every 4th token starting at index 3. observed min = 30, max = 88
        #vels = (x[3::4] - OFFSET_VEL).long()  # Every 4th token starting at index 4. observed min = 30, max = 88

        # Data augmentation
        # Time stretching
        stretch_factor = random() * DATA_AUGMENT_TIME_STRETCH_MAX * 2
        stretch_factor += 1 - DATA_AUGMENT_TIME_STRETCH_MAX
        dtimes = (dtimes.float() * stretch_factor).long()
        dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)
  
        stretch_factor = random() * DATA_AUGMENT_TIME_STRETCH_MAX * 2
        stretch_factor += 1 - DATA_AUGMENT_TIME_STRETCH_MAX
        durs = (durs.float() * stretch_factor).long()
        durs = torch.clamp(durs, min=0, max=RANGE_DUR_SHIFT)

        # Chord micro-alterations
        # Convert to absolute times for easier chord detection
        abs_times = torch.cumsum(dtimes, dim=0)
              
        # Find chord groups
        chord_groups = []
        current_chord = [0]  # Start with first note
        
        for i in range(1, len(abs_times)):
            if abs_times[i] - abs_times[i-1] <= AUGMENT_CHORD_THRESHOLD:
                current_chord.append(i)
            else:
                if len(current_chord) > 1:  # Only process if it's actually a chord
                    chord_groups.append(current_chord)
                current_chord = [i]
        
        if len(current_chord) > 1:
            chord_groups.append(current_chord)
        
        # Apply micro-alterations to chord notes
        for chord in chord_groups:
            # Generate small random shifts for each note in the chord
            shifts = torch.randint(-1, 2, (len(chord),))  # Random shifts of -1, 0, or 1
            abs_times[chord] = abs_times[chord] + shifts
        
        # Re-sort the sequence based on new absolute times
        sorted_indices = torch.argsort(abs_times)
        abs_times = abs_times[sorted_indices]
        durs = durs[sorted_indices]
        pitches = pitches[sorted_indices]
        
        # Convert back to delta times
        dtimes = torch.cat([abs_times[0:1], abs_times[1:] - abs_times[:-1]])
        dtimes = torch.clamp(dtimes, min=0, max=RANGE_DTIME_SHIFT)

        # Transposition
        transposition_factor = randint(
            -DATA_AUGMENT_TRANSPOSE_MAX, DATA_AUGMENT_TRANSPOSE_MAX
        )
        # Apply transposition and ensure pitches stay within valid range (0-127)
        # TODO: Clamp isn't a good idea as we alter the interval relationships. But we hope transposing +-6 we don't clamp
        pitches = torch.clamp(pitches + transposition_factor, min=0, max=VOCAB_SIZE_PITCH-1)

        feature_data = {
                'dtime': dtimes,
                'dur':  durs,
                'pitch': pitches
                #'vel': vels
            }
        return feature_data

def main():
    # Set up CUDA settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    #==========================================================================

    ''' WANDB '''
    if(USE_LOGS):
        #tensorboard_summary = SummaryWriter()
        import wandb
        wandb.login()
        config = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "seq_len": SEQ_LEN,
            "emb_dim": EMB_DIM,
            "num_layers": NUM_LAYERS,
            "loss_margin": LOSS_MARGIN_MULTIPLIER,
            "loss_contour": LOSS_CONTOUR_MULTIPLIER,
            "loss_deviate": LOSS_DEVIATE_MULTIPLIER,
            "data%": DATA_SIZE,
            "model": MODEL_NAME,
            "description": DESCRIPTION
        }
        wandb.init(project="monsterGenie", config=config)

    #==========================================================================

    ''' DEVICE '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type='cuda' if torch.cuda.is_available() else 'cpu'

    #==========================================================================

    ''' DATA '''

    """ LOAD TRAINING DATA """

    # Loading dataset from a pickle in ./Training-Data
    train_data = Tegridy_Any_Pickle_File_Reader('./Training-Data/giantMIDI_sel')   
    data_train = torch.Tensor(train_data)
    eval_data = Tegridy_Any_Pickle_File_Reader('./Training-Data/giantMIDI_test')   
    data_eval = torch.Tensor(eval_data)

    # Dataloader
    train_dataset = MusicSamplerDataset(data_train, SEQ_LEN) # train in chunks of SEQ_LEN
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"Dataset size: {len(train_dataset)}")
    train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    print(f"Number of batches: {len(train_loader)}")
    val_dataset = MusicSamplerDataset(data_eval, SEQ_LEN, is_eval=True) # train in chunks of SEQ_LEN
    val_loader  = DataLoader(val_dataset, batch_size = BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    #==========================================================================

    ''' MODEL '''

    '''
    model = AutoregressiveAutoencoder(
        ignore_index = PAD_IDX, 
        #pad_value=PAD_IDX,
        decoder = Decoder(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            dim = EMB_DIM,
            depth = NUM_LAYERS,
            heads = NUM_HEADS,
            rotary_pos_emb = True,
            attn_flash = True
            ),
        encoder = Encoder(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            dim = EMB_DIM,
            depth = NUM_LAYERS,
            heads = NUM_HEADS,
            rotary_pos_emb = True,
            attn_flash = True
            )
        )'''


    model = AutoregressiveAutoencoder_no_dtime(
        ignore_index = PAD_IDX, 
        #pad_value=PAD_IDX,
        decoder = Decoder_no_dtime(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            dim = EMB_DIM,
            depth = NUM_LAYERS,
            heads = NUM_HEADS,
            rotary_pos_emb = True,
            attn_flash = True
            ),
        encoder = Encoder_no_dtime(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            dim = EMB_DIM,
            depth = NUM_LAYERS,
            heads = NUM_HEADS,
            rotary_pos_emb = True,
            attn_flash = True
            )
        )
    '''
    model = EncoderOnly(
        ignore_index = PAD_IDX, 
        #pad_value=PAD_IDX,
        encoder = Encoder(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            dim = EMB_DIM,
            depth = NUM_LAYERS,
            heads = NUM_HEADS,
            rotary_pos_emb = True,
            attn_flash = True
            )
        )

    model = DecoderOnly(
        ignore_index = PAD_IDX, 
        #pad_value=PAD_IDX,
        decoder = DecoderSimple(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            dim = EMB_DIM,
            depth = NUM_LAYERS,
            heads = NUM_HEADS,
            rotary_pos_emb = True,
            attn_flash = True
            )
        )'''

    model.to(device)

    #print(model)

    ''' PRECISION/OPTIMIZER/SCALER '''

    dtype = torch.bfloat16

    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler(device_type)

    ''' TRAINING '''

    nsteps = 0

    for ep in range(NUM_EPOCHS):
        print('Epoch #', ep)
        
        model.train()
        with tqdm.tqdm(total=len(train_loader)) as bar_train:
            for i, batch in enumerate(train_loader):            
                optim.zero_grad()

                # move to device
                x = {
                    'dtime': batch['dtime'].to(device),
                    'dur': batch['dur'].to(device),
                    'pitch': batch['pitch'].to(device)
                }

                with ctx:
                    loss, acc = model(x)  # Update your model to accept target separately
                scaler.scale(loss['loss_total']).backward()
                
                if (i % PRINT_STATS_EVERY == 0) or TESTING:
                    if(USE_LOGS):                
                        wandb.log({"train_loss": loss['loss_total'].item()}, step=nsteps)
                        wandb.log({"train_acc": acc.item()}, step=nsteps)
                        ''' if LOSS_NORM_POS_MULTIPLIER>0:
                            wandb.log({"train_loss_norm_pos": LOSS_NORM_POS_MULTIPLIER*loss['loss_norm_pos'].item()}, step=nsteps)
                        if LOSS_DEVIATE_MULTIPLIER>0:
                            wandb.log({"train_loss_deviate": LOSS_DEVIATE_MULTIPLIER*loss['loss_deviate'].item()}, step=nsteps)
                        if LOSS_CONTOUR_MULTIPLIER>0:
                            wandb.log({"train_loss_contour": LOSS_CONTOUR_MULTIPLIER*LOSS_CONTOUR_MULTIPLIER*loss['loss_contour'].item()}, step=nsteps)
                        if LOSS_MULTI_STEP_PERC>0:
                            wandb.log({"train_loss_multi_step": LOSS_CONTOUR_MULTIPLIER*LOSS_MULTI_STEP_PERC*loss['loss_multi_step'].item()}, step=nsteps)
                        if LOSS_INTERVAL_PERC>0:
                            wandb.log({"train_loss_interval": LOSS_CONTOUR_MULTIPLIER*LOSS_INTERVAL_PERC*loss['loss_interval'].item()}, step=nsteps)
                        if LOSS_SHAPE_PERC>0:
                            wandb.log({"train_loss_shape": LOSS_CONTOUR_MULTIPLIER*LOSS_SHAPE_PERC*loss['loss_shape'].item()}, step=nsteps)
                        if LOSS_BUTTON_HELD_MULTIPLIER>0: 
                            wandb.log({"train_loss_button_held": LOSS_BUTTON_HELD_MULTIPLIER*loss['loss_button_held'].item()}, step=nsteps)
                        if LOSS_NORM_POS_MULTIPLIER>0:
                            wandb.log({"train_loss_norm_pos": LOSS_NORM_POS_MULTIPLIER*loss['loss_norm_pos'].item()}, step=nsteps)
                        '''
                        nsteps += 1


                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optim)
                scaler.update()


                bar_train.set_description(f'Epoch: {ep} Loss: {float(loss["loss_total"]):.4}')# LR: {float(lr):.8}')
                bar_train.update(1)

                if (i % VALIDATE_EVERY == 0) or TESTING:
                    try:
                        x = next(iter(val_loader)) # extract batches from test dataloader
                    except StopIteration:
                        val_loader_iter = iter(val_loader)
                        batch = next(iter(val_loader_iter)) # extract batches from test dataloader           
                    model.eval()
                    with torch.no_grad():
                        with ctx:
                            # move to device
                            x = {
                                'dtime': batch['dtime'].to(device),
                                'dur': batch['dur'].to(device),
                                'pitch': batch['pitch'].to(device)
                            }
                            # run the model
                            val_loss, val_acc = model(x)  # Update your model to accept target separately

                        if(USE_LOGS):                
                            wandb.log({"val_loss": val_loss['loss_total'].item()}, step=nsteps)
                            wandb.log({"val_acc": val_acc.item()}, step=nsteps)
                            '''if LOSS_NORM_POS_MULTIPLIER>0:
                                wandb.log({"val_loss_norm_pos": LOSS_NORM_POS_MULTIPLIER*val_loss['loss_norm_pos'].item()}, step=nsteps)
                            if LOSS_DEVIATE_MULTIPLIER>0:
                                wandb.log({"val_loss_deviate": LOSS_DEVIATE_MULTIPLIER*val_loss['loss_deviate'].item()}, step=nsteps)
                            if LOSS_CONTOUR_MULTIPLIER>0:
                                wandb.log({"val_loss_contour": LOSS_CONTOUR_MULTIPLIER*LOSS_MULTI_STEP_PERC*val_loss['loss_contour'].item()}, step=nsteps)
                            if LOSS_MULTI_STEP_PERC>0:
                                wandb.log({"val_loss_multi_step": LOSS_CONTOUR_MULTIPLIER*LOSS_MULTI_STEP_PERC*val_loss['loss_multi_step'].item()}, step=nsteps)
                            if LOSS_INTERVAL_PERC>0:
                                wandb.log({"val_loss_interval": LOSS_CONTOUR_MULTIPLIER*LOSS_INTERVAL_PERC*val_loss['loss_interval'].item()}, step=nsteps)
                            if LOSS_SHAPE_PERC>0:
                                wandb.log({"val_loss_shape": LOSS_CONTOUR_MULTIPLIER*LOSS_SHAPE_PERC*val_loss['loss_shape'].item()}, step=nsteps)
                            if LOSS_BUTTON_HELD_MULTIPLIER>0: 
                                wandb.log({"val_loss_button_held": LOSS_BUTTON_HELD_MULTIPLIER*val_loss['loss_button_held'].item()}, step=nsteps)
                            if LOSS_NORM_POS_MULTIPLIER>0:
                                wandb.log({"val_loss_norm_pos": LOSS_NORM_POS_MULTIPLIER*val_loss['loss_norm_pos'].item()}, step=nsteps)
                            '''

                    model.train()

        
        if ep % SAVE_EVERY == 0:
            fname = './save_models/' + MODEL_NAME + '_' + str(ep) + '_eps_' + str(nsteps) + '_steps_' + str(round(float(loss['loss_total'].item()), 4)) + '_loss_' + str(round(float(acc.item()), 4)) + '_acc.pth'
            torch.save(model.state_dict(), fname)


if __name__ == '__main__':
    mp.freeze_support()
    main()

