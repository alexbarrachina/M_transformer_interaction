import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import tqdm
#from torch.utils.tensorboard import SummaryWriter
from params import *

#!set USE_FLASH_ATTENTION=1
os.environ['USE_FLASH_ATTENTION'] = '1'

import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset, load_from_disk
from TMIDIX import tegridy_tokens_to_dict, Tegridy_Any_Pickle_File_Writer

from x_transformer_1_23_2 import *

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
class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len, is_eval=False):
        super().__init__()

        self.feature_data = []  # Changed from set to list for indexing
        self.seq_len = seq_len
        self.num_notes = 0
        self.is_eval = is_eval
        self.indices = []

        self.feature_data, self.num_notes = tegridy_tokens_to_dict(data)

        if self.is_eval:
            max_indices = self.num_notes - (self.seq_len+1)
            self.indices = list(range(0, max_indices, self.seq_len+1))

    def __len__(self):
        return self.num_notes // (self.seq_len+1)  # Return number of possible sequences

    def __getitem__(self, index):
        # Calculate start position for this index
        if self.is_eval:
            # For evaluation, use sequential samples
            rand = self.indices[index % len(self.indices)]
        else:
            # For training, use random sampling
            max_start_idx = self.num_notes - (self.seq_len+1)
            rand = torch.randint(0, max_start_idx, (1,)).item()

        # Extract sequences for each feature, +1 to include the current token
        # convert to tensors, move to device
        if TESTING:
            x = {
            'dtime': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.long).to(device),
            'dur': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.long).to(device),
            'pitch': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=torch.long).to(device)
            }
        else:
            dtimes = torch.tensor(self.feature_data['dtime'][rand:rand + self.seq_len+1], dtype=torch.long)
            # converting to continuous values
            #dtimes = torch.div(dtimes, OFFSET_DUR)
            durs = torch.tensor(self.feature_data['dur'][rand:rand + self.seq_len+1], dtype=torch.long)
            # converting to continuous values
            #durs = torch.div(durs, OFFSET_DUR)
            pitches = torch.tensor(self.feature_data['pitch'][rand:rand + self.seq_len+1], dtype=torch.long)
            x = {
            'dtime': dtimes.to(device),
            'dur':  durs.to(device),
            'pitch': pitches.to(device)
            }
        return x

# Load the dataset from disk
monster_piano = load_from_disk(local_dataset_path)

# If you need specific splits, you can select them after loading
train_dataset = monster_piano['train']
if TESTING or LIGHT_DATASET:
    monster_piano_train = train_dataset.select(range(int(len(train_dataset) * 0.01)))  # 1% for training
    monster_piano_val = train_dataset.select(range(int(len(train_dataset) * 0.99), len(train_dataset)))  # Last 1% for validation
else:
    monster_piano_train = train_dataset.select(range(int(len(train_dataset) * 0.2)))  # 1% for training
    monster_piano_val = train_dataset.select(range(int(len(train_dataset) * 0.95), len(train_dataset)))  # Last 1% for validation


# Dataloader
train_data = MusicSamplerDataset(monster_piano_train, SEQ_LEN)
val_data = MusicSamplerDataset(monster_piano_val, SEQ_LEN, is_eval=True) 

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

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
'''
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

# Train the model

#train_losses = []
#val_losses = []

#train_accs = []
#val_accs = []

nsteps = 0

for ep in range(NUM_EPOCHS):
    print('Epoch #', ep)
    
    model.train()
    for i, x in enumerate(tqdm.tqdm(train_loader, desc='Training')):
        
        optim.zero_grad()

        with ctx:
            loss, loss_margin, loss_deviate, loss_contour, loss_multi_step, loss_interval, loss_shape, loss_button_held, acc = model(x)  # Update your model to accept target separately
        scaler.scale(loss).backward()
        
        if i % PRINT_STATS_EVERY == 0:
            if(USE_LOGS):                
                #tensorboard_summary.add_scalar("train_loss", loss.item(), nsteps)
                wandb.log({"train_loss": loss.item()}, step=nsteps)
                wandb.log({"train_acc": acc.item()}, step=nsteps)
                if LOSS_MARGIN_MULTIPLIER>0:
                    wandb.log({"train_loss_margin": loss_margin.item()}, step=nsteps)
                if LOSS_DEVIATE_MULTIPLIER>0:
                    wandb.log({"train_loss_deviate": LOSS_DEVIATE_MULTIPLIER*loss_deviate.item()}, step=nsteps)
                if LOSS_CONTOUR_MULTIPLIER>0:
                    wandb.log({"train_loss_contour": LOSS_CONTOUR_MULTIPLIER*LOSS_CONTOUR_MULTIPLIER*loss_contour.item()}, step=nsteps)
                if LOSS_MULTI_STEP_PERC>0:
                    wandb.log({"train_loss_multi_step": LOSS_CONTOUR_MULTIPLIER*LOSS_MULTI_STEP_PERC*loss_multi_step.item()}, step=nsteps)
                if LOSS_INTERVAL_PERC>0:
                    wandb.log({"train_loss_interval": LOSS_CONTOUR_MULTIPLIER*LOSS_INTERVAL_PERC*loss_interval.item()}, step=nsteps)
                if LOSS_SHAPE_PERC>0:
                    wandb.log({"train_loss_shape": LOSS_CONTOUR_MULTIPLIER*LOSS_SHAPE_PERC*loss_shape.item()}, step=nsteps)
                if LOSS_BUTTON_HELD_MULTIPLIER>0: 
                    wandb.log({"train_loss_button_held": LOSS_BUTTON_HELD_MULTIPLIER*loss_button_held.item()}, step=nsteps)
                nsteps += 1
           #train_losses.append(loss.item())
            #train_accs.append(acc.item())

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()

 
        if (i % VALIDATE_EVERY == 0) or TESTING:
            try:
                x = next(iter(val_loader)) # extract batches from test dataloader
            except StopIteration:
                val_loader_iter = iter(val_loader)
                x = next(iter(val_loader_iter)) # extract batches from test dataloader           
            model.eval()
            with torch.no_grad():
                with ctx:
                    # run the model
                    #val_loss, val_acc = model(x)
                    val_loss, loss_margin, loss_deviate, loss_contour, loss_multi_step, loss_interval, loss_shape, loss_button_held, acc = model(x)  # Update your model to accept target separately

                if(USE_LOGS):                
                    #tensorboard_summary.add_scalar("val_loss", val_loss.item(), nsteps)
                    #tensorboard_summary.add_scalar("val_acc", val_acc.item(), nsteps)
                    wandb.log({"val_loss": val_loss.item()}, step=nsteps)
                    wandb.log({"val_acc": acc.item()}, step=nsteps)
                    if LOSS_MARGIN_MULTIPLIER>0:
                        wandb.log({"val_loss_margin": LOSS_MARGIN_MULTIPLIER*loss_margin.item()}, step=nsteps)
                    if LOSS_DEVIATE_MULTIPLIER>0:
                        wandb.log({"val_loss_deviate": LOSS_DEVIATE_MULTIPLIER*loss_deviate.item()}, step=nsteps)
                    if LOSS_CONTOUR_MULTIPLIER>0:
                        wandb.log({"val_loss_contour": LOSS_CONTOUR_MULTIPLIER*LOSS_MULTI_STEP_PERC*loss_contour.item()}, step=nsteps)
                    if LOSS_MULTI_STEP_PERC>0:
                        wandb.log({"val_loss_multi_step": LOSS_CONTOUR_MULTIPLIER*LOSS_MULTI_STEP_PERC*loss_multi_step.item()}, step=nsteps)
                    if LOSS_INTERVAL_PERC>0:
                        wandb.log({"val_loss_interval": LOSS_CONTOUR_MULTIPLIER*LOSS_INTERVAL_PERC*loss_interval.item()}, step=nsteps)
                    if LOSS_SHAPE_PERC>0:
                        wandb.log({"val_loss_shape": LOSS_CONTOUR_MULTIPLIER*LOSS_SHAPE_PERC*loss_shape.item()}, step=nsteps)
                    if LOSS_BUTTON_HELD_MULTIPLIER>0: 
                        wandb.log({"val_loss_button_held": LOSS_BUTTON_HELD_MULTIPLIER*loss_button_held.item()}, step=nsteps)

                    #val_losses.append(val_loss.item())
                    #val_accs.append(val_acc.item())


            model.train()

 
        if i % SAVE_EVERY == 0:
            #print('Saving model progress. Please wait...')
            #print('model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(loss.item()), 4)) + '_loss_' + str(round(float(acc.item()), 4)) + '_acc.pth')
            fname = './save_models/' + MODEL_NAME + '_' + str(ep) + '_eps_' + str(nsteps) + '_steps_' + str(round(float(loss.item()), 4)) + '_loss_' + str(round(float(acc.item()), 4)) + '_acc.pth'
            torch.save(model.state_dict(), fname)

            #data = [train_losses, train_accs, val_losses, val_accs]
            #Tegridy_Any_Pickle_File_Writer(data, './save_models/losses_accs')

            #print('Done!')

