
# Import all needed modules
import secrets

from midiUtils import Tegridy_Any_Pickle_File_Reader
from model import *
from trainUtils import LrStepTracker, train

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from params import *

""" SET HYPERPARAMETERS """

config = GPTConfig(
                   block_size=BLOCK_SIZE, # block_size
                   dim_feedforward=DIM_FEEDFORWARD, # 2048 Size of the feedforward linear layer after attention
                   n_layer=N_LAYERS, 
                   n_head=N_HEADS, 
                   n_embd=N_EMBED, # 1024 Number of embeddings
                   enable_rpr=True,
                   er_len=SEQ_LEN)

""" LOAD TRAINING DATA """

# Loading dataset from a pickle in ./Training-Data
train_data = Tegridy_Any_Pickle_File_Reader('./Training-Data/giant_sel')   
data_train = torch.Tensor(train_data)

class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

        # Separate interleaved data into feature streams
        self.total_tokens = self.data.size(0)
        assert self.total_tokens % 4 == 0, "Data length must be divisible by 4 (dtime, vel, pitch, dur)"
        
        self.num_notes = self.total_tokens // 4
        self.feature_data = {
            'dtime': self.data[0::4],  # Every 4th token starting at index 0
            'vel': self.data[1::4],    # Every 4th token starting at index 1
            'pitch': self.data[2::4],  # Every 4th token starting at index 2
            'dur': self.data[3::4]     # Every 4th token starting at index 3
        }

    def __getitem__(self, index): # TODO empalma els midi files a sac en un stream. Els samples poden ser el final d'un midi + inici d'un altre.
        # Calculate random start position (must be aligned with note boundaries)
        max_start_idx = ((self.num_notes - self.seq_len) // (self.seq_len)) * (self.seq_len)
        rand = secrets.randbelow(max_start_idx)
        #rand = secrets.randbelow((self.data.size(0)-(self.seq_len)) // (self.seq_len)) * (self.seq_len)

        # Extract sequences for each feature
        # x = self.data[rand: rand + self.seq_len].long() # shape = seq_len (2048) seleciona seq_len tokens random del dataset
        x = {
            'dtime': self.feature_data['dtime'][rand:rand + self.seq_len].long(),
            'vel': self.feature_data['vel'][rand:rand + self.seq_len].long(),
            'pitch': self.feature_data['pitch'][rand:rand + self.seq_len].long(),
            'dur': self.feature_data['dur'][rand:rand + self.seq_len].long()
        }

        # Target is the next pitch tokens
        # trg = self.data[(rand+1): (rand+1) + self.seq_len].long() 
        # # target, ground truth to compare with the prediction
        tgt = self.feature_data['pitch'][rand + 1:rand + 1 + self.seq_len].long()
        
        return x, tgt

    def __len__(self):
        return self.num_notes  # Return length in notes (not tokens)

train_dataset = MusicSamplerDataset(data_train, SEQ_LEN) # train in chunks of SEQ_LEN
train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=NUM_WORKERS)

print('=' * 50)
print('DATA LOADED')
print('=' * 50)


""" CREATE MODEL """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerAutoencoder(config)
#model = nn.DataParallel(model) # Multi-GPU training...
model.to(device)

""" SETUP OPTIMIZER """

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(N_EMBED, SCHEDULER_WARMUP_STEPS, init_step)
train_loss_recons = nn.CrossEntropyLoss(ignore_index=VOCAB_SIZE_PITCH)

opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
lr_scheduler = LambdaLR(opt, lr_stepper.step)

""" TRAIN MODEL """

loss_train = []

for epoch in range(0, EPOCHS):
    
    loss = train(epoch+1, 
                 model, train_loader, 
                 train_loss_recons, 
                 opt, 
                 lr_scheduler, 
                 save_checkpoint_steps=4000, # autosave checkpoints frequency, in tokens
                 tensorboard_steps=200,
                 device=device) 
    
    loss_train.append(loss)
        

