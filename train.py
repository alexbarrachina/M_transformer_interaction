
# Import all needed modules
import secrets
from collections import OrderedDict

from tqdm import tqdm

from midiUtils import Tegridy_Any_Pickle_File_Reader
from model import *
from trainUtils import LrStepTracker, train

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from params import *

""" SET HYPERPARAMETERS """


SEQ_LEN = 2048 # block_size
DIC_SIZE = 512 # vocab_size
BATCH_SIZE = 2 # Change this to your specs (4 batches per 48GB)
DIM_FEEDFORWARD = 2048 # Size of the feedforward linear layer after attention
N_LAYERS = 24 # Number of layers
N_HEADS = 8 # Number of attention heads
N_EMBED = 1024 # Number of embeddings
EPOCHS = 5 # Number of epochs
NUM_WORKERS = 27 # Number of workers    

config = GPTConfig(DIC_SIZE, # vocab_size
                   SEQ_LEN, # block_size
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

    def __getitem__(self, index): # TODO empalma els midi files a sac en un stream. Els samples poden ser el final d'un midi + inici d'un altre.

        # self.data.size(0) = total_dataset
        rand = secrets.randbelow((self.data.size(0)-(self.seq_len)) // (self.seq_len)) * (self.seq_len)

        x = self.data[rand: rand + self.seq_len].long() # shape = seq_len (2048) seleciona seq_len tokens random del dataset
        trg = self.data[(rand+1): (rand+1) + self.seq_len].long() # target, ground truth per comparar amb la predicció
        
        return x, trg

    def __len__(self):
        return self.data.size(0) #  self.seq_len if you want exact training time per epoch

train_dataset = MusicSamplerDataset(data_train, SEQ_LEN) # train in chunks of SEQ_LEN
train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=NUM_WORKERS)

print('=' * 50)
print('DATA LOADED')
print('=' * 50)


""" CREATE MODEL """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT(config)
#model = nn.DataParallel(model) # Multi-GPU training...
model.to(device)

""" SETUP OPTIMIZER """

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(N_EMBED, SCHEDULER_WARMUP_STEPS, init_step)
eval_loss_func = nn.CrossEntropyLoss(ignore_index=DIC_SIZE)
train_loss_func = eval_loss_func

opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
lr_scheduler = LambdaLR(opt, lr_stepper.step)

""" TRAIN MODEL """

loss_train = []

for epoch in range(0, EPOCHS):
    
    loss = train(epoch+1, 
                 model, train_loader, 
                 train_loss_func, 
                 opt, 
                 lr_scheduler, 
                 save_checkpoint_steps=4000, # autosave checkpoints frequency, in tokens
                 tensorboard_steps=200,
                 device=device) 
    
    loss_train.append(loss)
        

