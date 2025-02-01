
import secrets

from midiUtils import Tegridy_Any_Pickle_File_Reader
from model import *
from trainUtils import LrStepTracker, train_and_eval

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from params import *

''' DEVICE and PRECISION '''
if torch.cuda.is_available():
    CUDA_AVAILABLE = True
else:
    CUDA_AVAILABLE = False

full_path_to_model_checkpoint = "./SaveModel/gpt2_rpr_checkpoint_1_epoch_1608000_steps_4.3537_loss.pth" 
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
deviceType = 'cuda' if CUDA_AVAILABLE else 'mps'

model_precision  = "bfloat16" # @param ["bfloat16", "float16", "float32"]
#@markdown bfloat16 == Third precision/triple speed (if supported, otherwise the model will default to float16)
#@markdown float16 == Half precision/double speed
#@markdown float32 == Full precision/normal speed

if model_precision == 'bfloat16' and CUDA_AVAILABLE and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
if(CUDA_AVAILABLE):
    ctx = torch.amp.autocast(device_type=deviceType, dtype=ptdtype)
else:
    ctx = None


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
train_data = Tegridy_Any_Pickle_File_Reader('./Training-Data/giant_sel_no_clair')   
data_train = torch.Tensor(train_data)
eval_data = Tegridy_Any_Pickle_File_Reader('./Training-Data/giant_sel_val')   
data_eval = torch.Tensor(eval_data)

class MusicSamplerDataset(Dataset):
    def __init__(self, data, seq_len, is_eval=False):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.is_eval = is_eval
        # For evaluation, pre-compute fixed indices
        if is_eval:
            self.indices = [(i * seq_len) for i in range((self.data.size(0)-self.seq_len) // self.seq_len)]


    def __getitem__(self, index): # TODO empalma els midi files a sac en un stream. Els samples poden ser el final d'un midi + inici d'un altre.
        if self.is_eval:
            # For evaluation, use sequential, fixed samples
            rand = self.indices[index % len(self.indices)]
        else:
            # For training, keep random sampling
            # change to torch.randint for faster training
            #rand = secrets.randbelow((self.data.size(0) - self.seq_len) // self.seq_len) * self.seq_len
            rand = torch.randint(0, (self.data.size(0) - self.seq_len) // self.seq_len, (1,)).item() * self.seq_len

        x = self.data[rand: rand + self.seq_len].long() # shape = seq_len (2048) seleciona seq_len tokens random del dataset
        trg = self.data[(rand+1): (rand+1) + self.seq_len].long() # target, ground truth per comparar amb la predicció
        
        return x, trg

    def __len__(self):
        if self.is_eval:
            # For evaluation, return actual number of complete sequences
            return len(self.indices)

        return self.data.size(0) #  self.seq_len if you want exact training time per epoch

train_dataset = MusicSamplerDataset(data_train, SEQ_LEN, is_eval=False) # train in chunks of SEQ_LEN
train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE,  shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
eval_dataset = MusicSamplerDataset(data_eval, SEQ_LEN, is_eval=True) # train in chunks of SEQ_LEN
eval_loader  = DataLoader(eval_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print('=' * 50)
print('DATA LOADED')
print('=' * 50)


""" CREATE MODEL """

model = GPT(config)
#model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=device))
#model = nn.DataParallel(model) # Multi-GPU training...
model.to(device)

""" SETUP OPTIMIZER """

init_step = 0
lr = LR_DEFAULT_START
lr_stepper = LrStepTracker(N_EMBED, SCHEDULER_WARMUP_STEPS, init_step)
eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, label_smoothing=0.1) # ignore padding token, originally no label smoothing
train_loss_func = eval_loss_func

opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON, weight_decay=ADAM_WEIGHT_DECAY) # originally no weight decay
lr_scheduler = LambdaLR(opt, lr_stepper.step)

""" TRAIN MODEL """

loss_train = []

for epoch in range(0, EPOCHS):
    
    loss = train_and_eval(epoch+1, 
                 model, train_loader, eval_loader, 
                 train_loss_func, eval_loss_func, 
                 opt, 
                 lr_scheduler, 
                 save_checkpoint_steps=SAVE_FREQ, # autosave checkpoints frequency, in tokens
                 tensorboard_steps=LOG_FREQ,
                 device=device,
                 ctx = ctx) 
    
    #loss_train.append(loss)
        

