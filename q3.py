# experimenting with rounding errors

#%%
import numpy as np
import torch
import copy
import torch.nn as nn 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import struct

def rne(x):
    return np.round(x)

def rtz(x):
    return np.fix(x)

def rtn(x):
    return np.floor(x)

def rtp(x):
    return np.ceil(x)

def round_probabilistic(x):
    floor_x = rtn(x)
    prob_up = x - floor_x
    rand = np.random.uniform(0,1,x.shape)
    return np.where(rand<prob_up,floor_x+1,floor_x)

rounding_fxns = {"rne":rne,"rtz":rtz,"rtn":rtn,"rtp":rtp,"round_probabilistic":round_probabilistic}

def quantize_weights(W,n_bits,rounding_fcn):
    qmax = 2**(n_bits-1)-1
    W_dq = np.zeros_like(W)
    scales = np.zeros_like(W.shape[0])
    
    for c in range(W.shape[0]):
        abs_max = float(np.max(np.abs(W[c])))
        abs_max = max(1e-8,abs_max)
        scale = abs_max/qmax
        W_scaled = W[c]/(scale+1e-8)
        W_q = np.clip(rounding_fcn(W_scaled),-qmax,qmax).astype(np.int32)
        W_dq[c] = W_q.astype(np.float32)*scale
    return W_dq,scales

def load_data():
    data = load_digits()
    X = data.data.astype(np.float32)
    Y = data.target
    print(X.shape,Y.shape)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    print(f"scaled data range {np.min(X_scaled)} , {np.max(X_scaled)}")
    X_train,X_test ,Y_train,Y_test = train_test_split(X_scaled,Y,test_size=0.2,stratify=Y)
    return torch.tensor(X_train),torch.tensor(X_test) ,torch.tensor(Y_train),torch.tensor(Y_test)

X_train,X_test ,Y_train,Y_test = load_data()
print(X_train.shape[0],X_test.shape[0])

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self,X):
        X = nn.functional.relu(self.fc1(X))
        X = nn.functional.relu(self.fc2(X))
        return self.fc3(X)

#torch.manual_seed(42)
model = MLP()
opt = torch.optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(100):
    model.train()
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(X_train),Y_train)
    loss.backward()
    opt.step()

model.eval()
fp32_acc = 0
with torch.no_grad():
    fp32_acc = (model(X_test).argmax(1)==Y_test).float().mean().item()
print(f"FP32 accuracy :{fp32_acc}")

# %%
for name,fxn in rounding_fxns.items():
    W_fc1 = model.fc1.weight.data.numpy()
    W_dq , _ = quantize_weights(W_fc1,8,fxn)
    err = W_fc1-W_dq
    print(name,"max:",np.max(np.abs(err))," mean: ",np.mean(err)," std:",np.std(err))
'''
rne max: 0.0009674579  mean:  -1.1077443e-06  std: 0.000471353
rtz max: 0.0018981546  mean:  9.5802585e-05  std: 0.00092505204
rtn max: 0.0018981546  mean:  0.00079508673  std: 0.00048683432
rtp max: 0.001928851  mean:  -0.0007953417  std: 0.00048679786
round_probabilistic max: 0.0017788894  mean:  7.760186e-06  std: 0.00066739385
'''


# %%

# now I will apply to all

def apply_rounding_to_model(model_base,bits,round_fxn):
    m = copy.deepcopy(model_base)
    with torch.no_grad():
        for module in m.modules():
            if isinstance(module,nn.Linear):
                W = module.weight.data.numpy()
                W_dq , _ = quantize_weights(W,bits,round_fxn)
                module.weight.data = torch.from_numpy(W_dq).to(module.weight.device)
    return m

model.eval()
with torch.no_grad():
    fp32_logits = model(X_test)

for name,fxn in rounding_fxns.items():
    quant_model = apply_rounding_to_model(model,8,fxn)
    quant_model.eval()
    with torch.no_grad():
        logits_q = quant_model(X_test)
    bias = (fp32_logits- logits_q).mean().item()
    rmse =  (fp32_logits- logits_q).pow(2).mean().sqrt().item()
    q_acc = (logits_q.argmax(1)==Y_test).float().mean().item()
    drop = fp32_acc - q_acc
    
    print(f"{name:>20} {bias:>12.8f} {rmse:>12.8f} {q_acc:12.8f} {drop:>12.8f}")
'''
                 rne  -0.00531928   0.00801211   0.94444442   0.00000000
                 rtz   0.00727224   0.06413691   0.94166666   0.00277776
                 rtn   0.09577906   0.11346350   0.93888891   0.00555551
                 rtp  -0.10039876   0.11732485   0.94444442   0.00000000
 round_probabilistic  -0.00397753   0.01471397   0.94444442   0.00000000

Although for this dataset we dont see much drop with rne , probabilistic , however this can lead to tricky accuracy bugs in my experience from work
'''

# %%

# At what bit width does rounding stop mattering ?
for bit in [16,8,7,6,5,4,3,2]:
    print("*"*20)
    print(f"bit {bit}\n")
    for name,fxn in rounding_fxns.items():
        quant_model = apply_rounding_to_model(model,bit,fxn)
        quant_model.eval()
        with torch.no_grad():
            logits_q = quant_model(X_test)
        acc = (logits_q.argmax(1)==Y_test).float().mean().item()
        drop = fp32_acc - acc
        print(f"{name:>20} {acc:12.8f} {drop:>12.8f}")
'''
As theoritically expected , the issue with rounding becomes prominant with lower bits . For this dataset & model , 4 bit & below  choise of rounding fxn actually matters
********************
bit 16

                 rne   0.94166666   0.00000000
                 rtz   0.94166666   0.00000000
                 rtn   0.94166666   0.00000000
                 rtp   0.94166666   0.00000000
 round_probabilistic   0.94166666   0.00000000
********************
bit 8

                 rne   0.94166666   0.00000000
                 rtz   0.94166666   0.00000000
                 rtn   0.94444442  -0.00277776
                 rtp   0.94166666   0.00000000
 round_probabilistic   0.94166666   0.00000000
********************
bit 7

                 rne   0.94444442  -0.00277776
                 rtz   0.94166666   0.00000000
                 rtn   0.93888891   0.00277776
                 rtp   0.93888891   0.00277776
 round_probabilistic   0.94166666   0.00000000
********************
bit 6

                 rne   0.94444442  -0.00277776
                 rtz   0.94444442  -0.00277776
                 rtn   0.94166666   0.00000000
                 rtp   0.93888891   0.00277776
 round_probabilistic   0.94166666   0.00000000
********************
bit 5

                 rne   0.94166666   0.00000000
                 rtz   0.94444442  -0.00277776
                 rtn   0.93333334   0.00833333
                 rtp   0.93333334   0.00833333
 round_probabilistic   0.94166666   0.00000000
********************
bit 4

                 rne   0.94166666   0.00000000
                 rtz   0.94444442  -0.00277776
                 rtn   0.91944444   0.02222222
                 rtp   0.91388887   0.02777779
 round_probabilistic   0.94444442  -0.00277776
********************
bit 3

                 rne   0.93888891   0.00277776
                 rtz   0.89166665   0.05000001
                 rtn   0.90833336   0.03333330
                 rtp   0.67777777   0.26388890
 round_probabilistic   0.93611109   0.00555557
********************
bit 2

                 rne   0.88055557   0.06111109
                 rtz   0.10000000   0.84166666
                 rtn   0.10000000   0.84166666
                 rtp   0.09722222   0.84444444
 round_probabilistic   0.80833334   0.13333333

'''

# %%
