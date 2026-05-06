#%%
import numpy as np
import torch
import torch.nn as nn 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import struct

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

def float32_bits(f):
    packed = struct.pack('>f',f)
    bits = ''.join(f'{b:08b}' for b in packed)
    return f"{bits[0]} | {bits[1:9]} | {bits[9:]}"

def float16_bits(f):
    packed = struct.pack('>f',f)
    bits = ''.join(f'{b:08b}' for b in packed)
    return f"{bits[0]} | {bits[1:5]} | {bits[6:]}"

def next_float32(f):
    return float(np.nextafter(np.float32(f),np.float32(np.inf)))

def ulp32(f):
    return next_float32(f)-float(np.float32(f))

X_train,X_test ,Y_train,Y_test = load_data()

print(X_train.shape[0],X_test.shape[0])

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,10)
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
acc = 0
with torch.no_grad():
    acc = (model(X_test).argmax(1)==Y_test).float().mean().item()
print(f"FP32 accuracy :{acc}")


# look at weights
sample_weights = model.fc1.weight.data[0,0:5].numpy()
print(sample_weights)
for w in sample_weights:
    bfp32 = float32_bits(float(w))
    bfp16 = float16_bits(float(w))
    print(f"{float(w):>14.8f} # {bfp32:>35} # {bfp16:>32}")

# convert fp32 -> fp16 weights

all_weights = []
for name,param in model.named_parameters():
    if 'weight' in name:
        all_weights.append(param.data.numpy().flatten())

w_fp32 = np.concatenate(all_weights)
w_fp16 = w_fp32.astype(np.float16).astype(np.float32)
err = np.abs(w_fp32-w_fp16)

print(f"Range | fp32: {np.max(w_fp32):>32} |{np.min(w_fp32)}")
print(f"Range | fp16: {np.max(w_fp16):>32} |{np.min(w_fp16)}")
print(f"max error : {np.max(err)}")
print(f"relative err {(err/np.abs(w_fp32+1e-12)).mean()}")


#check if activations overflow

activations = {}
def make_hook(name):
    def hook(module,inp,out):
        activations[name] = out.detach()
    return hook

def add_hook():
    h1 = model.fc1.register_forward_hook(make_hook('fc1'))
    h2 = model.fc2.register_forward_hook(make_hook('fc2'))
    h3 = model.fc3.register_forward_hook(make_hook('fc3'))
    return h1,h2,h3
h1,h2,h3=add_hook()
with torch.no_grad():
    _ = model(X_test)

def remove_hook(h1,h2,h3):
    h1.remove()
    h2.remove()
    h3.remove()

remove_hook(h1,h2,h3)

def check_overflow():
    for name,act in activations.items():
        a = act.numpy()
        overflow = np.any(abs(a)>65504)#max fp16
        print(name,a.min(),a.max(),a.mean(),"overflow" if overflow else "not overflowing")
check_overflow()
# so current inputs are more or less safe for quantizing to fp16

scale_factor = [10,100,500,10000]

for scale in scale_factor:
    print(f"scale:{scale}")
    X_scale = X_test*scale
    h1,h2,h3= add_hook()
    activations={}
    with torch.no_grad():
        _ = model(X_scale)
    check_overflow()
    remove_hook(h1,h2,h3)
    print("**"*10)

print("\n half precision\n")
# what if i do half precision
for scale in scale_factor:
    print(f"scale:{scale}")
    X_scale = X_test*scale
    X_scale = torch.tensor(X_scale.numpy().astype(np.float16).astype(np.float32))
    h1,h2,h3= add_hook()
    activations={}
    with torch.no_grad():
        _ = model(X_scale)
    check_overflow()
    remove_hook(h1,h2,h3)
    print("**"*10)

# %%
import copy
model.eval()
with torch.no_grad():
    acc_fp32 = (model(X_test).argmax(1) == Y_test).float().mean().item()
    print("fp32:", acc_fp32)

# Deep copy + cast
model_fp16 = copy.deepcopy(model).half().eval()

with torch.no_grad():
    X_test_fp16 = X_test.half()
    acc_fp16 = (model_fp16(X_test_fp16).argmax(1) == Y_test).float().mean().item()
    print("fp16:", acc_fp16)

# So no differnce in accuracy for fp16 & fp32 ( kindof similar to out observation in 1st experiment , q1.py)
#fp32: 0.9222221970558167
#fp16: 0.9222221970558167

#INT8 dynamic quantization (weights int8, activations float)
model_int8 = torch.quantization.quantize_dynamic(
    copy.deepcopy(model),
    {torch.nn.Linear},   # quantize Linear layers
    dtype=torch.qint8
).eval()

with torch.no_grad():
    acc_int8 = (model_int8(X_test).argmax(1) == Y_test).float().mean().item()
    print("int8 (dynamic):", acc_int8)

# %%
def quantize_weight_int4(W):
    qmin, qmax = -8, 7
    scale = W.abs().max() / qmax + 1e-8
    W_q = torch.clamp((W / scale).round(), qmin, qmax)
    return W_q * scale

model_int4 = copy.deepcopy(model).eval()

for m in model_int4.modules():
    if isinstance(m, torch.nn.Linear):
        m.weight.data = quantize_weight_int4(m.weight.data)
with torch.no_grad():
    acc_int4 = (model_int4(X_test).argmax(1) == Y_test).float().mean().item()
    print("4 bit weight quantization : acc ",acc_int4)

#some minor drop with weight quantization , but which cases are failing will be interesting to look at . Also might be a bettter idea to checkout some other dataset
# %%
