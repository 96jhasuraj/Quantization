'''
Idea is to train a simple LR model on float data & evaluate the effect of passing quantized data

Learnings : Since the decision boundary has healthy margins as compared to the error pertubation due to quantization ; int8 data is an acceptable choise with this fp32 model with this data
'''


#%%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#%%
data = load_digits()
X = data.data.astype(np.float32)
Y = data.target
print(X.shape,Y.shape)
# %%
unique_class = np.unique(Y)
X_min = np.min(X)
X_max = np.max(X)
print(X_min,X_max,unique_class)
# %%

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
print(np.min(X_scaled),np.max(X_scaled))

# %%
for n_bits in [16,8,4,2,1]:
    levels = 2**n_bits
    step = 1/(levels-1)
    max_err = step/2
    print(f"bits:{n_bits} step : {step} max_err : {max_err}")
# %%

def quantize(f,bits):
    levels = 2**bits
    step = 1/(levels-1)
    #print(bits,levels,step)
    index = np.clip(np.round(f/step),0,levels-1).astype(np.int32)
    return index*step

#%%
sample = 100
for n_bits in [16,8,4,2,1]:
    xq = quantize(X_scaled[sample],n_bits)
    error = X_scaled[sample]-xq
    levels = 2**n_bits
    step = 1/(levels-1)
    max_err = np.max(np.abs(error))
    snr = 10 * np.log10(np.mean(X_scaled[sample]**2) / (np.mean(error**2) + 1e-14))
    print(f"bits:{n_bits} step : {step} max_err : {max_err} snr:{snr}")
    
# %%

import matplotlib.pyplot as plt

def print_digit(flat, label):
    grid = flat.reshape(8, 8)
    
    plt.imshow(grid, cmap='gray')
    plt.title(f"True label: {label}")
    plt.axis('off')
    plt.show()
# %%

import matplotlib.pyplot as plt

sample = 101
def compare_bits(x, label):
    fig, axs = plt.subplots(1, 6, figsize=(12, 3))
    
    axs[0].imshow(x.reshape(8,8), cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    for i, n_bits in enumerate([16, 8, 4, 2, 1]):
        xq = quantize(x, n_bits)
        axs[i+1].imshow(xq.reshape(8,8), cmap='gray')
        axs[i+1].set_title(f"{n_bits} bits")
        axs[i+1].axis('off')
    
    plt.suptitle(f"Label: {label}")
    plt.show()
    
compare_bits(X_scaled[sample],Y[sample])
# %%

X_train,X_test ,Y_train,Y_test = train_test_split(X_scaled,Y,test_size=0.2,stratify=Y)

#1 full precision model
clf = LogisticRegression()
clf.fit(X_train,Y_train)
fp_acc = accuracy_score(Y_test,clf.predict(X_test))
print(f"baseline fp32 model accuracy {fp_acc}")
# %%
for n_bits in [16,8,4,2,1]:
    X_test_q = quantize(X_test,n_bits)
    acc = accuracy_score(Y_test,clf.predict(X_test_q))
    drop = fp_acc - acc
    print(f"nbits {n_bits} {acc} | drop in acc {drop}")

# %%

def logit_perturbation(clf, X, bits):
    Xq = quantize(X, bits)

    logits_fp = clf.decision_function(X)
    logits_q  = clf.decision_function(Xq)

    delta = logits_q - logits_fp
    return delta

# %%
def margin(clf, X):
    logits = clf.decision_function(X)
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    return top2[:,1] - top2[:,0]
# %%
for n_bits in [16,8,4,2,1]:
    delta = logit_perturbation(clf, X_test, n_bits)

    perturb = np.max(np.abs(delta), axis=1)
    margins = margin(clf, X_test)

    print(f"\n{n_bits} bits:")
    print("mean |Δlogit|:", np.mean(perturb))
    print("max  |Δlogit|:", np.max(perturb))
    print("mean margin  :", np.mean(margins))

    safe = np.mean(perturb < margins)
    print("fraction safe (no flip):", safe)
# %%
