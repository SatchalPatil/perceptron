import numpy as np

inp = np.array([1,2,3,4,5])
out = np.array([43,48,53,58,63])

w = 0.1
b=0
lrn_rt = 0.01
iters = 5000

def predict(i):
    return w*i+b

for n in range(iters):
    pred = [predict(i)for i in inp]

    cost = [o-p for o,p in zip(out,pred)] # to calculate the error

    for c in cost:
        if c > 0:
            sign_cost = 1 
        else : 
            sign_cost = -1 
    avg_cost = np.mean(np.abs(cost))
    print(f"iterations: {n}, Bias: {b} Weight: {w:.2f} Cost:{avg_cost:.2f}")

    w += lrn_rt * np.dot(cost,inp)/len(inp)
    b += lrn_rt * np.sum(cost)/len(inp)

test_inp = [6,7]
test_out = [68,73]

pred = [predict(i) for i in test_inp]

for i, o, p in zip(test_inp, test_out, pred):
    print(f"input:{i}, target:{o}, pred: {p}")
