import numpy as np


##Q3


##first define the probability distributions as defined in the excercise:

# define 0 as false, 1 as true
def p_C(c):
    p = np.array([0.5, 0.5])

    return p[c]


def p_S_given_C(s, c):
    p = np.array([[0.5, 0.9], [0.5, 0.1]])
    return p[s, c]


def p_R_given_C(r, c):
    p = np.array([[0.8, 0.2], [0.2, 0.8]])
    return p[r, c]


def p_W_given_S_R(w, s, r):
    p = np.array([
        [[1.0, 0.1], [0.1, 0.01]],  # w = False
        [[0.0, 0.9], [0.9, 0.99]],  # w = True
    ])
    return p[w, s, r]


##1. enumeration and conditioning:


## condition and marginalize:
## compute joint:
p = np.zeros((2, 2, 2, 2))  # c,s,r,w
for c in range(2):
    for s in range(2):
        for r in range(2):
            for w in range(2):
                p[c, s, r, w] = p_C(c) * p_S_given_C(s, c) * p_R_given_C(r, c) * p_W_given_S_R(w, s, r)


p_C_given_W = np.zeros(2)
for c in range(2):
    for s in range(2):
        for r in range(2):
            p_C_given_W[c] += p[c, s, r, 1]

p_C_given_W /= np.sum(p_C_given_W)

print('There is a {:.2f}% chance it is cloudy given the grass is wet'.format(p_C_given_W[1] * 100))

##2. ancestral sampling and rejection:
# https://www.cs.ubc.ca/~fwood/CS532W-539W/lectures/mcmc.pdf

num_samples = 10000
samples = np.zeros(num_samples)
rejections = 0
i = 0
while i < num_samples:
    c = np.argmax(np.random.multinomial(1, [p_C(0), p_C(1)]))
    s = np.argmax(np.random.multinomial(1, [p_S_given_C(0, c), p_S_given_C(1, c)]))
    r = np.argmax(np.random.multinomial(1, [p_R_given_C(0, c), p_R_given_C(1, c)]))
    w = np.argmax(np.random.multinomial(1, [p_W_given_S_R(0, s, r), p_W_given_S_R(1, s, r)]))
    if w != 1:
        rejections += 1
        continue
    else:
        samples[i] = c
        i += 1

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean() * 100))
print('{:.2f}% of the total samples were rejected'.format(100 * rejections / (samples.shape[0] + rejections)))

# # 3: Gibbs
# we can use the joint above to condition on the variables, to create the needed
# conditional distributions:


# we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
# indexing is [c,s,r,w]
p_R_given_C_S_W = p / p.sum(axis=2, keepdims=True)
p_S_given_C_R_W = p / p.sum(axis=1, keepdims=True)

# but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
# but since p(W|S,R) does not depend on C, we can factor it out:
# p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

# first create p(R,S,C):
p_C_S_R = np.zeros((2, 2, 2))  # c,s,r
for c in range(2):
    for s in range(2):
        for r in range(2):
            p_C_S_R[c, s, r] = p_C(c) * p_S_given_C(s, c) * p_R_given_C(r, c)

# then create the conditional distribution:
p_C_given_S_R = p_C_S_R[:, :, :] / p_C_S_R[:, :, :].sum(axis=(0), keepdims=True)

##gibbs sampling
num_samples = 10000
samples = np.zeros(num_samples)
state = np.zeros(4, dtype='int')
# c,s,r,w, set w = True

c, s, r, w = 0, 1, 2, 3
i = 0
state[w] = 1
while i < num_samples:
    state[c] = np.argmax(np.random.multinomial(1, p_C_given_S_R[:, state[s], state[r]]))
    state[s] = np.argmax(np.random.multinomial(1, p_S_given_C_R_W[state[c], :, state[r], state[w]]))
    state[r] = np.argmax(np.random.multinomial(1, p_R_given_C_S_W[state[c], state[s], :, state[w]]))

    samples[i] = state[c]
    i += 1


print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean() * 100))
