import numpy as np


symbol_map = ['H', 'T']
pi = np.array([0.5, 0.4, .1])
A = np.array([[0.1, 0.8, .1], [0.7, 0.2, .1], [.3, .3, .4]])
B = np.array([[0.6, 0.4], [0.3, 0.7], [.5, .5]])
M, V = B.shape


def generate_sequence(N):
    s = np.random.choice(range(M), p=pi) # initial state
    x = np.random.choice(range(V), p=B[s]) # initial observation
    sequence = [x]
    for n in range(N-1):
        s = np.random.choice(range(M), p=A[s]) # next state
        x = np.random.choice(range(V), p=B[s]) # next observation
        sequence.append(x)
    return sequence


def main():
    with open('coin_data.txt', 'w') as f:
        for n in range(10000):
            size = int( 20 + np.random.random()*10)
            sequence = generate_sequence(size)
            sequence = ''.join(symbol_map[s] for s in sequence)
            print(sequence)
            f.write("%s\n" % sequence)


if __name__ == '__main__':
    main()