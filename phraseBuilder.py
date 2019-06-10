import numpy
import string
import os
import mk


os.chdir(r'C:\Users\Lucas\Desktop\Coding\python\Markov machine learning')

def preProcess():

    f = open( "robert_frost.txt" , 'r' )
    tokens = []
    for line in f:

        s = line.rstrip().lower()
        #s = s.translate( str.maketrans( ('' for p in string.punctuation) , string.punctuation ))
        s = s.translate(str.maketrans( '' , '' , string.punctuation ) )
        tokens.append( s.split() )
    f.close()
    return tokens

def turn(tokens):

    vocabulary = set()
    for token in tokens:
        vocabulary |= set(token)
    
    seq = list(vocabulary)
    n = len(vocabulary)
    vocabulary = {}
    for i in range( n ):
        vocabulary[ seq[i] ] = i 
    
    data = []
    for token in tokens:
        if token != []:
            line = [ vocabulary[word] for word in token ]
            data.append(line)

    keyVoc = {}
    for word,num in vocabulary.items():
        keyVoc[num] = word
    
    return keyVoc,data


def translate(seq, vocabulary):
    return " ".join( vocabulary[x] for x in seq )

if __name__ == "__main__":

    tokens = preProcess()
    vocabulary,data = turn(tokens)

    H = mk.SHMM(500)
    #H = mk.SHMM(30)
    H.fit( data , clock = 1, maxIters = 100)

    for i in range(100):
        seq = H.generateSeq(8)
        s = translate(seq,vocabulary)
        print(s)  




