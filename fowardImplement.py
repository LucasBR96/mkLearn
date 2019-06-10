import numpy

piState = numpy.array([ .75 , .25])
hiddenStates = numpy.array([ [.1 , .9 ] , [.6 , .4 ] ] )
obStates = numpy.array( [ [ .0 , 1.] , [.7 , .3] ])
Labels = { "K":0 , "C":1 }


#LAZY PROGRAMMER
def foward( seq ):

    T = len( seq )
    M = hiddenStates.shape[1]

    # alpha = numpy.zeros( ( T , M ) )
    
    pi = piState
    B = obStates[ : , Labels[ seq[0] ] ]
    # alpha[ 0 ] = pi*B
    alpha = pi*B

    for t in range( 1 , T):

        # a = alpha[ t - 1]
        # B = obStates[ : , Labels[ seq[t] ] ]
        # alpha[ t ] = a.dot( hiddenStates )*B

        B = obStates[ : , Labels[ seq[t] ] ]
        alpha = alpha.dot( hiddenStates )*B

    return alpha.sum()

def backward( seq ):

    T = len( seq )
    M = hiddenStates.shape[ 1 ]
    beta = numpy.ones( shape = (M , ))

    for t in range( T - 2 , -1, -1 ):
        B = obStates[ : , Labels[ seq[t] ] ]
        k = hiddenStates.dot(B)
        beta = k*beta
    
    return beta.sum()

def viterbi( seq ):

    T = len( seq )
    M = hiddenStates.shape[ 1 ]
    
    delta = numpy.zeros((T , M))
    psi = numpy.zeros((T , M))

    B = obStates[ : , Labels[ seq[0] ] ]
    delta[0] = piState*B

    for t in range( 1 , T):
        for j in range( M ):

            a = delta[ t - 1]*hiddenStates[ : , j]
            B = obStates[ j , Labels[ seq[t] ] ]
            delta[ t , j] = numpy.max( a )*B
            psi[ t , j ]  = numpy.argmax( a )

    states = numpy.zeros( T , dtype = numpy.int16 )
    
    states[ T - 1] = numpy.argmax( delta[ T - 1] )
    for t in range(T - 2, -1 , -1):
        states[ t ] = psi[ t + 1 , states[t + 1] ]
    return states
    
# MUH VERSION

def muhAlpha( observation , A , B , PI ):

    M = B.shape[1]
    N = len( observation )

    bCol = B[ : , observation[0]]
    alpha = numpy.multiply( PI , bCol )

    for t in range( 1 , N ):
        
        bcol = B[ : , observation[t] ]
        prod = alpha@A
        alpha = numpy.multiply( prod , bCol )
    
    return alpha.sum()
    

# print( p )

def muhBackward(seq , A , B , PI ):

    T = len( seq )
    N = A.shape[1]
    M = B.shape[1]

    beta = numpy.ones( M )

    for t in range( T - 2 , -1 , -1):
        
        bCol = B[ : , seq[t] ]
        beta = numpy.multiply( A@bcol , beta )
    
    return beta.sum()

def muhViterbi( seq , A , B , PI):
    
    N = A.shape[1]
    T = len(seq)
    delta = numpy.multiply( PI , B[ : , seq[0] ] )
    psi = numpy.zeros( (T , N ) )

    for t in range( 1 , T ):
        nuDelta = numpy.zeros( N )

        for j in range( N ):
            
            prod = numpy.multiply( delta , A[ : , j] )
            bVal = B[ j , seq[t] ]

            nuDelta[ j ] = numpy.max( prod )*bVal
            psi[ t , j ] = numpy.argmax( prod )
        
        delta = nuDelta

    result = numpy.zeros( T ,dtype = int )
    result[ -1 ] = numpy.argmax( delta )

    for i in range( T - 2 , -1 , -1 ):
        result[ i ] = psi[ i + 1 , result[ i + 1 ] ]

    return result

#p = muhViterbi( (0,0,1,0,1,1,1,1,1,0,1,1,0) , hiddenStates , obStates , piState  )
p = muhAlpha( (0,0,1,0) , hiddenStates , obStates, piState )
print( p )