import os
import numpy
import random

os.chdir(r'C:\Coding\python\Markov machine learning')

class HMM:

    @staticmethod
    def sample( arr ):

        seq = range( len( arr ) )
        result = random.choices( seq , arr ).pop()
        return result

    @staticmethod
    def generateArr( d1 , d2 ):

        arr = numpy.random.rand( d1 , d2 )
        for i in range(d1):
            arr[i] /= arr[i].sum()
        
        return arr

    @staticmethod
    def evaluate( data , ratio = ( 8 , 2 ), *args ):

        N = len( data )
        train , test = ratio
        edge = int( N*train/( train + test ) )

        trainData = data[ : edge ]
        testData = data[ edge : ]

        for x in args:
            
            x.fit( trainData , clock = 0 )
            cTrain = x.logLikelihoodMulti( trainData ).sum()
            cTest = x.logLikelihoodMulti( testData ).sum()

            print( " M = " , x.M )
            print("cost on training: ", cTrain )
            print("cost on testing: ", cTest )
            print()
        
    def __init__(self, M ):

        self.M = M
        self.PI = numpy.ones( M )/M
        self.A = HMM.generateArr( M , M )


    def initObservations(self, data ):

        self.K = max( max( line ) for line in data ) + 1
        self.B = HMM.generateArr( self.M , self.K )
    
    def fit(self, data , clock = 5 , maxIters = 50):

        numpy.random.seed(123)

        self.initObservations( data )
        N = len( data )

        costs = []
        #main learning loop
        for it in range( maxIters ):

            A = numpy.zeros( self.A.shape )
            B = numpy.zeros( self.B.shape )
            PI = numpy.zeros( self.M )
            cost = 0

            for seq in data:

                alpha = self.foward( seq )
                beta = self.backward( seq )
                psi,fi = self.baumWelch( seq , alpha , beta )

                dPI , dA , dB = self.getGrads( psi , fi , seq )
                A += dA
                B += dB
                PI += dPI

                cost += numpy.log( alpha[ -1 ].sum() )
            
            self.A = A/N
            self.B = B/N
            self.PI = PI/N

            costs.append( cost )
            if clock != 0:
                if it%clock == 0:
                    print("it {}".format( it ))
                    print("cost {}".format( cost ))



    def foward(self, seq):

        A = self.A
        B = self.B 
        PI = self.PI

        T = len( seq )
        alpha = numpy.zeros( (T , self.M ) )

        #UNVECTORIZED_________________________________________________________________________________________________

        # for i in range( self.M ):
        #     alpha[ 0 , i ] = PI[i]*B[ i , seq[0] ]
        
        # for t in range( 1 , T):
        #     for i in range( self.M ):
        #         b = B[ i , seq[t] ]
                
        #         soma = 0
        #         for j in range( self.M ):
        #             soma += alpha[ t - 1 , j ]*A[ j , i ] 
        #         alpha[ t , i ] = b*soma
        
        #VECTORIZED_________________________________________________________________________________________________

        alpha[ 0 ] =  PI * B[ : , seq[0] ] 
        for t in range( 1 , T):
            a = alpha[ t - 1 ]@A 
            alpha[ t ] = a*B[ : , seq[t] ]

        return alpha
    
    def backward(self , seq):

        A = self.A
        B = self.B 
        PI = self.PI
        T = len( seq )
        beta = numpy.zeros( (T , self.M) )
        
        #UNVECTORIZED_________________________________________________________________________________________________

        # beta[-1] = 1
        # for t in range(T - 2, -1 , -1):
        #     for i in range( self.M ):
        #         for j in range( self.M ):
        #             beta[ t , i ] += A[ i , j ]*B[ j , seq[ t + 1 ] ]*beta[ t + 1 , j ]

        #VECTORIZED_________________________________________________________________________________________________
        
        beta[-1] = 1
        for t in range(T - 2, -1 , -1):
            b = B[ : , seq[t + 1] ]*beta[t + 1]
            beta[t] = A@b

        return beta

    def baumWelch( self , seq , alpha , beta ):

        T = len( seq )
        M = self.M

        fi = numpy.zeros( ( T , M ) )
        psi = numpy.zeros( ( T - 1, M , M ) )
        
        #UNVECTORIZED-----------------------------------------------------------------------------------------------
        
        # for t in range(T - 1):
        #     den = 0
        #     for k in range( M ):
        #         den += alpha[ t , k ]*beta[ t , k ]            
        #     for i in range( M ):
        #         fi[ t , i ] = ( alpha[ t , i]*beta[ t , i ] )/den
        #         for j in range( M ):
        #             a = self.A[ i , j ]
        #             b = self.B[ j , seq[ t + 1 ] ]
        #             psi[ t , i , j ] = ( alpha[ t , i]*a*b*beta[ t + 1 , j ] )/den
        # t += 1
        # den = 0
        # for k in range( M ):
        #     den += alpha[ t , k ]*beta[ t , k ]            
        # for i in range( M ):
        #     fi[ t , i ] = ( alpha[ t , i]*beta[ t , i ] )/den

        #VECTORIZED_________________________________________________________________________________________________
        
        for t in range( T - 1 ):

            den = (alpha[t]*beta[t]).sum()
            fi[t] = alpha[t]*beta[t]/den

            for i in range( M ):
                prod = self.A[i]*self.B[ : , seq[t + 1] ]*beta[ t + 1 ]
                psi[ t , i ] = (prod*alpha[ t , i ])/den
        
        t += 1
        den = (alpha[t]*beta[t]).sum()
        fi[t] = alpha[t]*beta[t]/den

        return psi,fi

    def getGrads( self , psi , fi , seq):

        dA = numpy.zeros( self.A.shape )
        dB = numpy.zeros( self.B.shape )
        dPI = numpy.zeros( self.M )

        T = len( seq )

        #UNVECTORIZED-----------------------------------------------------------------------------------------------

        # for i in range( self.M ):
        #     dPI[ i ] = fi[ 0 , i]
        
        # for i in range( self.M ): #for A

        #     den = 0
        #     for t in range( T - 1 ):
        #         den += fi[ t , i]
            
        #     for j in range( self.M ):

        #         num = 0
        #         for t in range( T - 1 ):
        #             num += psi[ t , i , j ]
                
        #         dA[ i , j ] = num/den
                
        # for k in range( self.K ): #for B

        #     for i in range( self.M ):

        #         num = 0
        #         den = 0
        #         for t in range( T ):

        #             den += fi[ t , i ]
        #             if seq[ t ] == k:
        #                 num += fi[ t , i ]
        #         dB[ i , k ] = num/den
            

        #VECTORIZED-----------------------------------------------------------------------------------------------
        
        dPI = fi[0]

        den = numpy.sum( fi[ :-1 ] , axis = 0)
        num = numpy.sum( psi, axis = 0 )
        for i in range( self.M ):
            dA[i] = num[i]/den[i] 
        
        den = numpy.sum( fi , axis = 0)
        for t in range( T ):
            x = seq[t]
            dB[ : , x ] += fi[t]/den
        
        return dPI,dA,dB
    
    def likelihood(self, seq):
        a = self.foward( seq )
        return a[-1].sum()

    def multiLikelyhood(self , data):
        return numpy.array([ self.likelihood(line) for line in data])

    def logLikelihoodMulti( self , data ):
        return numpy.log(self.multiLikelyhood(data)) 

    def viterbi(self , seq):
    
        M = self.M
        T = len(seq)
        delta = self.PI*self.B[ : , seq[0] ] 
        psi = numpy.zeros( (T , M ) )

        #UNVECTORIZED----------------------------------------------------------------------------------------------

        # for t in range( 1 , T ):
        #     nuDelta = numpy.zeros( M )
        
        #     for i in range( self.M ):

        #         maxProd = -1
        #         maxIdx = - 1
        #         b = self.B[ i , seq[ t ] ]
        #         for j in range( self.M ):

        #             prod = delta[ j ]*self.A[ j , i ]
        #             if prod > maxProd:
        #                 maxProd = prod
        #                 maxIdx = j

        #         nuDelta[ i ] = b*maxProd
        #         psi[ t , i ] = maxIdx
        
        
        #VECTORIZED------------------------------------------------------------------------------------------------
        for t in range( 1 , T ):
            nuDelta = numpy.zeros( M )

            for j in range( M ):
                
                prod = delta*self.A[ : , j] 
                bVal = self.B[ j , seq[t] ]

                nuDelta[ j ] = numpy.max( prod )*bVal
                psi[ t , j ] = numpy.argmax( prod )
            
            delta = nuDelta

        #backtracking ( Same for both versions )
        result = numpy.zeros( T ,dtype = int )
        result[ -1 ] = numpy.argmax( delta )

        for i in range( T - 2 , -1 , -1 ):
            result[ i ] = psi[ i + 1 , result[ i + 1 ] ]

        return result

    def generateSeq( self , size):

        a = self.sample(self.PI)
        b = self.sample(self.B[a])
        result = [b]

        for i in range(1,size):
            a = self.sample(self.A[a])
            b = self.sample(self.B[a])
            result.append(b)
        
        return result

class SHMM( HMM ):

    def __init__( self , M ):
        HMM.__init__( self , M )
    
    def fit( self, data , clock = 5 , maxIters = 50 ):

        self.initObservations( data )
        N = len( data )

        costs = []
        #main learning loop
        for it in range( maxIters ):

            denA = numpy.zeros( self.M )
            numA = numpy.zeros( self.A.shape )

            denB = numpy.zeros(self.M)
            numB = numpy.zeros( self.B.shape )

            PI = numpy.zeros( self.M )
            cost = 0

            for seq in data:

                c , alpha  = self.foward( seq )
                beta = self.backward( seq , c)
                
                PI += alpha[0]*beta[0]

                a , b = self.setDenominators( alpha, beta )
                denA += a
                denB += b

                a , b = self.setNumerators( seq , alpha , beta , c)
                numA += a
                numB += b                

                cost += numpy.log( c ).sum()
            
            self.PI = PI/N

            for i in range( self.M ):
                self.A[i] = numA[i]/denA[i]
                self.B[i] = numB[i]/denB[i]
                    
            costs.append( cost )
            if clock != 0:
                if it%clock == 0:
                    print("it {}".format( it ))
                    print("cost {}".format( cost ))
                    
    
    #Auxiliary for self.fit and self.likelyhood
    def foward( self , seq ):

        T = len( seq )
        alpha = numpy.zeros( (T , self.M) )
        c = numpy.zeros( T )

        alpha[0] = self.PI[ 0 ]*self.B[ : , seq[0] ]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]

        for t in range( 1 , T ):
            aPrime = self.B[ : , seq[t] ]*( alpha[ t - 1 ]@self.A )
            c[ t ] = aPrime.sum()
            alpha[t] = aPrime/c[t]

        return c , alpha
    
    # Auxiliary for self.fit
    def backward( self , seq , c):

        T = len( seq )
        beta = numpy.zeros( (T , self.M) )

        beta[ -1 ] = 1
        for t in range( T - 2 , -1 , -1 ):
            num = self.A@( beta[ t + 1 ]*self.B[ : , seq[ t + 1 ] ] )
            den = c[ t + 1 ]
            beta[ t ] = num/den
            
        return beta
    
    # Auxiliary for self.fit 
    def setNumerators( self , seq, alpha, beta, c ):

        T = len( seq )

        numA = numpy.zeros( self.A.shape )
        numB = numpy.zeros( self.B.shape )


        #UNVECTORIZED-----------------------------------------------------------------------------------------------
        for i in range( self.M ):
            for j in range( self.M ):
                for t in range( T - 1 ):
                    a = alpha[ t , i ]*self.A[ i , j]
                    b = beta[ t + 1 , j  ]*self.B[ j , seq[ t + 1] ]

                    numA[ i , j ] += (a*b)/c[ t + 1 ]

        for t in range( T ):
            x = seq[ t ]
            for i in range( self.M ):
                numB[ i , x ] += alpha[ t , i ]*beta[ t , i ]

        #VECTORIZED---------------------------------------------------------------------------------------------

        # for t in range( T - 1 ):
        #     a = alpha[t]*self.A
        #     b = beta[ t + 1 ]*self.B[  : , seq[ t + 1 ] ]
        #     numA += (b*a)/c[ t + 1 ]
        
        # for t in range(T):
        #     x = seq[ t ]
        #     numB[ : , x ] += alpha[ t ]*beta[ t ]

        return numA , numB 

    def setDenominators( self, alpha , beta ):

        T,M = alpha.shape
        denA = numpy.zeros( M )
        denB = numpy.zeros( M )

        #UNVECTORIZED--------------------------------------------------------------------------------------------
        #print(T)
        for i in range(M):
            for t in range( T - 1 ):
                d = alpha[ t , i ]*beta[ t , i ]
                denA[i] += d
                denB[i] += d
            d = alpha[ -1 , i ]*beta[ -1 , i ]
            denB[ i ] += d

        #VECTORIZED--------------------------------------------------------------------------------------------
        # P = alpha*beta
        # denB = P.sum( axis = 1 )
        # denA = P[ : T - 1].sum( axis = 1 )

        return denA , denB

    def likelihood( self , seq ):
        c , alpha = self.foward( seq )
        return numpy.log( c ).sum()

    def logLikelihoodMulti( self , data ):
        return numpy.array( [ self.likelihood( x ) for x in data ] )


def test():

    X = []
    f = open('coin_data.txt')
    for line in f:
        x = [0 if e == "H" else 1 for e in line.rstrip()]
        X.append(x)
    f.close()

    H = SHMM(4)

    H.fit(X, clock = 10 , maxIters = 80)
    L = H.logLikelihoodMulti(X).sum()
    print('\n' + "fit {}".format(L))

    H1 = HMM(3)
    H1.PI = numpy.array([0.5, 0.4, .1])
    H1.A =  numpy.array([[0.1, 0.8, .1], [0.7, 0.2, .1], [.3, .3, .4]])
    H1.B = numpy.array([[0.6, 0.4], [0.3, 0.7], [.5, .5]])
    L = H1.logLikelihoodMulti(X).sum()
    print("true {}".format(L))

def main():

    X = []
    f = open('coin_data.txt')
    for line in f:
        x = [0 if e == "H" else 1 for e in line.rstrip()]
        X.append(x)
    f.close()

    H = HMM(4)

    H.fit(X, clock = 10)
    L = H.logLikelihoodMulti(X).sum()
    print("fit {}".format(L))
    # H1.generateSeq(10)

    H1 = HMM(3)
    H1.PI = numpy.array([0.5, 0.4, .1])
    H1.A =  numpy.array([[0.1, 0.8, .1], [0.7, 0.2, .1], [.3, .3, .4]])
    H1.B = numpy.array([[0.6, 0.4], [0.3, 0.7], [.5, .5]])
    L = H1.logLikelihoodMulti(X).sum()
    print("true {}".format(L))


if __name__ == "__main__":
    #main()
    test()