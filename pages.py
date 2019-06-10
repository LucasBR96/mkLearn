import pandas

stateTrans = {}
initialStates = {}
values  = range( 10 )

if __name__ == "__main__":

    transitions = open("" , mode = 'r' )

    #getting the data
    for line in transitions:
        start , end = line.rstrip().split( sep = ',' )
        stateTrans[ ( start , end ) ] = stateTrans.get(  ( start , end ) , 0 ) + 1
        initialStates[ start ] = initialStates.get( start , 0 ) + 1
    transitions.close()

    for key , val in stateTrans.iteritems():
        start , end = key
        stateTrans[ key ] = val/initialStates[ start ]

    print("Initial rate")
    for x in values:

        val = stateTrans[ ( '-1' , str(x) ) ]
        print( str(x) + str(val) )

    print("Boucing rate")
    for x in values:

        val = stateTrans[ ( str(x) , 'B' ) ]
        print( str(x) + str(val) )

    

    

    

