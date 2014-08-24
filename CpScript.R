#######
# Usage:
# source( "CpScript.R" )
# set 'subDir' variable if != local sub-directory 'data'
# cpTest()
#######

library( randomForest )
cpScript <- function( subDir = "data" ) {

    # load data, create and split data frames, shuffle rows, results stored in d tuple
    d <- loadData( subDir, doPrint = TRUE )
    
    # see if rfcv() variable importance useful for features selection
    evalRfcvVarImport( d$trainDf, nrows=2000, doPrint=TRUE )
    
    # get sorted coefficients from Linear Model fit to see if useful
    # for feature selection
    lmBest20Coeffs <- getMostImportantLmCoeffs( d$trainDf, nrows=1000, doPrint=TRUE )

    # fit Random Forests over a grid of ranges for params 'mtry' and 'ntree'
    # use full 52 features in training set
    pr( "---> examining Random Forest performance for a range of 'mtry' and 'ntree' parameters..." )
    mtryVals <- c( 2, 5, 10 )
    ntreeVals <- c( 1, 2, 3, 5, 10, 20 )
    rf52 <- findBestRfParams( d$trainDf, mtryVals, ntreeVals, doPrint = TRUE ) # rf52 is a tuple
    pr( "--> Random Forest accuracy values for evaluated (mtry, ntree) grid pairs:" )
    pr( "--> NOTE: row names are mtry values; column names are ntree values" )
    pr( "--> NOTE: matrix entries are classification accuracy on train set" )
    cat( "\n" )
    pr( rf52$accMatrix )
    cat( "\n" )    
    pr( "--->parameters from best 52-feature Random Forest:" )
    pr( sprintf( "resultant rf52: train accuracy=%f mtry=%d ntree=%d OOB error=%f",
        rf52$bestAcc, rf52$bestMtry, rf52$bestNtree, getOob( rf52$bestRf ) ) )
    cat( "\n" )
    
    # now just print out the Random Forest to get confusion matrix and OOB error est.
    pr( "---> printing best (rf52) Random Forest for above parameters: " )
    pr( rf52$bestRf )
    cat( "\n" )
    
    # now use rfcv() to evaluate cross-validation error
    pr( "---> computing rfcv() cross-validation error - this may take several minutes..." )
    set.seed( 1 )
    rfcvOutput <- rfcv( d$trainDf[, -53], d$trainDf[, 53] )
    pr( "...done computing rfcv() output" )
    pr( "rfcv() cross-validation estimates for training set vs. number variables used:")
    pr( rfcvOutput$error.cv )    
    cat( "\n" )

    # compute performance of rf52 best 52-feature random forest on T-E-S-T set
    acc <- evalRf( rf52$bestRf, d$testDf )
    pr( sprintf( "accuracy of best (rf52) Random Forest on 20%% TEST set: %f", acc ) )
    cat( "\n" )

    # additional exercise: find best 20-feature Random Forest using Linear Model top-20 coeffs
    pr( "---> As additional exercise fit Random Forest to top-20 features from Linear Model" )
    lmBest20Coeffs[ 21 ] = "classe" # Need to append the 'classe' to feature list
    best20TrainDf <- d$trainDf[, lmBest20Coeffs ] # subset training set, only the 20 'top' features
    mtryVals <- c( 2, 5, 10 ) # use different parameter ranges as expect lower accuracy using fewer features
    ntreeVals <- c( 10, 50, 100, 200 ) # ditto
    rf20 <- findBestRfParams( d$trainDf, mtryVals, ntreeVals, doPrint = TRUE )
    pr( "--> Random Forest accuracy values for evaluated (mtry, ntree) grid pairs:" )
    pr( "--> NOTE: row names are mtry values; column names are ntree values" )
    pr( "--> NOTE: matrix entries are classification accuracy on train set" )
    cat( "\n" )

    pr( rf20$accMatrix )
    cat( "\n" )    
    pr( "--->parameters from best 20-feature Random Forest:" )
    pr( sprintf( "resultant rf20: train accuracy=%f mtry=%d ntree=%d OOB error=%f",
                 rf20$bestAcc, rf20$bestMtry, rf20$bestNtree, getOob( rf20$bestRf ) ) )
    cat( "\n" )
    
    # PRINT BEST RANDOM FOREST (to get OOB and confusion matrix)
    pr( "---> printing best Random Forest for above parameters: " )
    pr( rf20$bestRf )
    cat( "\n" )    

    # compute performance of rf20 Random Forest on T-E-S-T set (20% of train set rows)
    pr( "---> evalute top-20 feature Random Forest: " )
    acc <- evalRf( rf20$bestRf, d$testDf )
    pr( sprintf( "accuracy of best (rf20) Random Forest on 20%% TEST set: %f", acc ) )
    cat( "\n" )

    # use rf52 to predict labels for course project 20-row data set (pml-testing.csv)
    pr( "---> use rf52 to predict labels for course project 20-row data set (pml-testing.csv" )
    predsRf52 <- predict( rf52$bestRf, d$predDf )
    pr( predsRf52 )
    cat( "\n")

    # use rf20 to predict labels for course project 20-row data set (pml-testing.csv)
    pr( "---> use rf20 to predict labels for course project 20-row data set (pml-testing.csv" )
    predsRf20 <- predict( rf20$bestRf, d$predDf )
    pr( predsRf20 )
    cat( "\n")
        
    # RETURN LIST OF SELECTED VARIABLES FOR CALLER
    list( data=d, lmBest20Coeffs=lmBest20Coeffs, rf52=rf52, rf20=rf20,
          rfcvOutput=rfcvOutput, predsRf52=predsRf52, predsRf20=predsRf20 )
}

###########
# helper function which fits a Linear Model and returns
# return value = 20 largest sorted coefficients of Linear Model
getMostImportantLmCoeffs <- function( df, nrows=0, doPrint=FALSE ) {
    if ( nrows == 0 )
        nrows = nrow( df )
    # EVAL LINEAR MODEL COEFFICIENT RANKING FOR *FEATURE SELECTION* (result: useful)
    # now try ranking variables by Linear Model coefficient values
    if ( doPrint )
        pr( "---> assessing whether Linear Model coefficients offer useful variable importance rankings..." )
    classeIdx <- getColIdx( df, "classe" )
    y <- as.numeric( df[ 1:nrows, classeIdx ] ) # use only 1,000 rows for LM fit
    lmDf <- cbind( df[ 1:nrows , 1:classeIdx - 1 ], y  )
    set.seed( 1 ) # set RNG seed for reproducibility
    lmFit <- lm( y ~ . , data = lmDf )
    sortedCoeffs <- sort( abs( lmFit$coefficients ), decreasing = TRUE )
    nCoeffs <- length( sortedCoeffs )
    sortedCoeffNames <- names( sortedCoeffs[2:nCoeffs ] )
    lm20MostImpFeatures <- names(sortedCoeffs)[2:21]  # skip intercept = coef[1]
    # plot linear model coefficients largest-to-smallest; skipping intercept coefficient
    plotLmCoeffVals( 2:nCoeffs, sortedCoeffs[2:nCoeffs], doPrint=doPrint )
    if ( doPrint )
        cat( "\n" )
    lmBest20Coeffs <- sortedCoeffNames[2:21]
}

###########
# helper function to use rfcv() output ranking of variables to see
# if the rankings are useful for feature selection/reduction
evalRfcvVarImport <- function( df, nrows=0, doPrint = FALSE ) {
    # evaluate rfcv() output to see if useful for feature selection
    if ( nrows == 0 )
        nrows = nrow( df )
    set.seed( 1 )
    rf <- randomForest( classe ~ ., data = df[1:nrows, ], importance = TRUE )
    impVal <- as.data.frame( importance( rf ) )
    impValMda <- impVal[ rev( order( impVal$MeanDecreaseAccuracy ) ), ]
    impValGini <- impVal[ rev( order( impVal$MeanDecreaseGini ) ), ]
    mdaDf <- data.frame( rownames( impValMda ), impValMda$MeanDecreaseAccuracy )
    giniDf <- data.frame( rownames( impValGini ), impValGini$MeanDecreaseGini )
    nVars <- nrow( mdaDf )
    # plot the results
    par( mfrow = c( 1, 2 ) )
    plotImportanceData( 1:nVars, mdaDf[,2], giniDf[,2], doPrint=doPrint )
    # plot conclusion - no clear dividing line between important vs. unimportant variables
    # results not actionable
    if ( doPrint ) {
        pr( "no clear dividing line differentiating important vs. unimportant variables" )
        cat( "\n" )
    }
}

###########
# helper function to load, subset, and shuffle data
# output: various data frames
# read and subset/process 2 CSV files, create data frames
loadData <- function( subDir = "data", doPrint = FALSE ) {
    if ( doPrint )
        pr( "---> loading data..." )
    trainFile <- file.path( subDir, "pml-training.csv" ) 
    testFile <- file.path( subDir, "pml-testing.csv" )
    trainFileDf <- prepDf( read.csv( trainFile ) )
    predDf <- prepDf( read.csv( testFile ) ) # 20-row prediction file
    set.seed( 1 ) # set seed for shuffle operation
    nrows <- nrow( trainFileDf )
    trainFileDf <- trainFileDf[ sample( nrows ), ] # randomly shuffle rows
    # Partition: 80% train, 20% test
    trainDf <- trainFileDf[ 1 : as.integer( 0.8 * nrows ), ]
    testDf <- trainFileDf[ as.integer( nrow( trainDf ) + 1 ) : nrows, ]
    if ( doPrint ) {
        pr( "...done reading data and creating data frames")
        cat( "\n" )
    }
    list( trainDf=trainDf, testDf=testDf, predDf=predDf )
}

###########
# helper function to evaluate Random Foresets over range of mtry and ntree parameter
findBestRfParams <- function( df, mtryVals, ntreeVals, doPrint = FALSE ) {
    bestAcc <- 0
    bestMtry <- 0
    bestNtree <- 0
    bestRf <- NULL
    accMatrix <- matrix( nrow=length( mtryVals ), ncol=length( ntreeVals ) )
    rownames( accMatrix ) <- as.character( mtryVals )
    colnames( accMatrix ) <- as.character( ntreeVals )
    for ( i in 1:length( mtryVals ) ) {                     # mtry
        mt <- mtryVals[ i ]
        for ( j in 1:length( ntreeVals ) ) {                # ntree
            nt <- ntreeVals[ j ]
            if ( doPrint )
                pr( sprintf( "evaluating Random Forest w/ mtry=%d ntree=%d", mt, nt ) )
            set.seed( 1 )
            rf <- randomForest( classe ~ ., data=df, mtry=mt, ntree=nt )
            acc <- evalRf( rf, df )
            accMatrix[i, j] = acc
            if ( acc > bestAcc ) {
                bestMtry <- mt
                bestNtree <- nt
                bestRf <- rf
                bestAcc <- acc
            }
        }
    }
    if ( doPrint )
        cat( "\n" )
    list( bestAcc=bestAcc, bestMtry=bestMtry, bestNtree=bestNtree, 
          bestRf=bestRf, accMatrix=accMatrix )
}

###########
# helper method to get OOB error estimate from Random Forest
getOob <- function( rf ) {
    rf$err.rate[rf$ntree, 1 ]
}


###########
# helper method to evaluate classification accuracy of Random Forest wrt a data frame
evalRf <- function( rf, df )  {
    predTestSet <- predict( rf, df ) # use Test set, NOT Cross-Validation set
    numAgree <- sum( predTestSet == df$classe )
    modelTestAccur <- numAgree/length( predTestSet )
}

###########
# helper method to reduce line lengths
pr <- function( msg ) {
    print( msg, quote = FALSE )
}

###########
# helper to de-clutter script code
plotLmCoeffVals <- function( coefIdxs, coefVals, doPrint=FALSE ) {
    
    if ( doPrint )
        pr( "plotting Linear Model coefficients sorted in decreasing order" )
    
    # plot to screen and knitr
    plot( coefIdxs, coefVals, pch=21, col="blue", bg="red",
          xlab="Linear Model Coef Index", 
          ylab="LM Coef Value", 
          main="Linear Model Coefficients" )
    lines(coefIdxs, coefVals, col="blue" )

    # also plot to PNG file
    png( "LmCoeff.png", height = 512, width = 512 )
    par( family = "sans" )
    plot( coefIdxs, coefVals, pch=21, col="blue", bg="red",
          xlab="Linear Model Coef Index", 
          ylab="LM Coef Value", 
          main="Linear Model Coefficients" )
    lines(coefIdxs, coefVals, col="blue" )
    dev.off()    
    
    if ( doPrint )
        cat( "\n" )
}

###########
# helper to de-clutter script code
plotImportanceData <- function( varIndices, mdaVals, mdGiniVals, doPrint=FALSE ) {
    
    if ( doPrint )
        pr( "plotting Random Forest variable importance() metrics" )
    
    # plot to screen and knitr
    plot( varIndices, mdaVals, 
          xlab="Variable Index", 
          ylab="Importance MD Accuracy Value", 
          main="Variable MD Accuracy/Importance",
          pch=21, col="blue", bg="red" )
    plot( varIndices, mdGiniVals,
          xlab="Variable Index", 
          ylab="Importance MD Gini Value", 
          main="Variable MD Gini Importance",
          pch=21, col="blue", bg="red" )
    
    # plot to file as well
    png( "RfImpVarMetrics.png", height = 512, width = 900 )
    par( family = "sans" )
    par( mfrow = c( 1, 2 ) )
    plot( varIndices, mdaVals, 
          xlab="Variable Index", 
          ylab="Importance MD Accuracy Value", 
          main="Variable MD Accuracy/Importance",
          pch=21, col="blue", bg="red" )
    plot( varIndices, mdGiniVals,
          xlab="Variable Index", 
          ylab="Importance MD Gini Value", 
          main="Variable MD Gini Importance",
          pch=21, col="blue", bg="red" )
    dev.off()
}

###########
# helper function to subset, coerce data
prepDf <- function( df ) {
    ## discard first 7 columns - may not be good for general data sets
    df <- df[ , 8:ncol( df ) ]
    ## discard columns with number na/blanks > num rows in data frame
    colNaSums <- apply( df, 2, function(x) { length( which ( is.na(x) | x == "" ) ) } )
    df <- df[ ,colNaSums < nrow(df)/2 ]
    ## coerce integers to numeric
    for ( i in 1:ncol( df ) ) {
        if ( class( df[ 1, i ] ) == "integer" )
            df[ , i ] <- as.numeric( df[ , i ] )
    }
    df # Caller must do the shuffling
}

###########
# a function which returns the numeric index of column in a data frame given the column name
getColIdx <- function( df, colName ) {
    grep( colName, colnames( df ) )
}
