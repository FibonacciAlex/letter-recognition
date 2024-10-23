#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H



#include <stdlib.h>
#include <math.h>


#include <QDebug>

//-----------------------
//file manipulation
#include <QFile>
#include <QTextStream>
#include <QDataStream>
//-----------------------

#include "globalVariables.h"

using namespace std;

class Backpropagation
{
public:
    Backpropagation();
    void initialise();
    void saveWeights(QString fileName);
    void loadWeights(QString fileName);
    void feedForward( );
    void backPropagate(bool l2checked);
    double sigmoid( double val );
    double sigmoidDerivative( double val );
    void assignRandomWeights( void );
    double trainNetwork(bool checkbox_L2,int batchsize, QString dataset);
    double getError_SSE();
    int action( double *vector );
    double* testNetwork(LetterStructure testPattern);
    double RAND_WEIGHT();
    double relu(double val);
    double reluDerivative(double val);
    double tanhDerivative(double val);
    double* softmax(double* input, int size);
    double he_init(int fan_in);
    double xavier_init(int fan_in, int fan_out);
    void saveLogs(QString fileName, int epoch, double SSE, double LEARNING_RATE, double L2_LAMBDA, QString dataset, QString activationFunc);
    void clearlogfiles(QString fileName);
    bool shuffleDataset(QString fileName, QString& shuffledFileName);
    QString generateLetter(int index);
    bool saveConfusionMatrixToCSV(QString fileName);


private:

    /* Weight Structures */

    /* Input to Hidden1 Weights (with Biases) */
    double wih1[INPUT_NEURONS+1][HIDDEN1_NEURONS];

    /* Hidden1 to Hidden2 weights (with Biases) */
    double wih2[HIDDEN1_NEURONS+1][HIDDEN2_NEURONS];

    /* Hidden2 to Output Weights (with Biases) */
    double who[HIDDEN2_NEURONS+1][OUTPUT_NEURONS];

    /* Activations */
    double inputs[INPUT_NEURONS];
    double hidden1[HIDDEN1_NEURONS];
    double hidden2[HIDDEN2_NEURONS];  //Define the second hidden layer
    double target[OUTPUT_NEURONS];
    double actual[OUTPUT_NEURONS];

    /* Unit Errors */
    double erro[OUTPUT_NEURONS];
    double errh1[HIDDEN1_NEURONS];
    double errh2[HIDDEN2_NEURONS];

    //-----------------------------------------
    double SSE;
    int i, sample, iterations;
    int sum;
};

#endif // BACKPROPAGATION_H
