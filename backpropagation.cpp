#include "backpropagation.h"


#define sqr(x)	((x) * (x))

#define getSRand()	((float)rand() / (float)RAND_MAX)
#define getRand(x)      (int)((float)(x)*rand()/(RAND_MAX+1.0))


///////////////////////////////////////////////////////////////////

double Backpropagation::RAND_WEIGHT(){

    return ( (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5) );
}

double Backpropagation::he_init(int fan_in) {
    // Use He initialization, suitable for ReLU activation function
    double range = sqrt(2.0 / fan_in);
    return (rand() / double(RAND_MAX)) * 2 * range - range;
}


double Backpropagation::xavier_init(int fan_in, int fan_out) {
    // Use Xavier initialization, suitable for sigmoid or tanh activation function
    double range = sqrt(6.0 / (fan_in + fan_out));
    return (rand() / double(RAND_MAX)) * 2 * range - range;
}


Backpropagation::Backpropagation()
{
   initialise();
}

void Backpropagation::initialise()
{
    SSE = 0;
    sample=0;
    iterations=0;
    sum = 0;

    /* Seed the random number generator */
    srand(static_cast<unsigned int>(123) );

    assignRandomWeights();
}

double Backpropagation::getError_SSE(){
    return SSE;
}



void Backpropagation::saveWeights(QString fileName){
    int out, hid2,hid1, inp;

    QFile file4(fileName);
    file4.open(QIODevice::WriteOnly | QIODevice::Text);

    QTextStream out4(&file4);

    char tempBuffer4[150];
    QByteArray temp4;

    //----------------------------------------------
    // weights
    //
    qDebug() << "updating weights...";
    qDebug() << "OUTPUT_NEURONS = " << OUTPUT_NEURONS;
    qDebug() << "HIDDEN2_NEURONS = " << HIDDEN2_NEURONS;
    qDebug() << "HIDDEN1_NEURONS = " << HIDDEN1_NEURONS;
    qDebug() << "INPUT_NEURONS = " << INPUT_NEURONS;

    // Update the weights for the output layer (step 4 for output cell)
    for (out = 0 ; out < OUTPUT_NEURONS ; out++) {
      temp4.clear();
      for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {
          //---save------------------------------------
            ::sprintf(tempBuffer4,"%f,",who[hid2][out]);
            // qDebug() << tempBuffer4;
            temp4.append(tempBuffer4);
          //---------------------------------------
      }

      // Update the Bias
      //---save------------------------------------
        ::sprintf(tempBuffer4,"%f",who[HIDDEN2_NEURONS][out]);
        temp4.append(tempBuffer4);
        temp4.append("\n");
        // qDebug() << tempBuffer4 << "\n";
        out4 << temp4;
      //---------------------------------------

    }
    // Update the weights for the hidden2 layer (step 4 for hidden cell)
    for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {
        temp4.clear();
        for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {

            //---save------------------------------------
            ::sprintf(tempBuffer4,"%f,",wih2[hid1][hid2]);
            temp4.append(tempBuffer4);
            // qDebug() << tempBuffer4;
            //---------------------------------------
        }

        // Update the Bias
        //---save------------------------------------
        ::sprintf(tempBuffer4,"%f",wih2[HIDDEN1_NEURONS][hid2]);
        temp4.append(tempBuffer4);
        temp4.append("\n");
        // qDebug() << tempBuffer4 << "\n";
        out4 << temp4;
        //---------------------------------------

    }

    // Update the weights for the hidden1 layer (step 4 for hidden cell)
    for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {
      temp4.clear();
      for (inp = 0 ; inp < INPUT_NEURONS ; inp++) {

        //---save------------------------------------
          ::sprintf(tempBuffer4,"%f,",wih1[inp][hid1]);
          temp4.append(tempBuffer4);
          // qDebug() << tempBuffer4;
        //---------------------------------------
      }

      // Update the Bias
      //---save------------------------------------
        ::sprintf(tempBuffer4,"%f",wih1[INPUT_NEURONS][hid1]);
        temp4.append(tempBuffer4);
        temp4.append("\n");
        // qDebug() << tempBuffer4 << "\n";
        out4 << temp4;
      //---------------------------------------

    }
    //----------------------------------------------
    file4.close();
    qDebug() << "Weights saved to file.";
}


void Backpropagation::loadWeights(QString fileName){
    int out, hid2,hid1, inp;

    QFile file4(fileName);
    file4.open(QIODevice::ReadOnly | QIODevice::Text);

    if(!file4.exists()){
        qDebug() << "Backpropagation::loadWeights-file does not exist!";
        return;
    }

    QTextStream in(&file4);

    char tChar;


    //----------------------------------------------
    // weights
    //

    QString strLine;

    qDebug() << "loading weights...";
    qDebug() << "OUTPUT_NEURONS = " << OUTPUT_NEURONS;
    qDebug() << "HIDDEN2_NEURONS = " << HIDDEN2_NEURONS;
    qDebug() << "HIDDEN1_NEURONS = " << HIDDEN1_NEURONS;
    qDebug() << "INPUT_NEURONS = " << INPUT_NEURONS;

    //qDebug() << &fixed << qSetRealNumberPrecision(12);
    qDebug() <<  qSetRealNumberPrecision(12);

    // Update the weights for the output layer (step 4 for output cell)
    for (out = 0 ; out < OUTPUT_NEURONS ; out++) {
      strLine = in.readLine();
      QTextStream streamLine(&strLine);

      streamLine.setRealNumberPrecision(12);
      qDebug() << "strLine = " << strLine << "\n";
      for (hid2 = 0 ; hid2 <= HIDDEN2_NEURONS ; hid2++) {
          //---load------------------------------------

            if(hid2 != HIDDEN2_NEURONS){
               streamLine >> who[hid2][out] >> tChar;
               qDebug() << who[hid2][out];
            } else {
               streamLine >> who[hid2][out]; //load the bias
               qDebug() << who[hid2][out];
            }
          //---------------------------------------
      }

    }

    // Update the weights for the hidden2 layer (step 4 for hidden2 cell)
    for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {
        for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {

                //---load------------------------------------
            if(hid2 != HIDDEN1_NEURONS-1){
                in >> wih2[hid1][hid2] >> tChar;
                qDebug() << wih2[hid1][hid2] ;
            } else {
                in >> wih2[hid1][hid2];
                qDebug() << wih2[hid1][hid2];
            }
            //---------------------------------------
        }

        // Update the Bias
        //---load------------------------------------
        in >> wih2[HIDDEN1_NEURONS][hid2] >> tChar;
        qDebug() << wih2[HIDDEN1_NEURONS][hid2] << "\n";
        //---------------------------------------

    }

    // Update the weights for the hidden1 layer (step 4 for hidden1 cell)
    for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {
      for (inp = 0 ; inp < INPUT_NEURONS ; inp++) {

        //---load------------------------------------
          if(hid1 != INPUT_NEURONS-1){
             in >> wih1[inp][hid1] >> tChar;
             qDebug() << wih1[inp][hid1] ;
          } else {
             in >> wih1[inp][hid1];
             qDebug() << wih1[inp][hid1];
          }
        //---------------------------------------
      }

      // Update the Bias
      //---load------------------------------------
        in >> wih1[INPUT_NEURONS][hid1] >> tChar;
        qDebug() << wih1[INPUT_NEURONS][hid1] << "\n";
      //---------------------------------------

    }

    //----------------------------------------------
    file4.close();
    qDebug() << "Weights loaded.";
}


int Backpropagation::action( double *vector )
{
  int index, sel;
  double max;

  sel = 0;
  max = vector[sel];

  for (index = 1 ; index < OUTPUT_NEURONS ; index++) {
    if (vector[index] > max) {
      max = vector[index];
      sel = index;
    }
  }
 // qDebug() << "Classification index = " << sel << ", with actual output=" << vector[sel];
  return( sel );
}

double* Backpropagation::testNetwork(LetterStructure testPattern){
    //retrieve input patterns
    for(int j=0; j < INPUT_NEURONS; j++){
       inputs[j] = double(testPattern.f[j]);
    }

    for(int i=0; i < OUTPUT_NEURONS; i++){
        target[i] = double(testPattern.outputs[i]);
    }

    feedForward();


    return actual;


}

double Backpropagation::trainNetwork(bool checkbox_L2, int batchsize, QString dataset)
{
    double err;
    int total_patterns = NUMBER_OF_PATTERNS;
    if (!patternsLoadedFromFile) {
        qDebug() << "unable to train network, no training patterns loaded.";
        return -999.99;
    }

    double accumulatedErr = 0.0;
    err = 0.0;
    sample = 0;

    if(dataset=="train"){
        total_patterns = NUMBER_OF_TRAINING_PATTERNS;
    }else if(dataset=="test"){
        total_patterns = NUMBER_OF_TEST_PATTERNS;
    }

    // Process batchsize samples each time
    while (sample < total_patterns) {
        // Error accumulation in batch processing
        double batchErr = 0.0;

        // Process a batch of samples
        for (int b = 0; b < batchsize && sample < total_patterns; b++) {
            // Get input samples
            for (int j = 0; j < INPUT_NEURONS; j++) {
                inputs[j] = letters[sample].f[j];
            }

            // Get output target
            for (int i = 0; i < OUTPUT_NEURONS; i++) {
                target[i] = letters[sample].outputs[i];
            }

            // Forward Propagation
            feedForward();

            // Calculation error
            err = 0.0;
            for (int k = 0; k < OUTPUT_NEURONS; k++) {
                err += sqr((letters[sample].outputs[k] - actual[k]));
            }
            err = 0.5 * err;
            batchErr += err;  // Accumulate the error of the current batch

            sample++;  // Move to next sample
        }

        // Back propagation, update weights
        backPropagate(checkbox_L2);

        // Accumulate the error of the current batch
        accumulatedErr += batchErr;
    }
    //TO debug
    // qDebug() << "1 epoch training complete with batch size = " << batchsize;
    return accumulatedErr;
}





/*
 *  assignRandomWeights()
 *
 *  Assign a set of random weights to the network.
 *
 */

void Backpropagation::assignRandomWeights( void )
{
    int hid1, hid2, inp, out;

    // Xavier or He initialization for input to hidden1 weights
    for (inp = 0 ; inp < INPUT_NEURONS+1 ; inp++) {
        for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {
            if(activation_Function == 0){
                wih1[inp][hid1] = he_init(INPUT_NEURONS);  // Activation Function is ReLU, using He to initialize
            }else if(activation_Function == 1){
                wih1[inp][hid1] = xavier_init(INPUT_NEURONS, HIDDEN1_NEURONS);  // Activation Function is Tanh, using xavier to initialize
            }


        }
    }

    // Xavier or He initialization for hidden1 to hidden2 weights
    for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS+1 ; hid1++) {
        for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {
            if(activation_Function == 0){
               wih2[hid1][hid2] = he_init(HIDDEN1_NEURONS);  // Activation Function is ReLU, using He to initialize
            }else if(activation_Function == 1){
              wih2[hid1][hid2] = xavier_init(HIDDEN1_NEURONS, HIDDEN2_NEURONS); // Activation Function is Tanh, using xavier to initialize
            }

        }
    }

    // Xavier or He initialization for hidden2 to output weights
    for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS+1 ; hid2++) {
        for (out = 0 ; out < OUTPUT_NEURONS ; out++) {
            if(activation_Function == 0){
                who[hid2][out] = he_init(HIDDEN2_NEURONS);  // Activation Function is ReLU, using He to initialize
            }else if(activation_Function == 1){
                who[hid2][out] = xavier_init(HIDDEN2_NEURONS, OUTPUT_NEURONS); // Activation Function is Tanh, using xavier to initialize
            }

        }
    }

}


/*
 *  sigmoid()
 *
 *  Calculate and return the sigmoid of the val argument.
 *
 */

double Backpropagation::sigmoid( double val )
{
  return (1.0 / (1.0 + exp(-val)));
}


/*
 *  sigmoidDerivative()
 *
 *  Calculate and return the derivative of the sigmoid for the val argument.
 *
 */

double Backpropagation::sigmoidDerivative( double val )
{
  return ( val * (1.0 - val) );
}

/*
 *  relu()
 *
 *  Calculate and return the ReLU of the val argument.
 *
 */
double Backpropagation::relu(double val)
{
    return max(0.0,val);
}

/*
 *  reluDerivative()
 *
 *  Calculate and return the derivative of the ReLU for the val argument.
 *
 */
double Backpropagation::reluDerivative(double val) {
    return val > 0 ? 1 : 0;
}

/*
 *  tanhDerivative()
 *
 *  Calculate and return the derivative of the Tanh for the val argument.
 *
 */
double Backpropagation::tanhDerivative(double val) {
    return 1.0 - val * val;
}

/*
 * softmax()
 * Calculate and return
 *
*/
double* Backpropagation::softmax(double* input, int size)
{
    double* output = new double[size];
    double sum_exp = 0.0;


    for (int i = 0; i < size; ++i) {
        output[i] = exp(input[i]);  // calculate e^z_i
        sum_exp += output[i];       // Accumulation
    }


    for (int i = 0; i < size; ++i) {
        output[i] /= sum_exp;       // Normalization
    }

    return output;
}



/*
 *  feedForward()
 *
 *  Feedforward the inputs of the neural network to the outputs.
 *
 */

void Backpropagation::feedForward( )
{
  int inp, hid1, hid2, out;
  double sum;
  double* outputLayer = new double[OUTPUT_NEURONS];


  /* Calculate input to hidden1 layer */
  for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {

    sum = 0.0;
    for (inp = 0 ; inp < INPUT_NEURONS ; inp++) {
      sum += inputs[inp] * wih1[inp][hid1];
    }

    /* Add in Bias */
    sum += wih1[INPUT_NEURONS][hid1];

    if(activation_Function == 0){
        /* Activation Function of hidden1 layer is ReLU */
        hidden1[hid1] = relu( sum );
    }else if(activation_Function == 1){
        /* Activation Function of hidden1 layer is Tanh */
        hidden1[hid1] = tanh( sum );
    }

  }

  /* Calculate hidden1 to hidden2 layer */
  for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {

      sum = 0.0;
      for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {
          sum += hidden1[hid1] * wih2[hid1][hid2];
      }

      /* Add in Bias */
      sum += wih2[HIDDEN1_NEURONS][hid2];

      if(activation_Function == 0){
          /* Activation Function of hidden2 layer is ReLU */
          hidden2[hid2] = relu( sum );
      }else if(activation_Function == 1){
          /* Activation Function of hidden2 layer is Tanh */
          hidden2[hid2] = tanh( sum );
      }


  }

  /* Calculate the hidden2 to output layer */
  for (out = 0 ; out < OUTPUT_NEURONS ; out++) {

    sum = 0.0;
    for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {
      sum += hidden2[hid2] * who[hid2][out];
    }

    /* Add in Bias */
    sum += who[HIDDEN2_NEURONS][out];
    outputLayer[out] = sum;


  }

  double* softmaxOutput = softmax(outputLayer, OUTPUT_NEURONS);

  // Store the softmax output into the actual output
  for (out = 0; out < OUTPUT_NEURONS; ++out) {
      actual[out] = softmaxOutput[out];
  }

  // Release
  delete[] outputLayer;
  delete[] softmaxOutput;

}


/*
 *  backPropagate()
 *
 *  Backpropagate the error through the network.
 *
 */

void Backpropagation::backPropagate( bool l2checked )
{
  int inp, hid1,hid2, out;


  /* Calculate the output layer error (step 3 for output cell) */
  for (out = 0 ; out < OUTPUT_NEURONS ; out++) {
      //Softmax & Cross-Entropy
      erro[out] = actual[out]-target[out];
  }

  /* Calculate the hidden2 layer error (step 3 for hidden2 cell) */
  for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {

      errh2[hid2] = 0.0;
      for (out = 0 ; out < OUTPUT_NEURONS ; out++) {
          errh2[hid2] += erro[out] * who[hid2][out];
      }

      if(activation_Function == 0){
          errh2[hid2] *= reluDerivative(hidden2[hid2]);   //ReLU
      }else if(activation_Function == 1){
          errh2[hid2] *= tanhDerivative(hidden2[hid2]);   //Tanh
      }


  }

  /* Calculate the hidden1 layer error (step 3 for hidden1 cell) */
  for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {

      errh1[hid1] = 0.0;
      for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {
          errh1[hid1] += errh2[hid2] * wih2[hid1][hid2];
      }

      if(activation_Function == 0){
          errh1[hid1] *= reluDerivative(hidden1[hid1]); //ReLU
      }else if(activation_Function == 1){
          errh1[hid1] *= tanhDerivative(hidden1[hid1]); //Tanh
      }


  }

  /* Update the weights for the output layer (step 4 for output cell) */
  for (out = 0 ; out < OUTPUT_NEURONS ; out++) {

    for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {

        if(!l2checked){

            who[hid2][out] =who[hid2][out] - LEARNING_RATE * erro[out] * hidden2[hid2];

        }else{
            // L2 regularization term added
            who[hid2][out] = who[hid2][out] - LEARNING_RATE * (erro[out] * hidden2[hid2] + L2_LAMBDA * who[hid2][out]);
        }

    }

    /* Update the Bias (bias doesn't include L2 regularization) */
    who[HIDDEN2_NEURONS][out] -= LEARNING_RATE * erro[out];

  }

  /* Update the weights for the hidden2 layer (step 4 for hidden2 cell) */
  for (hid2 = 0 ; hid2 < HIDDEN2_NEURONS ; hid2++) {

    for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {
        if(!l2checked){

            wih2[hid1][hid2] = wih2[hid1][hid2] - LEARNING_RATE * errh2[hid2] * hidden1[hid1];

        }else{

            // L2 regularization term added
            wih2[hid1][hid2] = wih2[hid1][hid2] - LEARNING_RATE * (errh2[hid2] * hidden1[hid1] + L2_LAMBDA * wih2[hid1][hid2]);

        }

      }

      /* Update the Bias */
      wih2[HIDDEN1_NEURONS][hid2] -= LEARNING_RATE * errh2[hid2];

  }

  /* Update the weights for the hidden1 layer (step 4 for hidden1 cell) */
  for (hid1 = 0 ; hid1 < HIDDEN1_NEURONS ; hid1++) {

    for (inp = 0 ; inp < INPUT_NEURONS ; inp++) {
        if(!l2checked){

            wih1[inp][hid1] =wih1[inp][hid1]- LEARNING_RATE * errh1[hid1] * inputs[inp];
        }else{

            // L2 regularization term added
            wih1[inp][hid1] = wih1[inp][hid1] - LEARNING_RATE * (errh1[hid1] * inputs[inp] + L2_LAMBDA * wih1[inp][hid1]);
        }

      }

      /* Update the Bias */
      wih1[INPUT_NEURONS][hid1] -= LEARNING_RATE * errh1[hid1];
  }

}

/* save log*/
void Backpropagation::saveLogs(QString fileName, int epoch, double SSE, double LEARNING_RATE, double L2_LAMBDA, QString dataset, QString activationFunc){

    QFile logFile(fileName);
    bool isNewFile = !logFile.exists();
    if (!logFile.open(QIODevice::Append | QIODevice::Text)) {
        qDebug() << "Unable to open log file.";
        return;
    }

    QTextStream logStream(&logFile);

    // If the file is new, write the head of columns
    if (isNewFile) {
        logStream << "Epoch,SSE,LearningRate,L2_LAMBDA, Dataset, ActivationFunction\n";
    }

    // write logs
    logStream << epoch << "," << SSE << "," << LEARNING_RATE << "," << L2_LAMBDA <<"," << dataset << "," << activationFunc <<"\n";

    // close the file
    logFile.close();
}

/* Clear log file */
void Backpropagation::clearlogfiles(QString fileName){

    QFile logFile(fileName);

    if (!logFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Unable to open log file.";
        return;
    }

    QTextStream logStream(&logFile);

    // clear
    logStream << "";

    logFile.close();
    qDebug() << "Log file cleared.";

}


bool Backpropagation::shuffleDataset(QString fileName, QString& shuffledFileName){
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Failed to open file: " << fileName;
        return false;
    }

    // Check if the file exists
    if (!file.exists()) {
        qDebug() << "Data file does not exist!";
        return false;
    }

    QTextStream in(&file);
    QStringList datasetLines;

    // Read all lines from the file
    while (!in.atEnd()) {
        QString line = in.readLine();
        datasetLines.append(line);
    }

    file.close();  // Close the file after reading

    // If the file is empty
    if (datasetLines.isEmpty()) {
        qDebug() << "The dataset is empty.";
        return false;
    }

    qDebug() << "Data loaded, now shuffling...";

    // Shuffle the dataset
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();  // Random seed for shuffle
    shuffle(datasetLines.begin(), datasetLines.end(), default_random_engine(seed));

    // Save shuffled data to a new file
    shuffledFileName = fileName.left(fileName.lastIndexOf('.')) + "_shuffled.txt";
    QFile shuffledFile(shuffledFileName);

    if (!shuffledFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Failed to create shuffled file.";
        return false;
    }

    QTextStream out(&shuffledFile);
    for (const QString& line : datasetLines) {
        out << line << "\n";  // Write each shuffled line back to the new file
    }

    shuffledFile.close();
    qDebug() << "Shuffled data saved to file: " << shuffledFileName;

    return true;
}


QString Backpropagation::generateLetter(int index) {
    if (index >= 0 && index < 26) {
        char letter = 'A' + index;
        return QString(1, letter); // Convert char to string
    } else {
        return "Invalid index"; // Return an error message for invalid indices
    }
}

bool Backpropagation::saveConfusionMatrixToCSV(QString fileName){
    QFile file(fileName);

    // Try to open the file for writing
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "Unable to open file for writing: " << fileName;
        return false;  // Return false on failure
    }

    QTextStream out(&file);

    // Write the header row for predicted class labels
    out << "Actual \\ Predicted,";
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        out << generateLetter(i);  // Column headers with class labels
        if (i < OUTPUT_NEURONS - 1) {
            out << ",";
        }
    }
    out << "\n";

    // Write the confusion matrix data
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        // Write the actual class label as the row header
        out << generateLetter(i) << ",";
        for (int j = 0; j < OUTPUT_NEURONS; j++) {
            out << confusionMatrix[i][j];  // Write each element of the matrix
            if (j < OUTPUT_NEURONS - 1) {
                out << ",";  // Add commas between values
            }
        }
        out << "\n";  // New line at the end of each row
    }

    file.close();  // Close the file after writing
    qDebug() << "Confusion matrix successfully saved to: " << fileName;
    return true;  // Return true on successful save
}

