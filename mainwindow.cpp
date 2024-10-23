#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "globalVariables.h"


//--------------------------------------

LetterStructure letters[20001];
LetterStructure testPattern;

bool patternsLoadedFromFile;
int MAX_EPOCHS;
double LEARNING_RATE;
double L2_LAMBDA;
int BATCH_SIZE;
int activation_Function;
int confusionMatrix[OUTPUT_NEURONS][OUTPUT_NEURONS];


///////////////////////////////////////////////////////

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //---------------------------------------
    //initialisation of global variables
    //
    LEARNING_RATE=0.001;
    patternsLoadedFromFile = false;
    MAX_EPOCHS = 1000;
    L2_LAMBDA = 0.001;
    BATCH_SIZE = 1000;
    activation_Function = 0;   //0:ReLU, 1:Tanh

    bp = new Backpropagation;

    //---------------------------------------
    //initialise widgets

    ui->spinBox_Max_training_Epochs->setValue(MAX_EPOCHS);
   // ui->horizScrollBar_LearningRate->setValue(int(LEARNING_RATE*1000));
    ui->horizScrollBar_BatchSize->setValue(BATCH_SIZE);

    ui->lcdNumber_L2Regularization->setEnabled(false);       // Disable the LCD display
    ui->horizScrollBar_L2Regularization->setEnabled(false);   // Disable a horizScrollBar
    //ui->horizScrollBar_L2Regularization->setValue(int(L2_LAMBDA*1000));
    ui->comboBox_ActivationFunction->setCurrentIndex(activation_Function);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_Read_File_clicked()
{

    QString fileName = ui->lineEdit_fileNameLoadDataSets->text().trimmed();  // Trim any extra spaces or line breaks

    qDebug() << "\nReading file: "<< fileName;

    // Check if the file name is empty
    if (fileName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a file name.");
        return;
    }

    QFile file(fileName);
    file.open(QIODevice::ReadOnly | QIODevice::Text);

    if(!file.exists()){
        patternsLoadedFromFile=false;
        qDebug() << "Data file does not exist!";
        return;
    }

    QTextStream in(&file);


    char t;
    char characterSymbol;
    QString line;


    QString lineOfData;
    QString msg;
    int i=0;
    int counter[OUTPUT_NEURONS];
    //set the defalut value of counter
    for (int i=0; i < OUTPUT_NEURONS; i++)
    {
        counter[i] = 0;
    }

    while(i < NUMBER_OF_PATTERNS){

        //e.g. T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8
        in >> characterSymbol >> t >> letters[i].f[0] >> t >>  letters[i].f[1] >> t
            >>  letters[i].f[2] >> t >>  letters[i].f[3] >> t >>  letters[i].f[4] >> t
            >>  letters[i].f[5] >> t >>  letters[i].f[6] >> t >>  letters[i].f[7] >> t
            >>  letters[i].f[8] >> t >>  letters[i].f[9] >> t >>  letters[i].f[10] >> t
            >>  letters[i].f[11] >> t >> letters[i].f[12] >> t >> letters[i].f[13] >> t >> letters[i].f[14] >> t >> letters[i].f[15];

        line = in.readLine();

       qDebug() << "Data: Letter " << characterSymbol;

        // Assume uppercase letters only
        int symbolIndex;
        if ((characterSymbol >= 'A') && (characterSymbol <= 'Z'))
        {
            symbolIndex = characterSymbol - 'A';
            if (symbolIndex >= OUTPUT_NEURONS)
                symbolIndex = OUTPUT_NEURONS - 1;
            letters[i].symbol = static_cast<Symbol>(symbolIndex);
        }
        else
        {
            symbolIndex = OUTPUT_NEURONS - 1;
            letters[i].symbol = UNKNOWN;
        }

//        qDebug() << "symbolIndex =  " << symbolIndex;

        // Set all outputs to zero
        for (int j=0; j < OUTPUT_NEURONS; j++)
        {
            letters[i].outputs[j] = 0;
        }

        // Set the output we want to 1
        letters[i].outputs[symbolIndex] = 1;
        counter[symbolIndex]++;

        if( i == (NUMBER_OF_PATTERNS-1)) {
            msg.clear();
            for (int j=0; j < OUTPUT_NEURONS; j++)
            {
                if (j == OUTPUT_NEURONS - 1)
                {
                    //lineOfData.sprintf("Number of patterns for UNKNOWN Letters = %d\n", counter[j]);

                    lineOfData.clear();
                    QTextStream(&lineOfData) << "number of patterns for UNKNOWN Letters = " << counter[j] << Qt::endl;



                }
                else
                {
                    //lineOfData.sprintf("Number of patterns for Letter %c = %d\n", j + 'A', counter[j]);
                    lineOfData.clear();
                    QTextStream(&lineOfData) << "number of patterns for Letter " << j + 'A' << " = "  << counter[j] << Qt::endl;

                }
                msg.append(lineOfData);
            }
            ui->plainTextEdit_results->setPlainText(msg);
            qApp->processEvents();
        }

        i++;
    }

    msg.append("done.");

    ui->plainTextEdit_results->setPlainText(msg);
    qApp->processEvents();

    patternsLoadedFromFile=true;

}

void MainWindow::on_horizScrollBar_LearningRate_valueChanged(int value)
{
    ui->lcdNumber_LearningRate->setSegmentStyle(QLCDNumber::Filled);
    ui->lcdNumber_LearningRate->display(value/1000.0);
    LEARNING_RATE = value/1000.0;
}



void MainWindow::on_pushButton_Classify_Test_Pattern_clicked()
{

    char characterSymbol, t;
    QString *q;
    double* classificationResults;

    double* outputs;
    outputs = new double[OUTPUT_NEURONS];
    if(!outputs) {
        qDebug() << "memory allocation error.";
        exit(-1);
    }

    classificationResults = new double[OUTPUT_NEURONS];
    if(!classificationResults) {
        qDebug() << "memory allocation error.";
        exit(-1);
    }

    //Initialize the confusionMatrix
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        for (int j = 0; j < OUTPUT_NEURONS; j++) {
            confusionMatrix[i][j] = 0;
        }
    }

    //QTextStream line;  Read input pattern from UI
    q = new QString(ui->plainTextEdit_Input_Pattern->toPlainText().trimmed());

    QTextStream line(q);

    line >> characterSymbol >> t >> testPattern.f[0] >> t >>  testPattern.f[1] >> t >>  testPattern.f[2] >> t >>  testPattern.f[3] >> t >>  testPattern.f[4] >> t >>  testPattern.f[5] >> t >>  testPattern.f[6] >> t >>  testPattern.f[7] >> t >>  testPattern.f[8] >> t >>  testPattern.f[9] >> t >>  testPattern.f[10] >> t >>  testPattern.f[11] >> t >> testPattern.f[12] >> t >> testPattern.f[13] >> t >> testPattern.f[14] >> t >> testPattern.f[15];


    //---------------------------------
    // Assume uppercase letters only
    int symbolIndex;
    if ((characterSymbol >= 'A') && (characterSymbol <= 'Z'))
    {
        symbolIndex = characterSymbol - 'A';
        testPattern.symbol = static_cast<Symbol>(symbolIndex);
    }
    else
    {
        symbolIndex = OUTPUT_NEURONS - 1;
        testPattern.symbol = UNKNOWN;
    }

    // Set all outputs to zero
    for (int j=0; j < OUTPUT_NEURONS; j++)
    {
        testPattern.outputs[j] = 0;
    }

    // Set the output we want to 1
    testPattern.outputs[symbolIndex] = 1;


    //---------------------------------
    // Get classification results from the network
    classificationResults = bp->testNetwork(testPattern);

    // Get predicted index
    int predictedIndex = bp->action(classificationResults);

    ui->lcdNumber_A->display(classificationResults[0]);
    ui->lcdNumber_B->display(classificationResults[1]);
    ui->lcdNumber_C->display(classificationResults[2]);
    ui->lcdNumber_D->display(classificationResults[3]);
    ui->lcdNumber_E->display(classificationResults[4]);
    ui->lcdNumber_F->display(classificationResults[5]);
    ui->lcdNumber_G->display(classificationResults[6]);
    ui->lcdNumber_H->display(classificationResults[7]);
    ui->lcdNumber_I->display(classificationResults[8]);
    ui->lcdNumber_J->display(classificationResults[9]);
    ui->lcdNumber_K->display(classificationResults[10]);
    ui->lcdNumber_L->display(classificationResults[11]);
    ui->lcdNumber_M->display(classificationResults[12]);
    ui->lcdNumber_N->display(classificationResults[13]);
    ui->lcdNumber_O->display(classificationResults[14]);
    ui->lcdNumber_P->display(classificationResults[15]);
    ui->lcdNumber_Q->display(classificationResults[16]);
    ui->lcdNumber_R->display(classificationResults[17]);
    ui->lcdNumber_S->display(classificationResults[18]);
    ui->lcdNumber_T->display(classificationResults[19]);
    ui->lcdNumber_U->display(classificationResults[20]);
    ui->lcdNumber_V->display(classificationResults[21]);
    ui->lcdNumber_W->display(classificationResults[22]);
    ui->lcdNumber_X->display(classificationResults[23]);
    ui->lcdNumber_Y->display(classificationResults[24]);
    ui->lcdNumber_Z->display(classificationResults[25]);


    //continue here...
    //...
    // Update confusion matrix (actual vs predicted)
    confusionMatrix[symbolIndex][predictedIndex]++;

    //-----------------------------------------------------------
    for(int k=0; k < OUTPUT_NEURONS; k++){
       outputs[k] = testPattern.outputs[k];
    }
    //-----------------------------------------------------------



     qDebug() << "actual output = " << bp->action(classificationResults) << ", desired output = " << bp->action(outputs) ;

     QString desiredClass = bp->generateLetter(bp->action(outputs));
     QString actualOutputClass = bp->generateLetter(bp->action(classificationResults));


    if ( predictedIndex== bp->action(outputs)) {
        qDebug() << "correct classification.";
        ui->label_Classification->setText("Correct! desired class = " + desiredClass + ", actual output class = " + actualOutputClass);


    } else {
        qDebug() << "incorrect classification.";
        ui->label_Classification->setText("Wrong. desired class = " + desiredClass + ", actual output class = " + actualOutputClass);
    }



    // delete[] classificationResults;
    // delete[] outputs;
}

// void MainWindow::on_pushButton_Train_Network_Max_Epochs_clicked()
// {
//     double SSE = 0.0;
//     QString msg;
//     bool checkbox_L2 = ui->checkBox_L2Regularization->isChecked();
//     int batchsize = ui->horizScrollBar_BatchSize->value();
//     QString dataset="complete";

//     if(!patternsLoadedFromFile) {
//         msg.clear();
//         msg.append("\nMissing training patterns.  Load data set first.\n");
//         ui->plainTextEdit_results->setPlainText(msg);
//         return;
//     }

//     MAX_EPOCHS = ui->spinBox_Max_training_Epochs->value();
//     QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
//     int e=0;
//     for(int i=0; i < MAX_EPOCHS; i++){
//       msg.clear();
//       msg.append("\nTraining in progress...\n");

//       SSE = bp->trainNetwork(checkbox_L2,batchsize,dataset); //trains for 1 epoch
//       ui->lcdNumber_SSE_Train->display(SSE);

//       qApp->processEvents();

//       update();
//       e++;
//      qDebug() << "epoch: " << e << ", SSE = " << SSE;
//       msg.append("\nEpoch=");
//       msg.append(QString::number(e));
//       ui->plainTextEdit_results->setPlainText(msg);

//       if((i > 0) && ((i % 100) == 0)) {
//           qDebug() << "i: " << i << "saveWeights ";
//          bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());

//          ui->plainTextEdit_results->setPlainText("Weights saved into file.");
//          qApp->processEvents();
//       }

//     }
//     QApplication::restoreOverrideCursor();

// }

void MainWindow::on_pushButton_Initialise_Network_clicked()
{
    bp->initialise();
}

void MainWindow::on_pushButton_Test_All_Patterns_clicked()
{
    char characterSymbol;

    double* classificationResults;
    double* outputs;
    int correctClassifications=0;

    classificationResults = new double[OUTPUT_NEURONS];
    outputs = new double[OUTPUT_NEURONS];


    for(int i=NUMBER_OF_TRAINING_PATTERNS; i < NUMBER_OF_PATTERNS; i++){

            characterSymbol = letters[i].symbol;
            for(int j=0; j < INPUT_NEURONS; j++){
                testPattern.f[j] = letters[i].f[j];
            }

            //This part could be implemented more concisely
            if(characterSymbol == LETTER_A){
                testPattern.symbol = LETTER_A;
                testPattern.outputs[0] = 1;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 0;

            } else if(characterSymbol == LETTER_A){
                testPattern.symbol = LETTER_B;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 1;
                testPattern.outputs[2] = 0;

            } else {
                testPattern.symbol = UNKNOWN;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 1;
            }

            //---------------------------------


            classificationResults = bp->testNetwork(testPattern);

            for(int k=0; k < OUTPUT_NEURONS; k++){
               outputs[k] = testPattern.outputs[k];
            }

            if (bp->action(classificationResults) == bp->action(outputs)) {
                 correctClassifications++;
            }

        }


      qDebug() << "TEST SET: correctClassifications = " << correctClassifications;

      QString msg;

      msg.clear();
      QTextStream(&msg) << "TEST SET: correctClassifications = " << correctClassifications << "\n";


      ui->plainTextEdit_results->setPlainText(msg);

      double pgc = (double(correctClassifications)/4000.0)*100.0;
      ui->lcdNumber_percentageOfGoodClassification_Test->display(pgc);

      qDebug() << "TEST SET pgc = " << pgc << Qt::endl;


      qApp->processEvents();
      update();

}

void MainWindow::on_pushButton_Save_Weights_clicked()
{
    bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());

    QString msg;
    QString lineOfText;

    lineOfText = "weights saved to file: " + ui->plainTextEdit_saveWeightsAs->toPlainText();

    msg.append(lineOfText);

    ui->plainTextEdit_results->setPlainText(msg);

}

void MainWindow::on_pushButton_Load_Weights_clicked()
{
   bp->loadWeights(ui->plainTextEdit_fileNameLoadWeights->toPlainText());

   QString msg;

   msg.append("weights loaded.\n");

   ui->plainTextEdit_results->setPlainText(msg);


}

void MainWindow::on_pushButton_testNetOnTrainingSet_clicked()
{

    char characterSymbol;

    double* classificationResults;
    double* outputs;
    int correctClassifications=0;

    outputs = new double[OUTPUT_NEURONS];

    for(int i=0; i < NUMBER_OF_TRAINING_PATTERNS; i++){

            characterSymbol = letters[i].symbol;
            for(int j=0; j < INPUT_NEURONS; j++){
                testPattern.f[j] = letters[i].f[j];
            }

            //This part could be implemented more concisely

            if(characterSymbol == LETTER_A){
                testPattern.symbol = LETTER_A;
                testPattern.outputs[0] = 1;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 0;

            } else if(characterSymbol == LETTER_B){
                testPattern.symbol = LETTER_B;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 1;
                testPattern.outputs[2] = 0;

            } else {
                testPattern.symbol = UNKNOWN;
                testPattern.outputs[0] = 0;
                testPattern.outputs[1] = 0;
                testPattern.outputs[2] = 1;
            }

            //---------------------------------
            classificationResults = bp->testNetwork(testPattern);

            for(int k=0; k < OUTPUT_NEURONS; k++){
               outputs[k] = testPattern.outputs[k];
            }

            if (bp->action(classificationResults) == bp->action(outputs)) {
                 correctClassifications++;
            }

     }
     qDebug() << "TRAINING SET: correctClassifications = " << correctClassifications;
     QString msg;

     msg.clear();
     QTextStream(&msg) << "TRAINING SET: correctClassifications = " << correctClassifications << "\n";


     ui->plainTextEdit_results->setPlainText(msg);

     double pgc = (double(correctClassifications)/16000.0)*100.0;
     ui->lcdNumber_percentageOfGoodClassification->display(pgc);
     qDebug() << "TRAINING SET pgc = " << pgc << Qt::endl;


     qApp->processEvents();
     update();

}

void MainWindow::on_horizScrollBar_LearningRate_actionTriggered(int action)
{
   if(action == 0){

       // int value = ui->horizScrollBar_LearningRate->value();

       // double learningRate = value / 1000.0;

       // ui->lcdNumber_LearningRate->display(learningRate);

       // LEARNING_RATE = learningRate;

       // qDebug() << "Learning rate adjusted to: " << LEARNING_RATE;
   }
}

void MainWindow::on_pushButton_ShuffleAllData_clicked()
{
    QString fileName = ui->lineEdit_fileNameLoadDataSets->text().trimmed();
    if (fileName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a file name.");
        return;
    }

    QString shuffledFileName;
    bool success = bp->shuffleDataset(fileName, shuffledFileName);

    if (success) {
        QMessageBox::information(this, "Success", "Data shuffled and saved to: " + shuffledFileName);
    } else {
        QMessageBox::warning(this, "Error", "Failed to shuffle data.");
    }
}


void MainWindow::on_pushButton_ShuffleTrainData_clicked()
{
    QString fileName = ui->lineEdit_fileNameLoadDataSets->text().trimmed();
    if (fileName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a file name.");
        return;
    }

    QString shuffledFileName;
    bool success = bp->shuffleDataset(fileName, shuffledFileName);

    if (success) {
        QMessageBox::information(this, "Success", "Data shuffled and saved to: " + shuffledFileName);
    } else {
        QMessageBox::warning(this, "Error", "Failed to shuffle data.");
    }
}


void MainWindow::on_pushButton_ShuffleTestData_clicked()
{
    QString fileName = ui->lineEdit_fileNameLoadDataSets->text().trimmed();
    if (fileName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a file name.");
        return;
    }

    QString shuffledFileName;
    bool success = bp->shuffleDataset(fileName, shuffledFileName);

    if (success) {
        QMessageBox::information(this, "Success", "Data shuffled and saved to: " + shuffledFileName);
    } else {
        QMessageBox::warning(this, "Error", "Failed to shuffle data.");
    }
}


void MainWindow::on_checkBox_L2Regularization_toggled(bool checked)
{

    qDebug() << "L2 Regularization Enabled ?"<<checked;
    ui->lcdNumber_L2Regularization->setEnabled(checked);       // Enable the LCD display
    ui->horizScrollBar_L2Regularization->setEnabled(checked);   // Enable/Disable a horizScrollBar


}


void MainWindow::on_horizScrollBar_BatchSize_valueChanged(int value)
{
    ui->lcdNumber_BatchSize->setSegmentStyle(QLCDNumber::Filled);
    ui->lcdNumber_BatchSize->display(value);
}


void MainWindow::on_horizScrollBar_L2Regularization_valueChanged(int value)
{
    ui->lcdNumber_L2Regularization->setSegmentStyle(QLCDNumber::Filled);
    ui->lcdNumber_L2Regularization->display(value/1000.0);
    L2_LAMBDA = value/1000.0;
}



void MainWindow::on_pushButton_Clearlog_clicked()
{
    QString logFileName = ui->plainTextEdit_fileNameLog->toPlainText().trimmed();

    if(logFileName.isNull()){
         QMessageBox::warning(this, "Warning", "Please enter a log file name.");
         return;
    }

    bp->clearlogfiles(logFileName);

    ui->plainTextEdit_results->setPlainText("Log file has been cleared.");
}


void MainWindow::on_pushButton_Train_Network_traindata_clicked()
{
    double SSE = 0.0;
    double PGC = 0.0;
    QString dataset = "train";
    QString ac;
    QString msg;

    // Early Stopping parameters
    int patience = 30;  // Set patience parameter
    int patience_counter = 0;  // Counter for tracking epochs without improvement
    double best_SSE = std::numeric_limits<double>::max();  // Initialize best SSE as the largest possible value
    bool early_stop = false;  // Early stopping flag

    // LEARNING_RATE = 0.0005;
    // L2_LAMBDA = 0.001;

    if(activation_Function ==0 ){
        ac = "ReLU";
    }else if(activation_Function == 1){
        ac = "Tanh";
    }

    if(!patternsLoadedFromFile) {
        msg.clear();
        msg.append("\nMissing training patterns.  Load data set first.\n");
        ui->plainTextEdit_results->setPlainText(msg);
        return;
    }

    MAX_EPOCHS = ui->spinBox_Max_training_Epochs->value();
    double maxPGC = ui->spinBox_Max_training_PGC->value();  // Get the Max Training PGC from the screen
    bool checkbox_L2= ui->checkBox_L2Regularization->isChecked();
    int batchsize = ui->horizScrollBar_BatchSize->value();

    if(batchsize>0){
        BATCH_SIZE = batchsize;
    }

    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    int e=0;
    QString logFileName = "log.csv";

    for(int i=0; i < MAX_EPOCHS; i++){

        // Early stopping check: if early_stop is true, break out of the loop
        if (early_stop) {
            break;
        }
        msg.clear();
        msg.append("\nTraining in progress(train data set)...\n");

        SSE = bp->trainNetwork(checkbox_L2,BATCH_SIZE,dataset); //trains for 1 epoch
        ui->lcdNumber_SSE_Train->display(SSE);

        // Calculate the current PGC (percentage of correct classification)
        PGC = bp->calculatePGC(dataset);
        ui->lcdNumber_percentageOfGoodClassification->display(PGC);

        // save log file
        bp->saveLogs(logFileName, e, SSE, LEARNING_RATE, L2_LAMBDA, dataset, ac, PGC);

        qApp->processEvents();

        update();
        e++;
        qDebug() << "epoch: " << e << ", SSE = " << SSE<<", Learning rate = "<<LEARNING_RATE<<", L2_LAMBDA = "<<L2_LAMBDA<<", dataset = "<<dataset<< ", Activation = "<<ac<<", PGC = "<<PGC;
        msg.append("\nEpoch=");
        msg.append(QString::number(e));
        ui->plainTextEdit_results->setPlainText(msg);

        // PGC Early Stopping Check
        if (PGC >= maxPGC) {
            early_stop = true;
            qDebug() << "Early stopping due to reaching Max PGC: " << PGC;
            msg.append("\nEarly stopping due to Max PGC reached. Training stopped.");
            ui->plainTextEdit_results->setPlainText(msg);
            break;
        }

        // Early Stopping check
        if (SSE < best_SSE) {
            best_SSE = SSE;  // Update Best SSE
            patience_counter = 0;  // Reset Patience Counter
            bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());  // Save current weights
            qDebug() << "Best SSE improved, saving weights at epoch: " << e;
        } else {
            patience_counter++;  // Increase patience counter
        }

        // If the patience counter exceeds the set value, stop training
        if (patience_counter >= patience) {
            early_stop = true;
            qDebug() << "Early stopping at epoch: " << e;
            msg.append("\nEarly stopping triggered. Training stopped.");
            ui->plainTextEdit_results->setPlainText(msg);
        }



        if((i > 0) && ((i % 100) == 0)) {
            qDebug() << "i: " << i << "saveWeights ";
            bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());

            ui->plainTextEdit_results->setPlainText("Weights saved into file.");
            qApp->processEvents();
        }

    }
    QApplication::restoreOverrideCursor();
}


void MainWindow::on_pushButton_Train_Network_testdata_clicked()
{
    double SSE = 0.0;
    double PGC = 0.0;
    QString dataset = "test";
    QString ac;
    QString msg;

    // Early Stopping parameters
    int patience = 30;  // Set patience parameter
    int patience_counter = 0;  // Counter for tracking epochs without improvement
    double best_SSE = std::numeric_limits<double>::max();  // Initialize best SSE as the largest possible value
    bool early_stop = false;  // Early stopping flag

    if(activation_Function ==0 ){
        ac = "ReLU";
    }else if(activation_Function == 1){
        ac = "Tanh";
    }


    if(!patternsLoadedFromFile) {
        msg.clear();
        msg.append("\nMissing training patterns.  Load data set first.\n");
        ui->plainTextEdit_results->setPlainText(msg);
        return;
    }

    MAX_EPOCHS = ui->spinBox_Max_training_Epochs->value();
    double maxPGC = ui->spinBox_Max_training_PGC->value();  // Get the Max Training PGC from the screen
    bool checkbox_L2= ui->checkBox_L2Regularization->isChecked();
    int batchsize = ui->horizScrollBar_BatchSize->value();
    if(batchsize>0){
        BATCH_SIZE = batchsize;
    }

    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    int e=0;
    QString logFileName = "log.csv";

    for(int i=0; i < MAX_EPOCHS; i++){

        // Early stopping check: if early_stop is true, break out of the loop
        if (early_stop) {
            break;
        }

        msg.clear();
        msg.append("\nTraining in progress(test data set)...\n");

        SSE = bp->trainNetwork(checkbox_L2,BATCH_SIZE,dataset); //trains for 1 epoch
        ui->lcdNumber_SSE_Test->display(SSE);

        // Calculate the current PGC (percentage of correct classification)
        PGC = bp->calculatePGC(dataset);
        ui->lcdNumber_percentageOfGoodClassification->display(PGC);

        // save log file
        bp->saveLogs(logFileName, e, SSE, LEARNING_RATE, L2_LAMBDA,dataset, ac, PGC);

        // PGC Early Stopping Check
        if (PGC >= maxPGC) {
            early_stop = true;
            qDebug() << "Early stopping due to reaching Max PGC: " << PGC;
            msg.append("\nEarly stopping due to Max PGC reached. Training stopped.");
            ui->plainTextEdit_results->setPlainText(msg);
            break;
        }

        // Early Stopping check
        if (SSE < best_SSE) {
            best_SSE = SSE;  // Update Best SSE
            patience_counter = 0;  // Reset Patience Counter
            bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());  // Save current weights
            qDebug() << "Best SSE improved, saving weights at epoch: " << e;
        } else {
            patience_counter++;  // Increase patience counter
        }

        // If the patience counter exceeds the set value, stop training
        if (patience_counter >= patience) {
            early_stop = true;
            qDebug() << "Early stopping at epoch: " << e;
            msg.append("\nEarly stopping triggered. Training stopped.");
            ui->plainTextEdit_results->setPlainText(msg);
        }

        qApp->processEvents();

        update();
        e++;
        qDebug() << "epoch: " << e << ", SSE = " << SSE<<", Learning rate = "<<LEARNING_RATE<<", L2_LAMBDA = "<<L2_LAMBDA<<", dataset = "<<dataset<< ", Activation = "<<ac<<", PGC = "<<PGC;
        msg.append("\nEpoch=");
        msg.append(QString::number(e));
        ui->plainTextEdit_results->setPlainText(msg);

        if((i > 0) && ((i % 100) == 0)) {
            qDebug() << "i: " << i << "saveWeights ";
            bp->saveWeights(ui->plainTextEdit_saveWeightsAs->toPlainText());

            ui->plainTextEdit_results->setPlainText("Weights saved into file.");
            qApp->processEvents();
        }

    }
    QApplication::restoreOverrideCursor();
}


void MainWindow::on_comboBox_ActivationFunction_currentIndexChanged(int index)
{
    activation_Function = index;
}


void MainWindow::on_pushButton_SaveTestDataConfusionMatrix_clicked()
{

    QString fileName = ui->lineEdit_confusionMatrix_filename->text().trimmed();
    if (fileName.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please enter a file name with .csv .");
        return;
    }

    bool result = bp->saveConfusionMatrixToCSV(fileName);

    if (result) {
        QMessageBox::information(this, "Success", "Confusion matrix data saved to: " + fileName);
    } else {
        QMessageBox::warning(this, "Error", "Failed to save confusion matrix data.");
    }

}

