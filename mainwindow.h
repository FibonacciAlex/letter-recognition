#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
//-----------------------
//file manipulation
#include <QFile>
#include <QTextStream>
#include <QDataStream>
#include <QMessageBox>
//-----------------------
#include "backpropagation.h"


namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_Read_File_clicked();

    void on_horizScrollBar_LearningRate_valueChanged(int value);

    void on_pushButton_Classify_Test_Pattern_clicked();

    // void on_pushButton_Train_Network_Max_Epochs_clicked();

    void on_pushButton_Initialise_Network_clicked();

    void on_pushButton_Test_All_Patterns_clicked();

    void on_pushButton_Save_Weights_clicked();

    void on_pushButton_Load_Weights_clicked();

    void on_pushButton_testNetOnTrainingSet_clicked();

    void on_horizScrollBar_LearningRate_actionTriggered(int action);

    void on_pushButton_ShuffleAllData_clicked();

    void on_pushButton_ShuffleTrainData_clicked();

    void on_pushButton_ShuffleTestData_clicked();

    void on_checkBox_L2Regularization_toggled(bool checked);

    void on_horizScrollBar_BatchSize_valueChanged(int value);

    void on_horizScrollBar_L2Regularization_valueChanged(int value);

    void on_pushButton_Clearlog_clicked();

    void on_pushButton_Train_Network_traindata_clicked();

    void on_pushButton_Train_Network_testdata_clicked();

    void on_comboBox_ActivationFunction_currentIndexChanged(int index);

    void on_pushButton_SaveTestDataConfusionMatrix_clicked();

private:
    Ui::MainWindow *ui;
    //-------------------------
    Backpropagation *bp;
};

#endif // MAINWINDOW_H
