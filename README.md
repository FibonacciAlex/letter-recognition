# Letter-Recognition
This small project are used for practice the Letter Recognition using Deep Neural  Nets with Softmax Units. 
## Key Points
- Implement backpropagation learning algorithm for a deep network classifier system.
- Consider different weight-update formula variations,  hyperparameter settings, optimization strategies to train and get the  best network configuration.
- Use QT Framework to build the GUI.

## Basic Technical Requirements
- C++ (Entry level)
- QT Framework (Entry Level)
- Neural Network & Deep Learning

## How to run the code
As I still don't figure out how to use `CMake` to build the project, so you have to use the .pro file to rebuild the project and then run qmake to compile and the exe file.

### Steps
1. Download and install QT 6.x desktop development.
2. Delete any file with the extension `.pro.user`.
3. Open the Qt Creator and rerun `qmake`. Go to the left Project panel, find and select the target project name, right clike and select `Run qmake`.
4. Copy the dataset file(name:xxx_data_set.txt) into the build folder(name:build-xxx-Desktop_xxx-Debug) to make it accessible to the program.

Note: A very long directory path name may cause some problem, keep it short.
