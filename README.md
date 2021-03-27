# test
Code structure:
Lines: 59-61 : address of the files: network.txt (electrical net connection of modules), symmetry.txt ( information about symmetry modules) ,  and finnf.txt ( number of fins and number of fingers and number of dummy dates for each transistor)
Lines 75 – 172: functions:
Dim: calculates the dimensions of transistors according to the information in finnf.txt file
Tot_wire_l : calculates the total wire length according to the information of network.txt file
Tot_overlap : calculates the total overlap of modules according to the current coordinates of modules on the chip
Area: calculates the total area of the chip
Sym_function: calculates the asymmetry of symmetry pairs
Tot_cost: estimates the total cost function
Lines 174 -346: Describes the environment which contains the information about coordinates and dimensions of modules. The step function (line 297) executes the action (change the coordinates of the modules) and generates the reward according to the output value of the Tot_cost function. The get_image function (line 346) generates a picture of the layout (as a representative of the current state of the modules) and feeds it to the agent. 
Lines 458- 567: Describes the Agent. It consists of the Model (DNN) and the train function (line 506) to train the DNN according to the reward it receives from the environment. 
Tall1.py :
This file is used for training the model to reach the cost function of Thr_cost ( line 33). The model is stored in trainmodel.ckpt file ( line 393).
Tall2.py:
in file we use the trained model (line 463: self.model.load_weights('trainallSL2.ckpt') ) to reach different threshold values. The for loop in lines (586 – 679) reduces the threshold for cost value. In each episode the state is reset to initial values. The maximum number of steps in each episodes to search for the solution (the floorplane with the cost function less than the threshold value) is defined by the EPI_STEP ( line 33). Lines 681-682 print the threshold values and the number of steps to reach the corresponding threshold values.


