# WeaklySupervisedActiveLearning-for-NILM

This code was designed for experiments of the following publication:
Tanoni, G., Sobot, T., Principi, E., Stankovic, V., Stankovic, L. and Squartini, S., 2023. A Weakly Supervised Active Learning Framework for Non-Intrusive Load Monitoring. Integrated Computer Aided Engineering, (vol), (pp), (doi)

If you use this code, please cite the paper above.

To use provided CRNN (or another) model inside the active learning framework, model and data generator creation should be set appropriately in AL_main.py. ALso, training and testing procedures should be set appropriately in AL_loop.py.

Data chould be structured in a directory as follows: <br />
 --(data_dir)/ <br />
&emsp;--agg/ <br />
&emsp;&emsp;--house_(# of house)/ <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--... <br />
&emsp;&emsp;--house_(# of house)/ <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--... <br />
&emsp;--labels/ <br />
&emsp;&emsp;--house_(# of house)/ <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--... <br />
&emsp;&emsp;--house_(# of house)/ <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--(filename).npy <br />
&emsp;&emsp;&emsp;--... <br />
