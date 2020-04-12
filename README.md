Repository URL: https://github.com/sbouab/comp472_project2
<br>
<br>
<b>NOTE:</b> <i>project2.py</i> contains the project code for the required model implementation while <i>project2-byom.py</i> and <i>project2-byom2.py</i> contain the BYOM (custom) implementations. The first BYOM is similar to the required one, but with a few tweaks, while the second BYOM uses another dataset to train the classifier and more tweaks in order to try to improve the language identification method (unsuccessfully so).
<br>
<br>
Possible hyper-parameters:
<br>
V=[0,1,2]
<br>
n=[1,2,3]
<br>
d=[0.0 ... 1.0]
<br>
<br>
How to run the models (considering Python3.7 and all necessary libraries are correctly installed on the machine):
<br>
To run the required model, provide the desired hyper-parameters, training and testing files like:
<br>
<i>python path/to/project2.py V-value n-value d-value path/to/training-file-name.txt path/to/testing-file-name.txt</i>
<br>
<br>
First BYOM runs in a similar fashion as the required model:
<br>
<i>python path/to/project2-byom.py V-value n-value d-value path/to/training-file-name.txt path/to/testing-file-name.txt</i>
<br>
<br>
To run the second BYOM, provide the testing file and make sure the training files (eu.txt, ca.txt, en.txt, gl.txt, pt.txt and es.txt) are in the same directory as the program:
<br>
<i>python path/to/project2-byom2.py path/to/testing-file-name.txt</i>
<br>
<br>
For more information regarding the project, please refer to the <i>COMP_472_2020_Winter_Project_2-SchoolClosing.pdf</i>
