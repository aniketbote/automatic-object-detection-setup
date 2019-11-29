**Pre-requiste:**
```
1. Git version control system
2. Tensorflow installed (GPU or CPU)
```
**Directions to use :**
1. Place the files into empty directory
2. cd into directory and do
    ```
    - pip install -r req.txt
    - python auto_obj_det.py
3. After labelimg software opens add the the datset into newly created train and test folder
    ```
    - train path : workspace/images/train
    - test path : workspace/images/test
4. Label the images using label img software and save the xml files in the same directory  

**Note : Carry on only if all the labelled dataset is put into respective directory.
       If prompt was closed when label img software popped up re-execute the script and follow instructions else delete all the contents of        folder and restart execution.**

5. Enter the no of objects you want to detect in prompt
6. Enter the name of the objects as given in label img software
7. Choose the model from the given stack
8. Execute the command shown on the prompt
