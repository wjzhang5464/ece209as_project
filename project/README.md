###Data downloading
The download link for our dataset is here. The 'dataset' folder contains the training data we used and 'our_results' contains our trained model weights. After downloading, please put both the 'dataset folder and 'our_results' folder under the Multimodal folder.

### Preprocess
Data preprocessing is performed by running preprocess.ipynb. You don't need to run this file if you want to run our results directly.

### Training
The training code is included in train.ipynb. If you want to run your own data, please change all --out_dir parameters to the folder where your data is located, including the parameters in evaluation. The generated weight file will be placed in the 'model' folder in the 'dataset' folder. Each time a new weight file is generated, the results of the old epoch will be deleted.

### Evaluation
If you want to test our experimental results, you can run evaluation.ipynb directly. Finally it outputs a boxplot of the true and false video scores and the AUC scores classified using thresholds and nearby values.


### Acknowledgements
Thanks to the code available at https://github.com/cs-giung/face-detection-pytorch, https://github.com/TengdaHan/DPC and https://github.com/joonson/syncnet_python.