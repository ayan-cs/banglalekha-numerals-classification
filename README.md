# BanglaLekha Handwritten Digits Classification

### About the dataset

The ![Original Dataset](https://data.mendeley.com/datasets/hf6sf8zrkc/2) contains Handwritten Bangla Characters and Digits. Here, only the Handwritten Digits have been taken into consideration. The dataset used here is available ![here](https://www.kaggle.com/datasets/ipythonx/banglalekhaisolatednumerals).

### Instructions for Training the model

- Open CMD/Terminal and clone the repository using the command : `git clone git@github.com:ayan-cs/banglalekha-numerals-classification`
- Download the BanglaLekha Numerals dataset from the given link above and extract inside the repository folder. It is recommended not to make any change to the dataset folder.
- Preprocess the data by executing the `Data_Preparation.ipynb` notebook. This should create a folder **Preprocessed_Dataset** containing *Train* and *Validation* splits.
- Configure `train_config.yaml` file.
- Run the script on CMD/Terminal : `python main.py train`
- The trained model will be available inside **Checkpoints** folder and the plots will be saved inside **Plots & Outputs** folder.

### Instructions for Inference/Prediction

- Open CMD/Terminal and clone the repository using the command : `git clone git@github.com:ayan-cs/banglalekha-numerals-classification`
- Make sure the data is preprocessed.
- Configure `inference_config.yaml` file. For demo, one trained ResNet-34 model has been provided.
- Open CMD/Terminal, run the command : `python main.py inference`
- The outputs will be available inside the **Plots & Outputs** folder.

### Let's Connect

![My Portfolio](https://sites.google.com/view/ayanabha)
![LinkedIn](https://www.linkedin.com/in/ayanabha-ghosh-cs)