This is the source code for the paper titled "Using Pre-trained Deep Learning Models as Feature Extractors in the Bag-of-Deep-Visual-Words Model: A Consistent Accuracy Improvement?".

1. Set the root directory path for the dataset in the settings.py file, for example: data_path = r'/home/x/Desktop/code/data'.
2. Unzip the datasets into the datasets folder, with each dataset in a separate folder. The folder names should be 15-Scenes, Caltech-101, COVID-19, MIT Indoor-67, NWPU, TF-Flowers.
3. All datasets can be downloaded from this URL: https://figshare.com/collections/Datasets/6756057/1.
4. After completing the above configuration, you can directly run the Python files in the main/tasks folder. The files are:
        DL.py: Classify using deep learning models.
        FV.py: Classify using the BoDVW (fisher vector encoding) model.
        HV.py: Classify using the BoDVW (hard-voting) model.
        SV.py: Classify using the BoDVW (soft-voting) model.
        LLC.py: Classify using the BoDVW (local-constraint linear coding) model.
        SVC.py: Classify using the BoDVW (super vector coding) model.
5. Run the print_results.py file to print the results. You need to specify the path of the .pkl file where the results are saved. By default, all results are stored in the data/intermediate_data/obtain_cls_scores/ folder.
