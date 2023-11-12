This is the source code for the paper titled "Can Using a Pre-trained Deep Learning Model as the Feature Extractor in the Bag-of-Deep-Visual-Words Model Always Improve Image Classification Accuracy?".

Dependencies:

joblib==1.1.1

matplotlib==3.7.0

numpy==1.23.5

opencv_python==4.7.0.72

Pillow==9.4.0

scikit_learn==1.2.1

scipy==1.10.0

torch==2.0.0

torchvision==0.15.1


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


In addition to our written documentation, we have also provided a comprehensive video tutorial that demonstrates how to set up the necessary environment, run the code, and view the results. This visual guide is designed to complement our written instructions, providing a step-by-step walkthrough that you can follow along at your own pace. You can access this video tutorial at https://zenodo.org/records/10115784
