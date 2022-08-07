# Satellite Imagery - Road Extraction
I am learning how to perform road extraction from satellite images.

The original code is from https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge. In this repository, it is now updated to run on Python 3 and a newer PyTorch version.

The authors provided a link to their paper: http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf

The dataset is downloaded from https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset. We put the *train*, *test* and *valid* in the downloaded zip file into *dataset* folder.

The authors also provided a trained D-LinkNet34 model, which can be downloaded from their Dropbox at https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=0. To test the model, we place it in a *weights* folder and run the test.py script.

# Libraries
* python==3.9.12
* numpy==1.23.1
* opencv-python==4.5.4.60
* torch==1.12.0
