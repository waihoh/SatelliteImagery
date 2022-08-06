# SatelliteImagery

Learning road extraction from satellite images.

Code is based on https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge

The paper is http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf

Dataset is from https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset. Place the *train*, *test* and *valid* subfolders in *dataset* folder.

Trained D-LinkNet34 model provided by authors can be downloaded from their Dropbox at https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=0. Place the file in a *weights* folder.

# Requirements
* python==3.9.12
* numpy==1.23.1
* opencv-python==4.5.4.60
* torch==1.12.0