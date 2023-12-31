# Poster2Rating
[overleaf latex paper](https://www.overleaf.com/9544989412kszmvvmqvgdp#323e03)
[presentation slide deck](https://docs.google.com/presentation/d/14YhTZdl2Qq6Ugt_u-0UglIgb3IvwMWfsI1KlNW0ZvKs/edit?usp=sharing)

## 10/31/23 Dataloading Update

Got data from this kaggle link: https://www.kaggle.com/datasets/getaolga/moviepostersimdb?select=MovieGenre.csv

If for whatever reason, you want to download the data for yourself, you'll probably have to re-save the csv as a csv with UTF-8 encoding (I used VScode to do this).

Under the notebooks folder is where I did all my work. 

First, we have 'Clean Data.ipynb' which just cleans the data by getting rid of NA's. You can find the final "data" folder in data.zip to download in the google drive folder: https://drive.google.com/drive/folders/16M8kNRldFHNQaMcgfPebCLMAdMK01meY?usp=sharing

Extract it into the root directory.

Next, we have 'Image Loader.ipynb' which gives a custom dataset. Along with the csv file path and image folder path, it takes in a list of genres, so you can filter your dataset by genre(s). This dataset class works with pytorch dataloaders. Example code on how to put the data into a dataloader is in the notebook. Note that sometimes in the notebook, you'll have to rerun every cell if you run into an error after changing a parameter or something. I'm not sure why that happens.

In terms of random other stuff, I set up a gitignore to ignore csv files, the data folder, and random other unnecessary jupyter notebook related files. There's also a requirements.txt which just for now has torch. torchvision, and pandas.
