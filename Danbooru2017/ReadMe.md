[Danbooru 2017 database](https://www.gwern.net/Danbooru2017)

The ```[Danbooru2017] training image list_``` file contains  113k 512x512 image file names for training a CNN-LSTM classifier. You may download them by `rsync` with `-- files-from`

The ```113k_imgs_512tags_encoded.7z``` is a json file containing tags for 113k images. The ```sk-LabelEncoder_512tags.pk``` is used for one hot encoding from scikit-learn. 

If you need a complete list of tags, please open an issue. I might be able to provide it. For a reference, if you process the large meta file from Danbooru2017, you may need around 8G to unzip the file and around 40 minutes with 8 cores to run through all json files. Good luck. 