Example Datasets
=====

We utilize the TCGA BRCA micro RNA dataset. We use cancer subtypes (ILC, IDC) as group labels and marker filtering based on importance score from a random forest classification. 
The dataset is then split into training and test dataset, where the test dataset has 100 samples per group, and the rest samples are called training data set. 
`SyNG-BTS <https://github.com/LXQin/SyNG-BTS>`_ is applied to the training dataset with model CVAE1-20 and epoch 285 and batch size 10% of the input size.
Each group is augmented into 500 samples and used to fit the IPLF. In comparison, the test samples can also be used to fit the IPLF.  
The comparison between the two fitted IPLF can help check the quality of the augmented samples. 