# Multimodal Sarcasm Detection Based on Multimodal Sentiment Co-training

Sarcasm detection is a difficult task in sentiment analysis because sarcasm often includes both positive and negative sentiments, making it difficult to identify. In recent years, visual information has been used to study sarcasm in social media data. Based on sentiment contrast in image and text, this paper proposes a multimodal sentiment and sarcasm classification gradient co-training (MSSGC) model to explicitly learn image and text sentimental features from image and text sentiment datasets.

# Usage

- ##### The  config parameters setting during model training and testing

  ```yaml
  epoch: 10         					## the epoches of training
  batch: 16         					## the batch sizes of training
  positive_se : 0.2 					## the positive sentence in Sp(only use in Sp-SPC)
  lr: 1e-5      					    ## the learning rate of Amda
  UNCASED: "../../transformers/bert-base-uncased"   ## bert local path
  VOCAB: 'vocab.txt'
  path: "2022-4-12"            ##save model checkpoint,log,result dictory
  start_dir: "../data/SemEval2017-Task1/train_data" ## The  dictory of train dataset
    
  ```

- Training the MTL model

  ```
  python mtl.py
  ```

- Traning the MSSGC model

  ```
  python main.py
  ```

# Data

* Multimodal sarcasm dataset: raw data downloaded from: [link](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)
* SemEval-2017 task4: raw data downloaded from: [link](https://alt.qcri.org/semeval2017/)
* Image sentiment data from tweets: collected by the author [Quanzeng You](https://qzyou.github.io/)


# Requirement

- Torch >= 1.10.0
- Torchvision >= 0.11.1
- Transformers >= 4.14.1
- numpy >= 1.21.4
- tqdm >= 4.62.3

