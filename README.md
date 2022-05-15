# Multimodal Sarcasm Detection Based on Multimodal Sentiment Co-training

Sarcasm detection is a difficult task in sentiment analysis because sarcasm often includes both positive and negative sentiments, making it difficult to identify. In recent years, visual information has been used to study sarcasm in social media data. Based on sentiment contrast in image and text, this paper proposes a multimodal sentiment and sarcasm classification gradient co-training (MSSGC) model. The model uses text and image feature sharing networks to explicitly learn image and text sentimental features from image and text sentiment datasets and integrates a cross-modal fusion module for multimodal sarcasm detection. 

# Usage

- ##### The  config parameters setting during model training and testing

  ```yaml
  model_output: 2   					##two or three classification 
  epoch: 30         					##the epoches of training
  batch: 32         					## the batch sizes of training
  positive_se : 0.2 					## the positive sentence in Sp(only use in Sp-SPC)
  lr: 0.00001       					## the learning rate of Amda 
  clean_tag: 'neutral'        ## Two classification filter the neutral out
  class: ["negative","positive"]  ## Two classificatioin
  #clean_tag: ''        ## Three classification filter the neutral out
  #class: ["negative","positive","neutral"]  ## Three classificatioin
  UNCASED: "../../transformers/bertweet-base" ##bertweet local path
  VOCAB: 'vocab.txt'
  path: "2022-4-12"            ##save model checkpoint,log,result dictory
  start_dir: "../data/SemEval2017-Task1/train_data" ## The  dictory of train dataset
  dataset:
    senti_train: []
    senti_test: ["../data/SemEval2017-Task1/test_data/SemEval2017-task4-test.subtask-A.english.txt"]   
    sms_train: ["../data/SemEval2018-Task3/train_data/sar_train.txt"]
    sms_test: ["../data/SemEval2018-Task3/test_data/sar_test.txt"]
    
  ```

  

- ##### The procession of training and testing

  ```bash
  cd SPC && bash -x start.sh
  cd MLT && bash -x start.sh
  cd SPC_S && bash -x start.sh
  cd Sp_SPC && bash -x start.sh
  ```

- ##### The analysis of experiments in all model

  ```bash
  cd SPC/path(config file define) && python3 RA.py
  cd MLT/path(config file define) && python3 RA.py
  cd SPC_S/path(config file define) && python3 RA.py
  cd Sp_SPC/path(config file define) && python3 RA.py
  ```

- Weight $\alpha$ in the Sp Task  

  ```bash
  - python3 run.py
  - cd path/2(3)
  - python3 weighted.py
  ```

  

- Encoding Feature Visualization

  ```bash
  Generate_picture.ipynb
  ```

- Case Study

  ```
  python3 example.py
  ```

# Data

1. Multimodal sarcasm dataset: raw data downloaded from: [link](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)
2. SemEval-2017 task4: raw data downloaded from: [link](https://alt.qcri.org/semeval2017/)
3. Image sentiment data from tweets: raw data given by the author [Quanzeng You](https://qzyou.github.io/)

# Requirement

- Torch >= 1.10.0
- Torchvision >= 0.11.1
- Transformers >= 4.14.1
- numpy >= 1.21.4
- tqdm >= 4.62.3

