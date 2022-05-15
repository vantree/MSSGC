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
3. Image sentiment data from tweets: collected by the author [Quanzeng You](https://qzyou.github.io/)


# Pretrained Models

Download pretrained BERT-Base from [here](https://huggingface.co/bert-base-uncased/tree/main) and put it in [this directory](resources/transformers).

Download pretrained ResNet-152 from [here](https://download.pytorch.org/models/resnet152-394f9c45.pth), rename the binary file as "resnet152.pth" and put it in [this directory](resources/resnet).


# Requirement

- Torch >= 1.10.0
- Torchvision >= 0.11.1
- Transformers >= 4.14.1
- numpy >= 1.21.4
- tqdm >= 4.62.3

