# Multimodal Sarcasm Detection Based on Multimodal Sentiment Co-training

Sarcasm detection is a difficult task in sentiment analysis because sarcasm often includes both positive and negative sentiments, making it difficult to identify. In recent years, visual information has been used to study sarcasm in social media data. Based on sentiment contrast in image and text, this paper proposes a multimodal sentiment and sarcasm classification gradient co-training (MSSGC) model to explicitly learn image and text sentimental features from image and text sentiment datasets.

# Usage

- ##### The  config parameters setting during model training and testing

  ```yaml
  epoch: 10       ## the epochs of training
  bs: 16          ## the batch size of training
  lr: 1e-5        ## the learning rate of Adam
  update_linear_lr: 3e-4      ## update step \beta
  update_embedding_lr: 1e-5   ## update step \alpha
  senti_weight: 0.3           ## the weight of sentiment loss
  mlp_hidden_size: 2816       ## the hidden size of fusion module MLP
  mlp_hidden_layer: 2         ## the hidden layers of fusion module MLP
  fusion: 'concat'            ## the way of fusion text-image features for MSD samples
  freeze_bert: False          ## whether to update the parameters of BERT during training
  freeze_resnet: False        ## whether to update the parameters of ResNet during training
  ```

- Training the MTL model

  ```
  python mtl.py
  ```

- Traning the MSSGC model

  ```
  python main.py
  ```

# Pretrained Models

Download pretrained BERT-Base from [here](https://huggingface.co/bert-base-uncased/tree/main) and put it in [this directory](resource/transformers).

Download pretrained ResNet-152 from [here](https://download.pytorch.org/models/resnet152-394f9c45.pth), rename the binary file as "resnet152.pth" and put it in [this directory](resource/resnet).


# Data

1. Multimodal sarcasm dataset: raw data downloaded from: [link](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)
2. SemEval-2017 task4: raw data downloaded from: [link](https://alt.qcri.org/semeval2017/)
3. Image sentiment data from tweets: collected by the author [Quanzeng You](https://qzyou.github.io/)


# Requirement

- Torch >= 1.10.0
- Torchvision >= 0.11.1
- Transformers >= 4.14.1
- numpy >= 1.21.4
- tqdm >= 4.62.3

