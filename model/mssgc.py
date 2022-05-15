from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from fusions.factory import factory
from transformers import PreTrainedModel, PreTrainedTokenizer

from data_text.dataset import MyDataPoint as TextDataPoint, MySentence as TextSentence, MyDataset as TextDataset
from data_image.dataset import MyDataPoint as ImageDataPoint, MyImage, MyDataset as ImageDataset
from data_mm.dataset import MyDataPoint as MMDataPoint, MyPair as MMPair, MyDataset as MMDataset


def _use_cache(module: nn.Module, data_points: List[MMDataPoint]):
    cached, freeze = True, True
    for data_point in data_points:
        cached = cached and data_point.embedding is not None
    for parameter in module.parameters():
        freeze = freeze and not parameter.requires_grad
    return cached and freeze


def _use_tsa_cache(module: nn.Module, data_points: List[TextDataPoint]):
    cached, freeze = True, True
    for data_point in data_points:
        cached = cached and data_point.embedding is not None
    for parameter in module.parameters():
        freeze = freeze and not parameter.requires_grad
    return cached and freeze


def _use_image_cache(module: nn.Module, data_points: List[ImageDataPoint]):
    cached, freeze = True, True
    for data_point in data_points:
        cached = cached and data_point.embedding is not None
    for parameter in module.parameters():
        freeze = freeze and not parameter.requires_grad
    return cached and freeze


class MTM(nn.Module):
    def __init__(
            self,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            sentence_embedding: PreTrainedModel,
            image_embedding: nn.Module,
            mlp_hidden_size: int,
            mlp_num_hidden_layer: int,
    ):
        super(MTM, self).__init__()
        self.drop_path_prob = 0.0

        self.tokenizer = tokenizer
        self.sentence_embedding = sentence_embedding
        self.sentence_embedding_length = sentence_embedding.config.hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_num_hidden_layer = mlp_num_hidden_layer

        self.image_embedding = image_embedding
        self.image_embedding_length = 2048

        self.fusion_type = 'cat_mlp'
        self.output_dim = 128
        self.sentence_fusion_length = 128
        self.image_fusion_length = 128

        self.dropout_text_linear = nn.Dropout(0.1)
        self.dropout_mm_linear = nn.Dropout(0.1)
        self.dropout_image_linear = nn.Dropout(0.1)
        self.linear_text = nn.Linear(self.sentence_embedding_length, self.sentence_fusion_length)
        self.linear_image = nn.Linear(self.image_embedding_length, self.image_fusion_length)

        self.classifier_text = nn.Linear(self.sentence_fusion_length, 2)
        self.classifier_mm = nn.Linear(self.output_dim, 2)
        self.classifier_image = nn.Linear(self.image_fusion_length, 2)

        self.fusion = factory({
            'type': self.fusion_type,
            'input_dims': [self.sentence_fusion_length, self.image_fusion_length],
            'output_dim': self.output_dim,
            'dimensions': [self.mlp_hidden_size] * self.mlp_num_hidden_layer
        })

        self.device = device
        self.to(device)

    def forward(self, pairs, second_forward: bool = False):
        if isinstance(pairs[0], MMPair):
            return self.forward_loss_mm(pairs, second_forward)
        elif isinstance(pairs[0], MyImage):
            return self.forward_loss_image(pairs, second_forward)
        else:
            return self.forward_loss_text(pairs)

    def _embed_mm_sentences(self, pairs: List[MMPair]):
        sentence_list = [pair.sentence for pair in pairs]
        if _use_cache(self.sentence_embedding, sentence_list):
            return

        bs = 3
        for i in range(0, len(sentence_list), bs):
            sentences = sentence_list[i:i + bs]
            texts = [sentence.text for sentence in sentences]
            inputs = self.tokenizer(
                texts,
                max_length=512, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)

            output = self.sentence_embedding(**inputs, return_dict=True)
            embeddings = output.last_hidden_state[:, 0, :]
            for sentence, embedding in zip(sentences, embeddings):
                sentence.embedding = embedding

    def _embed_mm_images(self, pairs: List[MMPair], second_forward: bool, embed_regions=False):
        images = [pair.image for pair in pairs]
        if _use_cache(self.image_embedding, images):
            return

        embeddings = torch.stack([image.data for image in images]).to(self.device)
        embeddings = self.image_embedding.embed(embeddings) if not embed_regions \
            else self.image_embedding.embed_regions(embeddings)
        for image, embedding in zip(images, embeddings):
            image.embedding = embedding
            if second_forward is True:
                image.data = None

    def _fuse_mm_text(self, pairs: List[MMPair]):
        self._embed_mm_sentences(pairs)
        return torch.stack([pair.sentence.embedding for pair in pairs])

    def _fuse_mm_image(self, pairs: List[MMPair], second_forward: bool):
        self._embed_mm_images(pairs, second_forward)
        return torch.stack([pair.image.embedding for pair in pairs])

    def _fuse_mm_concat(self, pairs: List[MMPair], second_forward: bool):
        self._embed_mm_sentences(pairs)
        self._embed_mm_images(pairs, second_forward)
        # return torch.stack([torch.cat((self.linear1(pair.sentence.embedding),
        #                                self.linear3(pair.image.embedding))) for pair in pairs])
        return torch.stack([self.fusion([self.linear_text(pair.sentence.embedding),
                                         self.linear_image(pair.image.embedding)]) for pair in pairs])

    def forward_mm(self, pairs: List[MMPair], second_forward: bool):
        embeddings = self._fuse_mm_concat(pairs, second_forward)
        scores = self.classifier_mm(self.dropout_mm_linear(embeddings))
        return scores

    def forward_loss_mm(self, pairs: List[MMPair], second_forward=False):
        scores = self.forward_mm(pairs, second_forward)
        flags = torch.tensor([pair.flag for pair in pairs]).to(self.device)
        loss = F.cross_entropy(scores, flags)
        return loss

    def evaluate_mm(self, dataset: MMDataset, batch_size: int = 64):
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        data_loader = DataLoader(dataset, batch_size, collate_fn=list)
        data_loader = tqdm(data_loader, unit='batch')

        true_flags, pred_flags, image_ids = [], [], []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                scores = self.forward_mm(batch, second_forward=True)
                # image_ids += [pair.image.image_id for pair in batch]
                true_flags += [pair.flag for pair in batch]
                pred_flags += torch.argmax(scores, dim=1).tolist()

        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        f1_score = f1_score(true_flags, pred_flags)
        precision = precision_score(true_flags, pred_flags)
        recall = recall_score(true_flags, pred_flags)
        accuracy = accuracy_score(true_flags, pred_flags)

        return f1_score, precision, recall, accuracy

    def _embed_images(self, images: List[MyImage], second_forward: bool, embed_regions=False):
        if _use_image_cache(self.image_embedding, images):
            return

        embeddings = torch.stack([image.data for image in images]).to(self.device)
        # print('embeddings: ', embeddings.shape)
        embeddings = self.image_embedding.embed(embeddings) if not embed_regions \
            else self.image_embedding.embed_regions(embeddings)
        # if self.training:
        #     embeddings = embeddings.unsqueeze(0)  # zero
        # print('embeddings2: ', embeddings.shape)
        for image, embedding in zip(images, embeddings):
            image.embedding = embedding
            if second_forward is True:
                image.data = None

    def _fuse_image(self, images: List[MyImage], second_forward: bool):
        self._embed_images(images, second_forward)
        return torch.stack([image.embedding for image in images])

    def forward_image(self, images: List[MyImage], second_forward: bool = False):
        fusion = {
            'image': self._fuse_image,
        }
        embeddings = fusion['image'](images, second_forward)
        scores = self.classifier_image(self.dropout_image_linear(self.linear_image(embeddings)))
        return scores

    def forward_loss_image(self, images: List[MyImage], second_forward: bool = False):
        scores = self.forward_image(images, second_forward)
        flags = torch.tensor([image.flag for image in images]).to(self.device)
        loss = F.cross_entropy(scores, flags)
        return loss

    def evaluate_image(self, dataset: ImageDataset, batch_size: int = 64) -> Tuple[float, float, float, float]:
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        data_loader = DataLoader(dataset, batch_size, collate_fn=list)
        data_loader = tqdm(data_loader, unit='batch')

        true_flags, pred_flags = [], []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                scores = self.forward_image(batch, second_forward=True)
                true_flags += [pair.flag for pair in batch]
                # pred_flags += [0 if score < 0.5 else 1 for score in scores]
                pred_flags += torch.argmax(scores, dim=1).tolist()

        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        f1_score = f1_score(true_flags, pred_flags)
        precision = precision_score(true_flags, pred_flags)
        recall = recall_score(true_flags, pred_flags)
        accuracy = accuracy_score(true_flags, pred_flags)

        return f1_score, precision, recall, accuracy

    def _embed_sentences(self, sentence_list: List[TextSentence]):
        bs = 1

        for i in range(0, len(sentence_list), bs):
            sentences = sentence_list[i:i + bs]
            texts = [sentence.text for sentence in sentences]
            inputs = self.tokenizer(
                texts,
                max_length=512, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
            output = self.sentence_embedding(**inputs, return_dict=True)
            embeddings = output.last_hidden_state[:, 0, :]
            for sentence, embedding in zip(sentences, embeddings):
                sentence.embedding = embedding

    def _fuse_text(self, sentence_list: List[TextSentence]):
        self._embed_sentences(sentence_list)
        return torch.stack([sentence.embedding for sentence in sentence_list])

    def forward_text(self, sentence_list: List[TextSentence]):
        fusion = {
            'text': self._fuse_text,
        }
        embeddings = fusion['text'](sentence_list)
        scores = self.classifier_text(self.dropout_text_linear(self.linear_text(embeddings)))
        # print(scores.shape)
        return scores

    def forward_loss_text(self, sentence_list: List[TextSentence]):
        scores = self.forward_text(sentence_list)
        flags = torch.tensor([sentence.flag for sentence in sentence_list]).to(self.device)
        loss = F.cross_entropy(scores, flags)
        return loss

    def evaluate_text(self, dataset: TextDataset, batch_size: int = 64) -> Tuple[float, float, float]:
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        data_loader = DataLoader(dataset, batch_size, collate_fn=list)
        data_loader = tqdm(data_loader, unit='batch')

        true_flags, pred_flags = [], []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                scores = self.forward_text(batch)
                true_flags += [pair.flag for pair in batch]
                pred_flags += torch.argmax(scores, dim=1).tolist()

        from sklearn.metrics import f1_score, recall_score, accuracy_score
        recall = recall_score(true_flags, pred_flags, average='macro')
        f1_n, f1_p = f1_score(true_flags, pred_flags, average=None)
        f1_pn = (f1_p + f1_n) / 2
        accuracy = accuracy_score(true_flags, pred_flags)

        return recall, f1_pn, accuracy
