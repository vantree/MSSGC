import argparse
from tqdm import tqdm

from numpy import mean, argmax
import numpy as np
import random
import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

from data_mm.loader import load_corpus as load_mm_corpus
from data_text.loader import load_corpus as load_text_corpus
from data_image.loader import load_corpus as load_image_corpus
from model.resnet import resnet152
from model.mssgc import MTM
from model.optimize import optimize_w_sgd, copy_model_params
from parallel.data_parallel import try_all_gpus

from torch.nn.parallel import DataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=7)
parser.add_argument('--device_ids', type=str, default='1')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--update_linear_lr', type=float, default=3e-4)
parser.add_argument('--update_embedding_lr', type=float, default=1e-5)
parser.add_argument('--senti_weight', type=float, default=0.3)
parser.add_argument('--mlp_hidden_size', type=int, default=2816)
parser.add_argument('--mlp_num_hidden_layer', type=int, default=2)
parser.add_argument('--fusion', type=str, default='concat', choices=('text', 'image', 'concat'))
parser.add_argument('--lm', type=str, default='bert', choices=('bert', 'roberta'))
parser.add_argument('--use_optimize', action='store_true', default=True)
parser.add_argument('--freeze_bert', action='store_true', default=False)
parser.add_argument('--freeze_resnet', action='store_true', default=False)
args = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


print('loss function: text_loss + image_loss + mm_loss')
seed_everything(args.seed)


mm_corpus = load_mm_corpus('/home/resource/MSD')
mm_data_loader = DataLoader(mm_corpus.train, batch_size=args.bs, collate_fn=list, shuffle=True)

text_corpus = load_text_corpus('/home/resource/MSD')
text_data_loader = DataLoader(text_corpus.train, batch_size=args.bs, collate_fn=list, shuffle=True)

image_corpus = load_image_corpus('/home/resource/MSD')
image_data_loader = DataLoader(image_corpus.train, batch_size=args.bs, collate_fn=list, shuffle=True)

lm_path = {
    'bert': '/home/resource/models/transformers/bert-base-uncased',
    'roberta': '/home/resource/models/transformers/roberta-base',
}

device = torch.device(args.cuda)
tokenizer = AutoTokenizer.from_pretrained(lm_path[args.lm])
sentence_embedding = AutoModel.from_pretrained(lm_path[args.lm]).to(device)
image_embedding = resnet152(pretrained=True).to(device)
model = MTM(device, tokenizer, sentence_embedding, image_embedding, args.mlp_hidden_size, args.mlp_num_hidden_layer)

if args.freeze_bert:
    for parameter in sentence_embedding.parameters():
        parameter.requires_grad = False
if args.freeze_resnet:
    for parameter in image_embedding.parameters():
        parameter.requires_grad = False

lm_ids = list(map(id, sentence_embedding.parameters()))
resnet_ids = list(map(id, image_embedding.parameters()))
other_parameters = filter(lambda p: id(p) not in lm_ids + resnet_ids, model.parameters())
optimizer = torch.optim.Adam([
    {'params': other_parameters, 'lr': args.lr},
    {'params': sentence_embedding.parameters(), 'lr': args.lr / 100},
    {'params': image_embedding.parameters(), 'lr': args.lr / 100},
])

recalls_text, metrics_text = [], []
f1_scores_image, metrics_image = [], []
f1_scores_mm, metrics_mm = [], []
best_f1_mm = 0
best_f1_image = 0
best_recall_text = 0
iteration = 1
text_data_loader_iterator = iter(text_data_loader)
image_data_loader_iterator = iter(image_data_loader)
for epoch in range(1, args.epochs + 1):
    loss_list_text = []
    loss_list_image = []
    loss_list_mm = []
    model.train()
    for mm in tqdm(mm_data_loader, unit='batch'):
        optimizer.zero_grad()
        # get text batch
        try:
            text = next(text_data_loader_iterator)
        except StopIteration:
            text_data_loader_iterator = iter(text_data_loader)
            text = next(text_data_loader_iterator)
        # get image batch
        try:
            image = next(image_data_loader_iterator)
        except StopIteration:
            image_data_loader_iterator = iter(image_data_loader)
            image = next(image_data_loader_iterator)

        # model 1
        text_loss = model(text)
        loss_list_text.append(text_loss.item())

        image_loss = model(image, second_forward=True)
        loss_list_image.append(image_loss.item())

        mm_loss = model(mm, second_forward=True)

        L = text_loss + image_loss + mm_loss
        L.backward()
        optimizer.step()

        del text, image, mm, L, text_loss, image_loss
        torch.cuda.empty_cache()

    loss_text = mean(loss_list_text)
    recall_text, f1_pn_text, accuracy_text = model.evaluate_text(text_corpus.test)
    recalls_text.append(recall_text)
    metrics_text.append((recall_text, f1_pn_text, accuracy_text))
    print(f'epoch #{epoch}, loss: {loss_text:.2f}, recall: {recall_text:2.2%}')

    if recall_text > best_recall_text:
        print('best text epoch: ', epoch)
        best_recall_text = recall_text
        # save_path = '/home/data1/liuyi/mtl.pt'
        # torch.save(model.state_dict(), save_path)
    del loss_text, recall_text, f1_pn_text, accuracy_text
    torch.cuda.empty_cache()

    loss_image = mean(loss_list_image)
    f1_score_image, precision_image, recall_image, accuracy_image = model.evaluate_image(image_corpus.test)
    f1_scores_image.append(f1_score_image)
    metrics_image.append((f1_score_image, precision_image, recall_image, accuracy_image))
    print(f'epoch #{epoch}, loss: {loss_image:.2f}, f1_score: {f1_score_image:2.2%}')

    if f1_score_image > best_f1_image:
        print('best image epoch: ', epoch)
        best_f1_image = f1_score_image
        # save_path = '/home/data/syd/MSD/mm/isa_t2i2_linear12.pt'
        # torch.save(model.state_dict(), save_path)
    del loss_image, f1_score_image, precision_image, recall_image, accuracy_image
    torch.cuda.empty_cache()

    loss_mm = mean(loss_list_mm)
    f1_score_mm, precision_mm, recall_mm, accuracy_mm = model.evaluate_mm(mm_corpus.test)
    f1_scores_mm.append(f1_score_mm)
    metrics_mm.append((f1_score_mm, precision_mm, recall_mm, accuracy_mm))

    print(f'epoch #{epoch}, loss: {loss_mm:.2f}, f1_score: {f1_score_mm:2.2%}')

    if f1_score_mm > best_f1_mm:
        print('best msd epoch: ', epoch)
        best_f1_mm = f1_score_mm
        save_path = '/home/data1/liuyi/mtl.pt'
        torch.save(model.state_dict(), save_path)

    del loss_mm, f1_score_mm, precision_mm, recall_mm, accuracy_mm
    torch.cuda.empty_cache()


print('loss function: (1 - {}) * mm_loss_first + {} * mm_loss_second'.format(args.senti_weight, args.senti_weight))
print('linear lr: ', args.lr)
print('embedding lr: ', args.lr / 100)
print('update linear lr: ', args.update_linear_lr)
print('update embedding lr: ', args.update_embedding_lr)


print('-----------------------------------------------------')
print('epoch\tf1_score\tprecision\trecall\t\taccuracy')
print('-----------------------------------------------------')
for epoch, metric in zip(range(1, args.epochs + 1), metrics_mm):
    f1_score, precision, recall, accuracy = metric
    print(f'{epoch}\t\t{f1_score:2.2%}\t\t{precision:2.2%}\t\t{recall:2.2%}\t\t{accuracy:2.2%}')
print('-----------------------------------------------------')

max_index = argmax(f1_scores_mm).item()
print(f'msd: best f1 score({f1_scores_mm[max_index]:2.2%}) at epoch#{max_index + 1}')

print('-----------------------------------------------------')
print('epoch\tf1_score\tprecision\trecall\t\taccuracy')
print('-----------------------------------------------------')
for epoch, metric in zip(range(1, args.epochs + 1), metrics_image):
    f1_score, precision, recall, accuracy = metric
    print(f'{epoch}\t\t{f1_score:2.2%}\t\t{precision:2.2%}\t\t{recall:2.2%}\t\t{accuracy:2.2%}')
print('-----------------------------------------------------')

max_index = argmax(f1_scores_image).item()
print(f'image: best f1 score({f1_scores_image[max_index]:2.2%}) at epoch#{max_index + 1}')

print('-----------------------------------------------------')
print('epoch\trecall\tf1_pn\t\taccuracy')
print('-----------------------------------------------------')
for epoch, metric in zip(range(1, args.epochs + 1), metrics_text):
    recall, f1_pn, accuracy = metric
    print(f'{epoch}\t\t{recall:2.2%}\t\t{f1_pn:2.2%}\t\t{accuracy:2.2%}')
print('-----------------------------------------------------')

max_index = argmax(recalls_text).item()
print(f'text: best f1 score({recalls_text[max_index]:2.2%}) at epoch#{max_index + 1}')
