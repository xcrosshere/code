# Copyright (C) 2021-2023 Ye Xu, Xin Zhang, Chongpeng Huang, Xiaorong Qiu

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import time, joblib
import os
import copy, random

from PIL import Image
from Obtain_Train_Test_Config import obtain_train_test_config
import uuid

plt.ion()   # interactive mode

class MyDataset(Dataset):
    def __init__(self, dsname, st, phase, params_ottc, root_path, val_stage=True, val_ratio=0.2, transform=None):
        tt_config = obtain_train_test_config(dsname, times=params_ottc['times'], split_setup=params_ottc['split_setup'], reset=params_ottc['reset'], data_path=root_path)
        self.classes = tt_config['y_labels']
        self.dsname = dsname
        self.root_path = root_path
        self.transform = transform
        self.phase = phase

        if phase == 'train':
            self.img_files = tt_config['data'][st]['train_x']
            self.labels = tt_config['data'][st]['train_y']
            if val_stage:
                np.random.seed(st)
                rand_indices = list(range(len(self.img_files))); np.random.shuffle(rand_indices)
                end_train_index = int(len(self.img_files) * (1 - val_ratio))
                self.img_files = [self.img_files[i] for i in rand_indices[0:end_train_index]]
                self.labels = [self.labels[i] for i in rand_indices[0:end_train_index]]
        elif phase == 'val':
            if val_stage:
                train_files = tt_config['data'][st]['train_x']; train_y = tt_config['data'][st]['train_y']
                np.random.seed(st)
                rand_indices = list(range(len(train_files))); np.random.shuffle(rand_indices)
                start_val_index = int(len(train_files) * (1 - val_ratio))
                self.img_files = [train_files[i] for i in rand_indices[start_val_index:]]
                self.labels = [train_y[i] for i in rand_indices[start_val_index:]]
        elif phase == 'test':
            self.img_files = tt_config['data'][st]['test_x']
            self.labels = tt_config['data'][st]['test_y']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.root_path + '/datasets/' +self. dsname + '/' + self.img_files[idx])
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label


data_transforms_swinb = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_transforms_resnet50 = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_DA_only_scale': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_DA': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_DA_only_scale': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_DA': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_transforms_resnext101_resnet152 = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_DA': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

}

data_transforms_resnext50 = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_DA_only_scale': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_DA': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_DA_only_scale': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_DA': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_transforms_vgg16bn = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_DA_only_scale': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_DA': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_DA_only_scale': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_DA': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, (0.375, 1), (0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def obtain_scores_for_cnn(dsname, model_name, ft_type, params_ottc, data_path, da_type=None, val_skip=False, num_epochs=25, reset=False):
    tt_config = obtain_train_test_config(dsname, times=params_ottc['times'],
                                         split_setup=params_ottc['split_setup'],
                                         reset=params_ottc['reset'], data_path=data_path)
    saved_path = data_path + '/intermediate_data/obtain_cls_scores/obtain_scores_for_cnn/' + dsname + '/'
    saved_scores_path = saved_path + '/%s_trainsize%d_splittimes%d_%s.pkl' % (model_name, params_ottc['split_setup']['train_size'], tt_config['split_times'], ft_type)
    if not os.path.exists(saved_scores_path) or reset:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        best_num_epoch = -1
        if not val_skip:
            val_scores = []
            for st in range(tt_config['split_times']):
                image_datasets = gen_image_datasets(da_type, 'val', True, data_path, dsname, model_name, params_ottc, st)
                dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0, pin_memory=True) for x in ['train', 'val']}
                dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
                class_names = image_datasets['train'].classes

                criterion, exp_lr_scheduler, model_ft, optimizer_ft = config_model(class_names, device, model_name, ft_type)
                _, scores = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=num_epochs)
                val_scores.append(scores)
            val_scores = np.vstack(val_scores)
            mean_val_scores = np.mean(val_scores, axis = 0)
            best_num_epoch = np.argmax(mean_val_scores) + 1

        test_scores = []; models = []
        for st in range(tt_config['split_times']):
            image_datasets = gen_image_datasets(da_type, 'test', False, data_path, dsname, model_name, params_ottc, st)
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=0, pin_memory=True) for x in ['train', 'test']}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
            class_names = image_datasets['train'].classes
            criterion, exp_lr_scheduler, model_ft, optimizer_ft = config_model(class_names, device, model_name, ft_type)
            model, scores = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, phase_config=['train', 'test'], num_epochs=num_epochs)
            test_scores.append(scores)

            model.to("cpu")
            model_ft.to("cpu")
            torch.cuda.empty_cache()
            models.append(model)

        if ft_type == 'all':
            ret = dict(accus=np.array(test_scores), best_num_epoch=best_num_epoch, models=models)
        else:
            ret = dict(accus=np.array(test_scores), best_num_epoch=best_num_epoch)

        saved_scores_dir = os.path.dirname(saved_scores_path)
        if not os.path.exists(saved_scores_dir):
            os.makedirs(saved_scores_dir)
        joblib.dump(ret, saved_scores_path, compress=True)

        max_accu = np.max(ret['accus'], 1)
        print('mean_accu=', np.mean(max_accu)); print('max_accus=', max_accu)

    return joblib.load(saved_scores_path)


def gen_image_datasets(da_type, y, val_stage, data_path, dsname, model_name, params_ottc, st):
    image_datasets = {}
    image_datasets['train'] = MyDataset(dsname, st, 'train', params_ottc, root_path=data_path, val_stage=val_stage)
    image_datasets['test'] = MyDataset(dsname, st, 'test', params_ottc, root_path=data_path)

    if model_name == 'resnet50':
        if da_type == None:
            image_datasets['train'].transform = data_transforms_resnet50['train']
        elif da_type == 'da_scale':
            image_datasets['train'].transform = data_transforms_resnet50['train_DA_only_scale']
        elif da_type == 'da_full':
            image_datasets['train'].transform = data_transforms_resnet50['train_DA']
        image_datasets['test'].transform = data_transforms_resnet50[y]
    elif model_name == 'vgg16bn':
        if da_type == None:
            image_datasets['train'].transform = data_transforms_vgg16bn['train']
        elif da_type == 'da_scale':
            image_datasets['train'].transform = data_transforms_vgg16bn['train_DA_only_scale']
        elif da_type == 'da_full':
            image_datasets['train'].transform = data_transforms_vgg16bn['train_DA']
        image_datasets['test'].transform = data_transforms_vgg16bn[y]
    elif model_name == 'resnext50':
        if da_type == None:
            image_datasets['train'].transform = data_transforms_resnext50['train']
        elif da_type == 'da_scale':
            image_datasets['train'].transform = data_transforms_resnext50['train_DA_only_scale']
        elif da_type == 'da_full':
            image_datasets['train'].transform = data_transforms_resnext50['train_DA']
        image_datasets['test'].transform = data_transforms_resnext50[y]
    elif model_name in ['resnext101', 'resnet152']:
        if da_type == None:
            image_datasets['train'].transform = data_transforms_resnext101_resnet152['train']
        elif da_type == 'da_full':
            image_datasets['train'].transform = data_transforms_resnext101_resnet152['train_DA']
        image_datasets['test'].transform = data_transforms_resnext101_resnet152[y]
    elif model_name == 'swinb':
        image_datasets['train'].transform = data_transforms_swinb['train']
        image_datasets['test'].transform = data_transforms_swinb[y]
    return image_datasets


def config_model(class_names, device, model_name, ft_type='all'):
    if model_name == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
    elif model_name == 'vgg16bn':
        model_ft = models.vgg16_bn(pretrained=True)
    elif model_name == 'resnext50':
        model_ft = models.resnext50_32x4d(pretrained=True)
    elif model_name == 'resnext101':
        model_ft = models.resnext101_32x8d(pretrained=True)
    elif model_name == 'resnet152':
        model_ft = models.resnet152(pretrained=True)
    elif model_name == 'swinb':
        model_ft = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)

    print(model_ft)

    if ft_type == 'part':
        for param in model_ft.parameters():
            param.requires_grad = False

    if model_name in ['resnet50', 'resnext50', 'resnext101', 'resnet152']:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'vgg16bn':
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, len(class_names)))
        model_ft.classifier = classifier
    elif model_name == 'swinb':
        model_ft.head = nn.Linear(model_ft.head.in_features, len(class_names))

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.0002)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return criterion, exp_lr_scheduler, model_ft, optimizer_ft


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, dataset_sizes, phase_config=['train', 'val'], num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    scores = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phase_config:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase != 'train':
                scores.append(float(epoch_acc.cpu().numpy()))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase != 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, scores

if __name__ == '__main__':
    params_ottc_15Scenes = dict(times=1, split_setup=dict(train_size=100, test_size=100), reset=False)
    ret = obtain_scores_for_cnn('15-Scenes', 'swinb', 'part', params_ottc_15Scenes, val_skip=True, num_epochs=30, data_path='z:/data', reset=True)
    print(ret['accus'])

    # params_obtain_tt_config_COVID19 = dict(times=2, split_setup=dict(train_size=20, test_size=300), reset=False)
    # ret = obtain_scores_for_cnn('COVID-19', 'resnet50', 'all', params_obtain_tt_config_COVID19, val_ratio=0.2, num_epochs=30, data_path='../../data', reset=False)
    # print(ret['accus'])
    # params_ottc_TFFlowers = dict(times=5, split_setup=dict(train_size=10, test_size=450), reset=False)
    # ret = obtain_scores_for_cnn('TF-Flowers', 'resnet50', 'all', params_ottc_TFFlowers, val_skip=True, num_epochs=30,
    #                             data_path='../../data', reset=True)
    # print(ret['accus'])
    # params_ottc_NWPU = dict(times=5, split_setup=dict(train_size=10, test_size=60), reset=False)
    # ret = obtain_scores_for_cnn('NWPU', 'vgg16bn', 'part', params_ottc_NWPU, val_skip=True, num_epochs=30,
    #                             data_path='../../data', reset=False)
    # print(ret['accus'])
    # params_ottc_MITIndoor67 = dict(times=10, split_setup=dict(train_size=80, test_size=20), reset=False)
    # ret = obtain_scores_for_cnn('MIT Indoor-67', 'resnet50', 'all', params_ottc_MITIndoor67, val_ratio=0.2, num_epochs=30,
    #                             data_path='../../data', reset=False)
    # print(ret['accus'])
    # params_ottc_Caltech256 = dict(times=10, split_setup=dict(train_size=60, test_size=20), reset=False)
    # ret = obtain_scores_for_cnn('Caltech-256', 'resnet50', 'all', params_ottc_Caltech256, val_ratio=0.2, num_epochs=30,
    #                             data_path='../../data', reset=False)
    # print(ret['accus'])

    # params_obtain_tt_config_TESTDS = dict(times=2, split_setup=dict(train_size=5, test_size=80), reset=False)
    # ret = obtain_scores_for_cnn('test_ds', 'resnet50', 'part', params_obtain_tt_config_TESTDS, val_ratio=0.2, num_epochs=30,
    #                             data_path='../../data', reset=True)
    # print(ret['accus'])
