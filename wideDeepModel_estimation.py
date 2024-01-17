import os
from model import WideDeep
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class GetData(Dataset):
    def __init__(self, wide_dense_num, deep_dense_num, data):
        super().__init__()

        # self.data = pd.read_csv(ori_data_path, sep=',')
        self.data = data
        self.Len = self.data.shape[0]
        self.label = self.data['Label'].values
        self.label = torch.from_numpy(self.label).type(torch.float32)
        del self.data['Label']

        self.wide_dense_part = self.data.iloc[:, :wide_dense_num].values

        self.deep_dense_part = self.data.iloc[:, wide_dense_num:wide_dense_num+deep_dense_num].values
        self.deep_sparse_part = self.data.iloc[:, wide_dense_num+deep_dense_num:].values

    def __getitem__(self, idx):
        wide_dense_part = torch.from_numpy(self.wide_dense_part[idx]).type(torch.float32)

        deep_dense_part = torch.from_numpy(self.deep_dense_part[idx]).type(torch.float32)
        deep_sparse_part = torch.from_numpy(self.deep_sparse_part[idx]).type(torch.long)
        return wide_dense_part, deep_dense_part, deep_sparse_part, self.label[idx]

    def __len__(self):
        return self.Len

    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def numpy(self):
        return self.data.values

# early_stopper = EarlyStopper(patience=3, min_delta=10)
#     for epoch in np.arange(n_epochs):
#         train_loss = train_one_epoch(model, train_loader)
#         validation_loss = validate_one_epoch(model, validation_loader)
#         if early_stopper.early_stop(validation_loss):
#             break

def data_split(data, random_seed):
    column = ['X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9',
              'price', 'poisson_mean', 'poisson_sample']
    X_P = ['X_p_0', 'X_p_1', 'X_p_2', 'X_p_3', 'X_p_4']
    X_W = ['X_w_0', 'X_w_1', 'X_w_2', 'X_w_3', 'X_w_4']

    data.columns = column
    data['X_p_0'] = data['price']
    data['X_p_1'] = data['price'] * data['X_1']
    data['X_p_2'] = data['price'] * data['X_2']
    data['X_p_3'] = data['price'] * data['X_3']
    data['X_p_4'] = data['price'] * data['X_4']
    data['X_w_0'] = 1
    data['X_w_1'] = data['X_1']
    data['X_w_2'] = data['X_2']
    data['X_w_3'] = data['X_3']
    data['X_w_4'] = data['X_4']

    x_data = data[X_P + column[:-3]]
    y_data = data['poisson_sample']
    y_data.rename("Label", inplace=True)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=random_seed)
    return x_train, x_valid, y_train, y_valid


def model(params):
    wide_dense_fea_num = params['wide_dense_fea_num']
    deep_dense_fea_num = params['deep_dense_fea_num']
    deep_sparse_fea_num = params['deep_sparse_fea_num']
    sparse_fea_list = params['sparse_fea_list']
    k = params['k']
    hidden_layer_list = params['hidden_layer_list']

    Model = WideDeep(wide_dense_fea_num,
                     deep_dense_fea_num,
                     deep_sparse_fea_num,
                     sparse_fea_list,
                     k,
                     hidden_layer_list)

    params.setdefault('save_model_path', None)
    params.setdefault('init_with_save_model', None)
    if params['init_with_save_model']:
        if params['save_model_path'] is not None and os.path.exists(params['save_model_path']):
            Model.load_state_dict(torch.load(params['save_model_path']))
            Model.eval()
            print("LOAD SAVED MODEL")
        else:
            print("NO SAVED MODEL FOUND !!!")
    else:
        Model.initialize()
        print("INITIALIZED MODEL")
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.0001,  weight_decay=0)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=False)
    return Model, optimizer, loss_fn


def mae(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return mean_absolute_error(y, pred)


def compute_weight_error(weight_pred, weight_true):
    weight = weight_pred.data.squeeze().numpy()
    error = mae(np.array(weight), np.array(weight_true))
    return error


def train(params, train_data, iter):
    loss_log = []
    Model, optimizer, loss_fn = model(params)

    for i in range(iter):
        train_data.shuffle()
        train_data_loader = DataLoader(
            dataset=train_data,
            batch_size=64,
            shuffle=True
        )
        for x_wide_dense, x_deep_dense, x_deep_sparse, label in train_data_loader:
            pred = Model(x_wide_dense, x_deep_dense, x_deep_sparse).squeeze()
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
    # print('loss:{:.4f}'.format(sum(loss_log)))
    return Model, sum(loss_log)


def test(Model, valid_data):
    valid_data_loader = DataLoader(
        dataset=valid_data,
        batch_size=64,
        shuffle=True
    )

    with torch.no_grad():
        for x_wide_dense, x_deep_dense, x_deep_sparse, label in valid_data_loader:
            pred = Model(x_wide_dense, x_deep_dense, x_deep_sparse)
            print('MAE', mae(pred.squeeze(), label))


def run(figure):
    epochs = 100
    iter = 3
    loss = []
    init_error, error = np.inf, np.inf
    real_weight = [-0.02, -0.005, -0.005, -0.005, -0.005]
    param_dict = {'wide_dense_fea_num': 5,
                  'deep_dense_fea_num': 10,
                  'deep_sparse_fea_num': 0,
                  'sparse_fea_list': [],
                  'k': 8,
                  'hidden_layer_list': [50, 1],
                  'save_model_path': './model/wide_deep_poisson_model.pt',
                  'init_with_save_model': False}
    wide_dense_fea_num = param_dict['wide_dense_fea_num']
    deep_dense_fea_num = param_dict['deep_dense_fea_num']
    data = pd.read_csv('./data/poisson_samples_test.csv')
    x_train, x_valid, y_train, y_valid = data_split(data, random_seed=1)
    df_train = pd.concat([x_train, y_train], axis=1)
    df_valid = pd.concat([x_valid, y_valid], axis=1)
    train_data = GetData(wide_dense_fea_num, deep_dense_fea_num, df_train)
    valid_data = GetData(wide_dense_fea_num, deep_dense_fea_num, df_valid)
    for i in range(epochs):
        print("EPOCH {} started".format(i + 1))
        # train_data.shuffle()
        if i >= 0:
            param_dict['init_with_save_model'] = True
        train_model, train_loss = train(param_dict, train_data, iter)
        loss.append(train_loss)
        if (i + 1) % 100 == 0:
            print("EPOCH {} finished".format(i + 1))
            test(train_model, valid_data)
        weight, bias = train_model.get_wide_params()
        error = compute_weight_error(weight, real_weight)
        print("Weight:", weight, "\nBias:", bias)
        print("PARAMETERS ERROR: ", error)
        if init_error > error:
            init_error = error
            train_model.save_model('./model/wide_deep_poisson_model.pt')
    if figure:
        plt.figure(figsize=(10, 6))
        t = np.arange(len(loss))
        plt.plot(t, loss)
        plt.show(block=True)


if __name__ == "__main__":
    run(figure=True)
