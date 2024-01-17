import torch
import torch.nn as nn


class Wide(nn.Module):
    def __init__(self, dense_fea_num):
        super(Wide, self).__init__()
        self.wide = nn.Linear(dense_fea_num, 1)
        self.normalizer = nn.BatchNorm1d(num_features=dense_fea_num)

    def forward(self, x):
        # x = self.normalizer(x)
        x = self.wide(x)
        return x

    def get_parameter(self):
        return self.wide.weight, self.wide.bias

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.01)


class Deep(nn.Module):
    def __init__(self, dense_fea_num, sparse_fea_num, sparse_fea_list, k, hidden_layer_list):
        """

        :param dense_fea_num: 稠密特征数量
        :param sparse_fea_num: 稀疏特征数量
        :param sparse_fea_list: 每个稀疏特征的取值个数
        :param k: 稀疏特征的嵌入维度
        """
        super(Deep, self).__init__()
        self.dense_fea_num = dense_fea_num
        self.sparse_fea_num = sparse_fea_num
        # 为每个稀疏特征的每种取值都定义一个嵌入层
        self.sparse_emb = nn.ModuleList(
            nn.Embedding(value_num, k) for value_num in sparse_fea_list
        )

        # 输入维度为 sparse_fea_num * k + dense_fea_num
        self.layer_list = [sparse_fea_num * k + dense_fea_num] + hidden_layer_list
        self.mlp_list = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in
             zip(self.layer_list[:-1], self.layer_list[1:])]
        )
        self.active = nn.ReLU()

    def forward(self, x):
        """

        :param x: 稀疏特征数据
        :return:
        """
        for layer in self.mlp_list:
            x = layer(x)
            x = self.active(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)


class PoissonModule(nn.Module):
    def forward(self, x):
        # x = x.sum(dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.poisson(x)
        return x


class WideDeep(nn.Module):
    def __init__(self, wide_dense_fea_num, deep_dense_fea_num, deep_sparse_fea_num, sparse_fea_dict, k, hidden_layer_list):
        super(WideDeep, self).__init__()
        self.dense_fea_num = wide_dense_fea_num + deep_dense_fea_num
        self.sparse_fea_num = deep_sparse_fea_num
        self.wide = Wide(wide_dense_fea_num)
        self.deep = Deep(deep_dense_fea_num, deep_sparse_fea_num, sparse_fea_dict, k, hidden_layer_list)
        self.together = nn.Sequential(
            # nn.Linear(1 + hidden_layer_list[-1], 1),
            PoissonModule()
        )

    def forward(self, x_wide_dense, x_deep_dense, x_deep_sparse):
        if x_deep_sparse.shape[1] > 0:
            sparse_embeds = [self.deep.sparse_emb[i](x_deep_sparse[:, i]) for i in range(x_deep_sparse.shape[1])]
            sparse_embeds = torch.cat(sparse_embeds, dim=-1)
            # dense_fea和sparse_fea拼接起来
            deep_input = torch.cat([x_deep_dense, sparse_embeds], dim=-1)
        else:
            deep_input = x_deep_dense

        normalizer = nn.BatchNorm1d(num_features=x_wide_dense.shape[1]+deep_input.shape[1])
        input_tensor = torch.cat([x_wide_dense, deep_input], dim=-1)
        # input_tensor = normalizer(input_tensor)
        x_wide_dense = input_tensor[:, :x_wide_dense.shape[1]]
        deep_input = input_tensor[:, x_wide_dense.shape[1]:]
        wide_part = self.wide(x_wide_dense)
        deep_part = self.deep(deep_input)
        pred = self.together(torch.cat((wide_part, deep_part), dim=-1))
        return pred

    def get_wide_params(self):
        weights, bias = self.wide.get_parameter()
        return weights, bias

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)

    def initialize(self):
        self.wide.initialize()
        self.deep.initialize()
