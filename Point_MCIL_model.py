import numpy as np
import torch
import torch.nn as nn
from knn_cuda import KNN
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

# --------transformer--------
# FPS + KNN
class Group(nn.Module): 
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # print(batch_size, num_points, _ )
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

# Transformer mlp
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Embedding module dim = 3 -> 384
class Embedding(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        # print("feature_global", feature_global.shape)
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1) # BG 512 n
        # print("feature", feature.shape)
        feature = self.second_conv(feature) # BG 384 n
        # print("feature", feature.shape)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0] # BG 384
        # print("feature_global", feature_global.shape)
        return feature_global.reshape(bs, g, self.encoder_channel)

# Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
# TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x
    
# TransformerDecoder
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        # 归一化
        self.norm = norm_layer(embed_dim)
        # 映射
        self.head = nn.Identity()

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)    
        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel     
        return x

# Mask_TransformerEncoder
class MaskTransformerEncoder(nn.Module):
    def __init__(self, mask_ratio, e_dim=384, en_depth=12, en_heads=6, drop_path_rate=0.1, mask_way='rand'):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.e_dim = e_dim
        self.en_depth = en_depth
        self.en_heads = en_heads
        self.drop_path_rate = drop_path_rate
        self.mask_way = mask_way
        # 分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.e_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.e_dim))
        self.embedding = Embedding(self.e_dim)
        # 位置编码
        self.embedding_pos = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.e_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.en_depth)]
        self.encoder = TransformerEncoder(
            embed_dim = self.e_dim,
            depth = self.en_depth,
            drop_path_rate = dpr,
            num_heads = self.en_heads,
        )
        
        self.norm = nn.LayerNorm(self.e_dim)

        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = np.random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def _mask_center_edge(self, center, noaug=False):
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug=False):
        if self.mask_way == 'rand':
            bool_masked = self._mask_center_rand(center, noaug = noaug) # B G
        elif self.mask_way == 'edge':
            bool_masked = self._mask_center_block(center, noaug = noaug)
        elif self.mask_way == 'block':
            bool_masked = self._mask_center_block(center, noaug = noaug)
        else:
            raise ValueError("Mask way is error!")

        x_embedding = self.embedding(neighborhood)
        B, Na, C = x_embedding.shape
        # 复制B个类标签token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        # 可见x
        x_vis = x_embedding[~bool_masked].reshape(B, -1, C)
        # 可见x_pos
        center_vis = center[~bool_masked].reshape(B, -1, 3)
        x_vis_pos = self.embedding_pos(center_vis)
        # 加上类标签token
        x_ct_vis = torch.cat((cls_tokens, x_vis), dim=1)
        pos_ct_vis = torch.cat((cls_pos, x_vis_pos), dim=1)

        # transformer
        x_pos_cls_vis = self.encoder(x_ct_vis, pos_ct_vis)
        x_pos_cls_vis = self.norm(x_pos_cls_vis)

        return x_pos_cls_vis, bool_masked


class PseudoRebuild(nn.Module):
    def __init__(self, e_dim=384, de_depth=4, de_heads=6, drop_path_rate=0.,num_group=32):
        super().__init__()
        self.e_dim = e_dim
        self.de_heads = de_heads
        self.de_depth = de_depth
        self.drop_path_rate = drop_path_rate
        self.num_group = num_group
        self.decoder = TransformerDecoder(
                    embed_dim=self.e_dim,
                    depth=self.de_depth,
                    drop_path_rate=self.drop_path_rate,
                    num_heads=self.de_heads,
        )
        self.onehead = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.e_dim, 3*self.num_group, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pos, return_token_num):
        x = self.decoder(x, pos, return_token_num)
        # 重建
        x = self.onehead(x.transpose(1, 2)).transpose(1, 2)
        B,Nm,_ = x.shape
        x = x.reshape(B*Nm, -1, 3)
        return x
                

class IncrementalClassifier(nn.Module):
    def __init__(self, n_classes, e_dim=384):
        super().__init__()
        self.n_classes = n_classes
        self.e_dim = e_dim
        self.cf = nn.Sequential(
                nn.Linear(self.e_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.n_classes)
            )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = nn.functional.softmax(self.cf(x), dim=1)
        # 返回置信度
        return x


class Point_MCIL(nn.Module):
    def __init__(self, n_classes, mask_ratio, e_dim=384, en_depth=12, en_heads=6, de_depth=4, de_heads=6, drop_path_rate=0.1, num_group=64, group_size=32, mask_way="rand"):
        super().__init__()
        self._n_classes = n_classes
        self.mask_ratio = mask_ratio
        self.e_dim = e_dim
        self.en_depth = en_depth
        self.de_depth = de_depth
        self.en_heads = en_heads
        self.de_heads = de_heads
        self.drop_path_rate = drop_path_rate
        self.mask_way = mask_way
        self.num_group = num_group
        self.group_size = group_size
        # 可学习的token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.e_dim))
        # 分组
        self.group = Group(self.num_group, self.group_size) # 64, 32
        # mask encoder
        self.masktransformer = MaskTransformerEncoder(mask_ratio=self.mask_ratio,
                                                    e_dim=self.e_dim,
                                                    en_depth=self.en_depth,
                                                    en_heads=self.en_heads,
                                                    drop_path_rate=self.drop_path_rate,
                                                    mask_way=self.mask_way)
        # 增量分类器
        self.classifier = IncrementalClassifier(n_classes=self._n_classes, e_dim=self.e_dim)
        

        self.pos_embed_rebuild = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.e_dim)
        )

        self.rebuild = PseudoRebuild(
            e_dim=self.e_dim,
            de_depth=self.de_depth,
            de_heads=self.de_heads,
            drop_path_rate=self.drop_path_rate,
            num_group=self.group_size,
        )
        self.loss_cd1_func = ChamferDistanceL1().cuda()
        self.loss_cd2_func = ChamferDistanceL2().cuda()
        self.loss_cls_func = nn.CrossEntropyLoss()

        self.class_pseudo = {}
        # self.average_pseudo = {}
        # self.chice_features = {}

    def pseudo_save(self, x, y):
        for pseudo_vector, label in zip(x, y):
            if str(label) in self.class_pseudo:
                self.class_pseudo[str(label)].append(list(pseudo_vector))
            else:
                self.class_pseudo[str(label)] = [list(pseudo_vector)]

        # for label, features_lists in self.class_pseudo.items():
        #     # print(features_lists)
        #     mean_features = sum(np.array(features_lists)) / len(features_lists)

        #     self.average_pseudo[label] = mean_features.tolist()

    def memory_bank(self):
        return self.class_pseudo
    
    # def criterion(self, re, gt, xp, xl):
    #     cd1 = self.loss_cd1_func(re, gt)
    #     cd2 = self.loss_cd2_func(re, gt)
    #     cls = self.loss_cls_func(xp, xl)
    #     return cd1, cd2, cls
    
    def update_class(self, n):
        self._n_classes += n
        weight = self.classifier.cf[-1].weight.data
        bias = self.classifier.cf[-1].bias.data
        self.classifier.cf[-1] = nn.Linear(256, self._n_classes)
        torch.nn.init.xavier_uniform_(self.classifier.cf[-1].weight)
        self.classifier.cf[-1].bias.data.fill_(0.01)
        self.classifier.cf[-1].weight.data[:self._n_classes - n] = weight
        self.classifier.cf[-1].bias.data[:self._n_classes - n] = bias

    # def forward(self, pts, pls): # 数据和标签
    def forward(self, pts): # 数据和标签
        # self.pls = pls
        neighborhood, center = self.group(pts) # (B, 64, 32, 3) | (B, 64, 3)
        x_vis, masked = self.masktransformer(neighborhood, center) # (B, 32+1, 384) | (B, 64)
        B,_,C = x_vis.shape
        # center[masked] # [200*32, 3]
        # 分类
        concat_f = torch.cat([x_vis[:, 0], x_vis[:, 1:].max(1)[0]], dim=-1) # (B, 768)
        x_predict = self.classifier(concat_f) # (B, num_class)
        # 重建
        x_vis_pos = self.pos_embed_rebuild(center[~masked]).reshape(B, -1, C) # (200, 32, 384)
        x_mask_pos = self.pos_embed_rebuild(center[masked]).reshape(B, -1, C) # (200, 32, 384)
        _,Nm,_ = x_mask_pos.shape
        mask_token = self.mask_token.expand(B, Nm, -1) # (200, 32, 384)
        x_full = torch.cat([x_vis[:, 1:], mask_token], dim=1) # (200, 64, 384)
        pos_full = torch.cat([x_vis_pos, x_mask_pos], dim=1) # (200, 64, 384)
        rebuild_points = self.rebuild(x_full, pos_full, Nm) # (200*32, 32, 3)
        gt_points = neighborhood[masked].reshape(B*Nm,-1,3) # (200*32, 32, 3)
        # 可视化

        mask_p = rebuild_points + center[masked].unsqueeze(1) # (200*32, 32, 3)
        vis_points = neighborhood[~masked].reshape(B*(self.num_group-Nm), -1, 3) # (200*32, 32, 3)
        
        vis_p = vis_points + center[~masked].unsqueeze(1) # (200*32, 32, 3)
        full_p = torch.cat([vis_p, mask_p], dim=0) # (200*64, 32, 3)
        full_center = torch.cat([center[masked], center[~masked]], dim=0) # (200*64, 3)
        # 
        mask_p_R = mask_p.reshape(B, Nm , self.group_size, 3)
        vis_p_R = vis_p.reshape(B, self.num_group-Nm, self.group_size, 3)
        full_p_R = torch.cat([vis_p_R, mask_p_R], dim=1).reshape(B, -1, 3) # (200, 64*32, 3)

        # 先拼接后加位置信息
        # full_points = torch.cat([rebuild_points, vis_points], dim=0)
        # full = full_points + full_center.unsqueeze(1)

        # 200个数据全混一起了
        # ret_full = full_p.reshape(-1, 3).unsqueeze(0) # 可见 + 掩蔽 (1, 200*64*32, 3)
        # ret_vis = vis_p.reshape(-1, 3).unsqueeze(0) # 可见 (1, 200*32*32, 3)
        # ret_mask = mask_p.reshape(-1, 3).unsqueeze(0) # 掩蔽 (1, 200*32*32, 3)
        # ret_center = full_center.reshape(-1, 3).unsqueeze(0) # 中心 (1, 200*64, 3)


        # 保存伪样本(全/遮蔽)
        # self.pseudo_save(full_p_R, pls)

        # 计算损失
        # loss_cd1, loss_cd2, loss_cls = self.criterion(rebuild_points, gt_points, x_predict, pls)
        # loss_dict = {'loss_cd1': loss_cd1,
        #             'loss_cd2': loss_cd2,
        #             'loss_cls': loss_cls}
        
        # loss_dict = {'loss_cd1': loss_cd1.item(),
        #             'loss_cd2': loss_cd2.item(),
        #             'loss_cls': loss_cls.item()}

        # single_loss = loss_cd2 + loss_cls
        # return x_predict, rebuild_points, gt_points, loss_dict
        return x_predict, rebuild_points, gt_points
        
        
