import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle



class ShapeNetDataLoader(Dataset):
    def __init__(self, root, Class, class_cls, split, uniform=False, use_normals=True):
        self.data_root = root + "/ShapeNet-55"
        self.pc_path = root + "/shapenet_pc"
        self.classes = class_cls
        self.split = split
        self.npoints = 2048
        self.uniform = uniform
        self.use_normals = use_normals
        
        self.data_list_file = os.path.join(self.data_root, f'{self.split}.txt')

        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        
        for line in lines: 
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            if taxonomy_id in Class:
                self.file_list.append({
                    'taxonomy_id': taxonomy_id, # 标签
                    'file_path': line # data
                }) 
   
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        cls = self.classes[sample['taxonomy_id']]
        label = np.array([cls]).astype(np.int32)
        # print(cls)
        data = np.load(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        if self.uniform:
            data = self.farthest_sample(data, self.npoints)
        else:
            data = self.random_sample(data, self.npoints)
        if self.use_normals:
            data = self.pc_norm(data)

        data = torch.from_numpy(data).float()
        return data, label

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def farthest_sample(self, pc, num):
        pass

    def random_sample(self, pc, num):
        N,D = pc.shape
        permutation = np.arange(N)
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
        return pc

 
class ModelNetDataLoader(Dataset):
    # output(B,D,N)
    def __init__(self, root, Class, class_cls, split, uniform=False, use_normals=True):
        self.root = root
        # print(self.root)
        self.split = split
        self.classes = class_cls
        self.uniform = uniform
        self.use_normals = use_normals
        self.npoints = 2048
        self.datapath = []
        for c in Class:
            path = [(c, line.strip()) for line in open(os.path.join(self.root, f'{self.split}/{c}.txt'), 'r')]
            # print(len(path))
            self.datapath.extend(path)
               
    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        label = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(os.path.join(self.root, fn[1]), delimiter=',').astype(np.float32)
         
        if self.uniform:
            point_set = self.farthest_point_sample(point_set, self.npoints)
            # print(point_set.shape)
        else:
            # 编写随机采样
            point_set = self.rand_point_sample(point_set, self.npoints)
            # point_set = point_set[0:self.npoints, :]

        if self.use_normals:
            point_set = self.pc_normalize(point_set[:, 0:3])
            # print("use_normals:", point_set)
        else:
            point_set = point_set[:, 0:3]
            # print(" no use_normals:", point_set[:, 0:3])
        
        return point_set, label
    
    def pc_normalize(self, pc):
        # 归一化，这里使用的是Z-score标准化方法，即为(x-mean)/std
        centroid = np.mean(pc, axis=0)
        # 求中心，对pc数组的每行求平均值，通过这条函数最后得到一个1×3的数组[x_mean,y_mean,z_mean];
        pc = pc - centroid
        # 中心点放到原点坐标
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def farthest_point_sample(self, point, npoint):
        """
        Input:
            xyz: pointcloud data, [N, D]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint, D]
        """
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point

    def rand_point_sample(self, point, npoint):
        N, D = point.shape
        permutation = np.arange(N)
        np.random.shuffle(permutation)
        point = point[permutation[:npoint]]
        # rand_point = np.array(random.sample(range(N), npoint))
        # point = point[rand_point.astype(np.int32)]
        return point
    

class ModelNetFewShotLoader(Dataset):
    def __init__(self, root, Class, class_cls, split, way, shot, fold, uniform=False, use_normals=True):

        self.root = root
        self.npoints = 2048
        self.use_normals = use_normals
        self.process_data = True
        self.uniform = uniform
        self.split = split

        self.way = way
        self.shot = shot
        self.fold = fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')


        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.split]

    def __len__(self):  
        return len(self.dataset)


    def __getitem__(self, index):
        points, label, _ = self.dataset[index]



    def __getitem__(self, index):
        points, label, _ = self.dataset[index]
        if self.uniform:
            points = self.farthest_point_sample(points, self.npoints)
            # print(point_set.shape)
        else:
            # 编写随机采样
            points = self.rand_point_sample(points, self.npoints)
            # point_set = point_set[0:self.npoints, :]
        if self.use_normals:
            points = self.pc_normalize(points[:, 0:3])
        else:
            points = points[:, 0:3]
        
        return points, label
    
    def pc_normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def farthest_point_sample(self, point, npoint):
        """
        Input:
            xyz: pointcloud data, [N, D]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint, D]
        """
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point

    def rand_point_sample(self, point, npoint):
        N, D = point.shape
        permutation = np.arange(N)
        np.random.shuffle(permutation)
        point = point[permutation[:npoint]]
        # rand_point = np.array(random.sample(range(N), npoint))
        # point = point[rand_point.astype(np.int32)]
        return point
 



class IncrementalDataSplitter():
    def __init__(self, root, data_name, split_way, task):
        self.root = root
        self.data_name = data_name
        self.split_way = split_way
        self.task = task

    def avg_split(self, T, categories):
        
        categories_per_subset = len(categories) // T

        subsets = []
        for i in range(T):
            start = i * categories_per_subset
            end = (i + 1) * categories_per_subset
            subset = categories[start:end]
            subsets.append(subset)
        return subsets

    def half_split(self, T, categories):
        categories_first_subset = len(categories) // 2
        categories_per_subset = len(categories) // ((T-1)*2)
        subsets = []
        subsets.append(categories[0:categories_first_subset])

        for i in range(T-1):
            start = categories_first_subset + i * categories_per_subset
            end = categories_first_subset + (i + 1) * categories_per_subset
            subset = categories[start:end]
            subsets.append(subset)

        return subsets

    def split_class(self):
        if self.data_name == 'modelnet10':
            catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        elif self.data_name == 'modelnet40':
            catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        elif self.data_name == 'fewshot':
            pass
        elif self.data_name == 'shapenet55':
            catfile = os.path.join(self.root, 'shapenet55_shape_names.txt')
        else:
            print("please select data_name")
        
        # 所有类
        cat = [line.rstrip() for line in open(catfile)]
        # random.shuffle(cat)

        class_cls = dict(zip(cat, range(len(cat))))
        # print(class_cls)
        # solit_cat_test_ = []
        # solit_cat_test = []
        
        if self.split_way == 'avg':
            
            solit_cat = self.avg_split(self.task, cat)

        elif self.split_way == 'half':
            solit_cat = self.half_split(self.task, cat)

        else:
            print("split way is false")
        
        return solit_cat, class_cls
        

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc


if __name__ == '__main__':

    Incremental = IncrementalDataSplitter("/home/xyj/data/modelnet", "modelnet10", 'avg', 5)
    # Incremental = IncrementalDataSplitter("/home/xyj/data/ShapeNet55-34", "shapenet55", "avg", 11)
    class_splites, class_cls = Incremental.split_class()
    print(class_cls)
    print(class_splites)
   
    dataloader_train = ModelNetDataLoader("/home/xyj/data/modelnet", class_splites[0], class_cls, split="train")
    # dataloader_train = ShapeNetDataLoader("/home/xyj/data/ShapeNet55-34", class_splites[0], class_cls, split="train")

    datas_train = torch.utils.data.DataLoader(dataloader_train, batch_size=200)
    for data, label in datas_train:
        print(data.shape)
        print(label.shape)

