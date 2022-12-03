#Author: YJJ
#代码中所有的"图片"均指numpy数组. 如需打开外部图片, 可以使用np.array(Image.open(***))将图片转化为numpy数组

import numpy as np
import cv2
import trimesh
from skimage import measure
from tqdm import trange

class SpaceCarving:
    #构造函数
    def __init__(self, NumViews:int, Rvec:np.ndarray, Tvec:np.ndarray, K:np.ndarray, Dist:np.ndarray, Mask:np.ndarray):
        #NumViews: 视角个数
        #Rvec, Tvec: 相机外参. 把相机平移到原点、正上方为+y、正前方为+z的向量列表.
        #K: 相机内参. 透视投影矩阵, 3x3
        #Dist: 扭曲系数
        #Mask: 轮廓图列表，一共包含NumViews个二值图. 每个二值图都是一个二维数组, 只能有0和非0两种取值
        self.NumViews = NumViews
        self.Rvec = Rvec
        self.Tvec = Tvec
        self.K = K
        self.Dist = Dist
        self.Mask = Mask
        self.HasParam = False
    #设置参数
    def SetParam(self, VoxelSize:float = 1.0,
                       Origin:list = [0., 0., 0.],
                       Xrange:float = 10.0,
                       Yrange:float = 10.0,
                       Zrange:float = 10.0):
        #VoxelSize: 单个体素的大小
        #Origin: 体素的中心点
        #Xrange, Yrange, Zrange: 沿三个方向的占据长度
        self.VoxelSize = VoxelSize
        self.Origin = Origin
        self.Xrange = Xrange
        self.Yrange = Yrange
        self.Zrange = Zrange
        self.HasParam = True
    #空间雕刻, 获得体素
    def Carve(self):
        assert self.HasParam == True
        #生成密集点集
        Voxel = self.CreateDenseVoxel()
        print("Before carving:", Voxel.shape[0], "points.")
        #进行雕刻
        for i in range(self.NumViews):
            Voxel = Voxel[self.CarveOneView(i, Voxel), :]
            print(str(i+1)+"-th carving:", Voxel.shape[0], "points.")
        return Voxel
    #生成体素点集, 点用横坐标表示
    def CreateDenseVoxel(self):
        Origin, VoxelSize, Xrange, Yrange, Zrange = self.Origin, self.VoxelSize, self.Xrange, self.Yrange, self.Zrange
        X, Y, Z = np.mgrid[Origin[0]-Xrange/2+VoxelSize/2:Origin[0]+Xrange/2+VoxelSize/2:VoxelSize,
                           Origin[1]-Yrange/2+VoxelSize/2:Origin[1]+Yrange/2+VoxelSize/2:VoxelSize,
                           Origin[2]-Zrange/2+VoxelSize/2:Origin[2]+Zrange/2+VoxelSize/2:VoxelSize]
        Points = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
        return Points.T
    #进行一个视角上的雕刻, 返回0/1二值数组
    def CarveOneView(self, ViewIndex:int, Points:np.ndarray):
        #ViewIndex: 视角序号, 用于确定相机矩阵和轮廓图
        #Points: 点集的横坐标
        #进行相机变换
        Projections, _ = cv2.projectPoints(Points, self.Rvec[ViewIndex], self.Tvec[ViewIndex], self.K, self.Dist)
        Projections = Projections.squeeze()
        #舍入为整数，这样才能把它投影到Mask上
        Projections = np.round(Projections).astype(int)
        #边界限制
        X_Good = np.logical_and(Projections[:, 0] >= 0, Projections[:, 0] < self.Mask[ViewIndex].shape[1])
        Y_Good = np.logical_and(Projections[:, 1] >= 0, Projections[:, 1] < self.Mask[ViewIndex].shape[0])
        Good = np.logical_and(X_Good, Y_Good)
        #将布尔数组转化为下标（值为True的下标）
        Indices = np.where(Good)[0]
        #根据Projections[:, indices]的坐标和mask的对应值，确定保留下来的体素
        Flag = np.zeros(Projections.shape[0], dtype = bool)
        Flag[Indices] = self.Mask[ViewIndex, Projections[Indices, 1], Projections[Indices, 0]]
        return Flag
    #把Points映射到空间x>=0, y>=0, z>=0的整点上, 返回值为三维数组, 1表示占用, -1表示不占用
    def ToVolume(self, Points:np.ndarray, Occupy:int = 1, Empty:int = -1):
        #Points: 点集的横坐标
        Origin, VoxelSize, Xrange, Yrange, Zrange = self.Origin, self.VoxelSize, self.Xrange, self.Yrange, self.Zrange
        #先把坐标转换为整数
        Points = self.ShiftVertices(Points)
        #然后生成三维数组
        Volume = np.full((int(Xrange/VoxelSize)+1, int(Yrange/VoxelSize)+1, int(Zrange/VoxelSize)+1), Empty, dtype = int)
        Volume[Points[:, 0], Points[:, 1], Points[:, 2]] = Occupy
        return Volume
    def ToMesh(self, Voxel:np.ndarray):
        #转化为整点
        Volume = self.ToVolume(Voxel)
        #生成面轮廓
        Vertices, Faces, Normals, _ = measure.marching_cubes(Volume, 0)
        #转化为原来的坐标
        Vertices = self.ShiftVertices(Vertices)
        #转化为trimesh变量
        Mesh = trimesh.Trimesh(vertices = Vertices, vertex_normals = Normals, faces = Faces)
        Mesh.faces = Mesh.faces[:, ::-1]
        #返回trimesh模型
        return Mesh
    def ToCubes(self, Voxel:np.ndarray):
        HalfSize = self.VoxelSize/2
        NumVoxel = Voxel.shape[0]
        Vertices = np.vstack([Voxel+[ HalfSize, HalfSize, HalfSize],
                       Voxel+[ HalfSize,-HalfSize, HalfSize],
                       Voxel+[-HalfSize,-HalfSize, HalfSize],
                       Voxel+[-HalfSize, HalfSize, HalfSize],
                       Voxel+[ HalfSize, HalfSize,-HalfSize],
                       Voxel+[ HalfSize,-HalfSize,-HalfSize],
                       Voxel+[-HalfSize,-HalfSize,-HalfSize],
                       Voxel+[-HalfSize, HalfSize,-HalfSize],
                       ])
        Faces = np.mgrid[0:NumVoxel,0:12,0:3][0].reshape(-1,3)
        Faces[ 0::12] += np.array([0, 1, 2])*NumVoxel
        Faces[ 1::12] += np.array([0, 3, 2])*NumVoxel
        Faces[ 2::12] += np.array([3, 2, 6])*NumVoxel
        Faces[ 3::12] += np.array([3, 7, 6])*NumVoxel
        Faces[ 4::12] += np.array([7, 6, 5])*NumVoxel
        Faces[ 5::12] += np.array([7, 4, 5])*NumVoxel
        Faces[ 6::12] += np.array([4, 5, 1])*NumVoxel
        Faces[ 7::12] += np.array([4, 0, 1])*NumVoxel
        Faces[ 8::12] += np.array([4, 0, 3])*NumVoxel
        Faces[ 9::12] += np.array([4, 7, 3])*NumVoxel
        Faces[10::12] += np.array([5, 1, 2])*NumVoxel
        Faces[11::12] += np.array([5, 6, 2])*NumVoxel
        return trimesh.Trimesh(vertices = Vertices, faces = Faces, validate = False)
        
    #将实际坐标的点和整数坐标的点进行转换
    def ShiftVertices(self, Vertices:np.ndarray):
        #Vertices: 点集的横坐标
        Origin, VoxelSize, Xrange, Yrange, Zrange = self.Origin, self.VoxelSize, self.Xrange, self.Yrange, self.Zrange
        MinCoord = np.array([Origin[0]-Xrange/2+VoxelSize/2,
                      Origin[1]-Yrange/2+VoxelSize/2,
                      Origin[2]-Zrange/2+VoxelSize/2])
        if (Vertices.dtype == np.int32 or Vertices.dtype == np.int64):
            Vertices = Vertices * VoxelSize + MinCoord
        else:
            Vertices = np.round((Vertices - MinCoord) / VoxelSize).astype(np.int32)
        return Vertices

