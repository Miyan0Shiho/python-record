# ***MMPOSE***

组成部分：

**apis** 提供用以模型推理的API

**structures**提供bbox,keypoint和PoseDataSample等数据结构

​						bbox用来处理边界框

​						[keypoint](C:\Users\liumi\Desktop\深度学习知识点整理\keypoint.md)用来处理关键点

​						PoseDataSample

**datasets**支持用于姿态估计的各种数据集
				**transforms** 包含各种数据增强变换
**codecs** 提供姿态编解码器：编码器用于将姿态信息（通常为关键点坐标）编码为模型学习目标（如热力图)，解码器则用于将模型输出解码为姿态估计结果
**models** 以模块化结构提供了姿态估计模型的各类组件
**pose_estimators** 定义了所有姿态估计模型类
**data_preprocessors** 用于预处理模型的输入数据
**backbones** 包含各种骨干网络
**necks** 包含各种模型颈部组件
**heads** 包含各种模型头部
**losses** 包含各种损失函数
**engine** 包含与姿态估计任务相关的运行时组件
**hooks** 提供运行时的各种钩子
**evaluation** 提供各种评估模型性能的指标
**visualization** 用于可视化关键点骨架和热力图等信息



##### 1.1 整体架构与设计

![img](https://img-blog.csdnimg.cn/4157de0d686b4ee488937bb1f24e8a7b.png)

一般来说，开发者在项目开发过程中经常接触内容的主要有五个方面：
**通用：**环境、钩子（Hook）、模型权重存取（Checkpoint）、日志（Logger）等
**数据：**数据集、数据读取（Dataloader）、数据增强等
**训练：**优化器、学习率调整等
**模型：**主干网络、颈部模块（Neck）、预测头模块（Head）、损失函数等（Loss）
**评测：**评测指标（Metric）、评测器（Evaluator）等
其中通用、训练和评测相关的模块往往由训练框架提供，开发者只需要调用和调整参数，不需要自行实现，开发者主要实现的是数据和模型部分。



##### 1.1.1配置文件

- 在MMPose中，我们通常 python 格式的配置文件，用于整个项目的定义、参数管理。
- 所有新增的模块都需要使用[注册器](C:\Users\liumi\Desktop\深度学习知识点整理\注册器（Register）.md)（Registry）进行注册，并在对应目录的 **init**.py 中进行 import，以便能够使用配置文件构建其实例。



##### 1.1.2数据

MMPose数据的组织主要包含三方面

​	数据集元信息

​	数据集

​	数据流水线



##### 1.1.2.1数据集元信息

元信息指具体标注之外的数据集信息。姿态估计数据集的元信息通常包括：关键点和骨骼连接的定义，对称性，关键点性质（如关键点权重，标注标准差所属上下半身）。在MMpose中，数据集的元信息使用python格式的配置文件保存。



**1.1.2.2数据集**

在MMPose中使用自定义数据集时，一般将数据集转化为已支持的格式（如COCO或MPII）。大部分2D关键点数据集以COCO形式组织，可以继承基类BaseCocoStyleDataset,并重写它的方法（init()和_load_annotations()方法），以扩展到新的2D关键点数据集。如果自定义的数据集无法被BaseCocoStyleDataset支持，则继承MMPose提供的BaseDataset基类。



**1.1.2.3数据流水线**

```python
# pipelines
train_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', target_type='heatmap', encoder=codec),
    dict(type='PackPoseInputs')
]
test_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

```

在关键点检测任务中，数据一般会在三个尺度空间中变换：

​			原始图片空间：图片存储时的原始空间

​			输入图片空间：模型输入的图片尺度空间，所有图片和标注被缩放到输入尺度

​			输出尺度空间：模型输出和训练监督信息所在的尺度空间，如64×64（热力图），1×1（回归										坐标值）





在MMPose中数据变换所需要的模块在MMPose/mmpose/datasets/transforms目录下

![img](https://img-blog.csdnimg.cn/a1a31d45d6a543b58f0e3b6beb3ac37b.png)

**1.1.2.4数据增强**

数据增强中常用的变换存放在MMPose/mpose/transforms/common_transforms.py中如RandomFlip、RandomHalfBody等

对于top-down方法。Shift,Rptate,Resize操作由RandomBBoxTrans来实现；对于botton-up方法，这些则是BottomupRandomAffine实现

值得注意的是，大部分数据变换都依赖于 bbox_center 和 bbox_scale，它们可以通过 GetBBoxCenterScale 来得到

GetBBoxCenterScale（由bbox_xyxy2cs具体实现）:
	功能把bboxes从（x1,y1,x2,y2）转换为center和scale

​	center(x,y)：是bbox中心的坐标

​			
$$
x=(x1+x2)*0.5
$$

$$
y=(y1+y2)*0.5
$$

​	scale(w,h):是bbox宽度和高度归一化比例因子， padding是bbox的填充比例,默认值为1.25

​		
$$
w=(x2-x1)*padding
$$

$$
h=(y2-y1)*padding
$$

##### 1.1.2.5 数据变换

我们使用仿射变换，将图像和坐标标注从原始图片空间变换到输入图片空间。这一操作在 top-down 方法中由 TopdownAffine 完成，在 bottom-up 方法中则由 BottomupRandomAffine 完成。

##### 1.1.2.5 数据编码

在模型训练时，数据从原始空间变换到输入图片空间后，需要使用 GenerateTarget 来生成训练所需的监督目标（比如用坐标值生成高斯热图），我们将这一过程称为编码（Encode），反之，通过高斯热图得到对应坐标值的过程称为解码（Decode）。

在 MMPose 中，我们将编码和解码过程集合成一个编解码器（Codec），在其中实现 encode() 和 decode()。





目前 MMPose 支持生成以下类型的监督目标：

​		heatmap: 高斯热图
​		keypoint_label: 关键点标签（如归一化的坐标值）
​		keypoint_xy_label: 单个坐标轴关键点标签
​		heatmap+keypoint_label: 同时生成高斯热图和关键点标签
​		multiscale_heatmap: 多尺度高斯热图



生成的监督目标会按以下关键字进行封装：

​		heatmaps：高斯热图
​		keypoint_labels：关键点标签（如归一化的坐标值）
​		keypoint_x_labels：x 轴关键点标签
​		keypoint_y_labels：y 轴关键点标签
​		keypoint_weights：关键点权重



**实现函数**

```python
@TRANSFORMS.register_module()
class GenerateTarget(BaseTransform):
    """Encode keypoints into Target.

    The generated target is usually the supervision signal of the model
    learning, e.g. heatmaps or regression labels.

    Required Keys:

        - keypoints
        - keypoints_visible
        - dataset_keypoint_weights

    Added Keys (depends on the args):
        - heatmaps
        - keypoint_labels
        - keypoint_x_labels
        - keypoint_y_labels
        - keypoint_weights

    Args:
        encoder (dict | list[dict]): The codec config for keypoint encoding
        target_type (str): The type of the encoded form of the keypoints.
            Should be one of the following options:

            - ``'heatmap'``: The encoded should be instance-irrelevant
                heatmaps and will be stored in ``results['heatmaps']``
            - ``'multilevel_heatmap'`` The encoded should be a list of
                heatmaps and will be stored in ``results['heatmaps']``.
                Note that in this case, ``self.encoder`` should also be
                a list, and each encoder encodes a single-level heatmaps.
            - ``'keypoint_label'``: The encoded should be instance-level
                labels and will be stored in ``results['keypoint_label']``
            - ``'keypoint_xy_label'``: The encoed should be instance-level
                labels in x-axis and y-axis respectively. They will be stored
                in ``results['keypoint_x_label']`` and
                ``results['keypoint_y_label']``
            - ``'heatmap+keypoint_label'``: The encoded should be heatmaps and
                keypoint_labels, will be stored in ``results['heatmaps']``
                and ``results['keypoint_label']``
        use_dataset_keypoint_weights (bool): Whether use the keypoint weights
            from the dataset meta information. Defaults to ``False``
    """

    def __init__(self,
                 encoder: MultiConfig,
                 target_type: str,
                 use_dataset_keypoint_weights: bool = False) -> None:
        super().__init__()
        self.encoder_cfg = deepcopy(encoder)
        self.target_type = target_type
        self.use_dataset_keypoint_weights = use_dataset_keypoint_weights

        if self.target_type == 'multilevel_heatmap':
            if not isinstance(self.encoder_cfg, list):
                raise ValueError(
                    'The encoder should be a list if target type is '
                    '"multilevel_heatmap"')
            self.encoder = [
                KEYPOINT_CODECS.build(cfg) for cfg in self.encoder_cfg
            ]
        else:
            self.encoder = KEYPOINT_CODECS.build(self.encoder_cfg)

    def transform(self, results: Dict) -> Optional[dict]:

        if results.get('transformed_keypoints', None) is not None:
            # use keypoints transformed by TopdownAffine
            keypoints = results['transformed_keypoints']
        elif results.get('keypoints', None) is not None:
            # use original keypoints
            keypoints = results['keypoints']
        else:
            raise ValueError(
                'GenerateTarget requires \'transformed_keypoints\' or'
                ' \'keypoints\' in the results.')

        keypoints_visible = results['keypoints_visible']

        if self.target_type == 'heatmap':
            heatmaps, keypoint_weights = self.encoder.encode(
                keypoints=keypoints, keypoints_visible=keypoints_visible)

            results['heatmaps'] = heatmaps
            results['keypoint_weights'] = keypoint_weights

        elif self.target_type == 'keypoint_label':
            keypoint_labels, keypoint_weights = self.encoder.encode(
                keypoints=keypoints, keypoints_visible=keypoints_visible)

            results['keypoint_labels'] = keypoint_labels
            results['keypoint_weights'] = keypoint_weights

        elif self.target_type == 'keypoint_xy_label':
            x_labels, y_labels, keypoint_weights = self.encoder.encode(
                keypoints=keypoints, keypoints_visible=keypoints_visible)

            results['keypoint_x_labels'] = x_labels
            results['keypoint_y_labels'] = y_labels
            results['keypoint_weights'] = keypoint_weights

        elif self.target_type == 'heatmap+keypoint_label':
            heatmaps, keypoint_labels, keypoint_weights = self.encoder.encode(
                keypoints=keypoints, keypoints_visible=keypoints_visible)

            results['heatmaps'] = heatmaps
            results['keypoint_labels'] = keypoint_labels
            results['keypoint_weights'] = keypoint_weights

        elif self.target_type == 'multilevel_heatmap':
            heatmaps = []
            keypoint_weights = []

            for encoder in self.encoder:
                _heatmaps, _keypoint_weights = encoder.encode(
                    keypoints=keypoints, keypoints_visible=keypoints_visible)
                heatmaps.append(_heatmaps)
                keypoint_weights.append(_keypoint_weights)

            results['heatmaps'] = heatmaps
            # keypoint_weights.shape: [N, K] -> [N, n, K]
            results['keypoint_weights'] = np.stack(keypoint_weights, axis=1)

        else:
            raise ValueError(f'Invalid target type {self.target_type}')

        # multiply meta keypoint weight
        if self.use_dataset_keypoint_weights:
            results['keypoint_weights'] *= results['dataset_keypoint_weights']

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += (f'(encoder={str(self.encoder_cfg)}, ')
        repr_str += (f'(target_type={str(self.target_type)}, ')
        repr_str += ('use_dataset_keypoint_weights='
                     f'{self.use_dataset_keypoint_weights})')
        return repr_str

```



标注信息会新增一个维度表示同一张图的不同目标

```python
[batch_size, num_instances, num_keypoints, dim_coordinates]

```





##### 1.1.2.5数据打包

数据经过前处理变换后，最终需要通过 PackPoseInputs 打包成数据样本。该操作定义在 $MMPOSE/mmpose/datasets/transforms/formatting.py 中。
打包过程会将数据流水线中用字典 results 存储的数据转换成用 MMPose 所需的标准数据结构， 如 InstanceData，PixelData，PoseDataSample 等。
具体而言，我们将数据样本内容分为 gt（标注真值） 和 pred（模型预测）两部分，它们都包含以下数据项：
		instances(numpy.array)：实例级别的原始标注或预测结果，属于原始尺度空间
		instance_labels(torch.tensor)：实例级别的训练标签（如归一化的坐标值、关键点可见性），																	属于输入尺度空间
		fields(torch.tensor)：像素级别的训练标签（如高斯热图）或预测结果，属于输出尺度空间



PoseDataSample 底层实现的例子：

```python
    def get_pose_data_sample(self, multilevel: bool = False):
        # meta
        pose_meta = dict(
            img_shape=(600, 900),  # [h, w, c]
            crop_size=(256, 192),  # [h, w]
            heatmap_size=(64, 48),  # [h, w]
        )
        # gt_instances
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.rand(1, 4)
        gt_instances.keypoints = torch.rand(1, 17, 2)
        gt_instances.keypoints_visible = torch.rand(1, 17)

        # pred_instances
        pred_instances = InstanceData()
        pred_instances.keypoints = torch.rand(1, 17, 2)
        pred_instances.keypoint_scores = torch.rand(1, 17)

        # gt_fields
        if multilevel:
            # generate multilevel gt_fields
            metainfo = dict(num_keypoints=17)
            sizes = [(64, 48), (32, 24), (16, 12)]
            heatmaps = [np.random.rand(17, h, w) for h, w in sizes]
            masks = [torch.rand(1, h, w) for h, w in sizes]
            gt_fields = MultilevelPixelData(
                metainfo=metainfo, heatmaps=heatmaps, masks=masks)
        else:
            gt_fields = PixelData()
            gt_fields.heatmaps = torch.rand(17, 64, 48)

        # pred_fields
        pred_fields = PixelData()
        pred_fields.heatmaps = torch.rand(17, 64, 48)

        data_sample = PoseDataSample(
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            gt_fields=gt_fields,
            pred_fields=pred_fields,
            metainfo=pose_meta)

        return data_sample

```







##### 1.1.3 模型

**在 MMPose 1.0中，模型由以下几部分构成：**
		预处理器（DataPreprocessor）：完成图像归一化和通道转换等前处理
		主干网络 （Backbone）：用于特征提取
		颈部模块（Neck）：GAP，FPN 等可选项
		预测头（Head）：用于实现核心算法功能和损失函数定义

​	我们在 $MMPOSE/models/pose_estimators/base.py 下为姿态估计模型定义了一个基类 		BasePoseEstimator，所有的模型（如 TopdownPoseEstimator）都需要继承这个基类，并重载对应的方法。



**在模型的 forward() 方法中提供了三种不同的模式：**
		mode == ‘loss’：返回损失函数计算的结果，用于模型训练
		mode == ‘predict’：返回输入尺度下的预测结果，用于模型推理
		mode == ‘tensor’：返回输出尺度下的模型输出，即只进行模型前向传播，用于模型导出
		开发者需要在 PoseEstimator 中按照模型结构调用对应的 Registry ，对模块进行实例化。以 top-down 模型为例：

```python
@MODELS.register_module()
class TopdownPoseEstimator(BasePoseEstimator):
    """Base class for top-down pose estimators.

    Args:
        backbone (dict): The backbone config
        neck (dict, optional): The neck config. Defaults to ``None``
        head (dict, optional): The head config. Defaults to ``None``
        train_cfg (dict, optional): The runtime config for training process.
            Defaults to ``None``
        test_cfg (dict, optional): The runtime config for testing process.
            Defaults to ``None``
        data_preprocessor (dict, optional): The data preprocessing config to
            build the instance of :class:`BaseDataPreprocessor`. Defaults to
            ``None``.
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to ``None``
    """

    _version = 2

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)


```



##### **1.1.3.1 前处理器（DataPreprocessor）**

从 MMPose 1.0 开始，在模型中添加了新的前处理器模块，用以完成图像归一化、通道顺序变换等操作。这样做的好处是可以利用 GPU 等设备的计算能力加快计算，并使模型在导出和部署时更具完整性。
在配置文件中，一个常见的 data_preprocessor 如下：

```python
data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
```

它会将输入图片的通道顺序从 bgr 转换为 rgb，并根据 mean 和 std 进行数据归一化。





##### 1.1.3.2 主干网络（Backbone）

MMPose 实现的主干网络存放在 $MMPOSE/mmpose/models/backbones 目录下。
在实际开发中，开发者经常会使用预训练的网络权重进行迁移学习，这能有效提升模型在小数据集上的性能。 在 MMPose 中，只需要在配置文件 backbone 的 init_cfg 中设置：

```python
init_cfg=dict(
    type='Pretrained',
    checkpoint='PATH/TO/YOUR_MODEL_WEIGHTS.pth'),
```


其中 checkpoint 既可以是本地路径，也可以是下载链接。因此，如果你想使用 Torchvision 提供的预训练模型（比如ResNet50），可以使用：

```python
init_cfg=dict(
    type='Pretrained',
    checkpoint='torchvision://resnet50')
```

除了这些常用的主干网络以外，你还可以从 MMClassification 等其他 OpenMMLab 项目中方便地迁移主干网络，它们都遵循同一套配置文件格式，并提供了预训练权重可供使用。
需要强调的是，如果你加入了新的主干网络，需要在模型定义时进行注册：

```python
@MODELS.register_module()
class YourBackbone(BaseBackbone):
```


同时在 $MMPOSE/mmpose/models/backbones/init.py 下进行 import，并加入到 all 中，才能被配置文件正确地调用。init.py的内容如下：

```python
from .alexnet import AlexNet
from .cpm import CPM
from .hourglass import HourglassNet
from .hourglass_ae import HourglassAENet
from .hrformer import HRFormer
from .hrnet import HRNet
from .litehrnet import LiteHRNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mspn import MSPN
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .rsn import RSN
from .scnet import SCNet
from .seresnet import SEResNet
from .seresnext import SEResNeXt
from .shufflenet_v1 import ShuffleNetV1
from .shufflenet_v2 import ShuffleNetV2
from .swin import SwinTransformer
from .tcn import TCN
from .v2v_net import V2VNet
from .vgg import VGG
from .vipnas_mbv3 import ViPNAS_MobileNetV3
from .vipnas_resnet import ViPNAS_ResNet

__all__ = [
    'AlexNet', 'HourglassNet', 'HourglassAENet', 'HRNet', 'MobileNetV2',
    'MobileNetV3', 'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SCNet',
    'SEResNet', 'SEResNeXt', 'ShuffleNetV1', 'ShuffleNetV2', 'CPM', 'RSN',
    'MSPN', 'ResNeSt', 'VGG', 'TCN', 'ViPNAS_ResNet', 'ViPNAS_MobileNetV3',
    'LiteHRNet', 'V2VNet', 'HRFormer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'SwinTransformer'
]
```



##### 1.1.3.3 颈部模块（Neck）

颈部模块通常是介于主干网络和预测头之间的模块，在部分模型算法中会用到，常见的颈部模块有：
Global Average Pooling (GAP)
Feature Pyramid Networks (FPN)
$MMPOSE/mmpose/models/necks/init.py的内容如下：

```python
from .fpn import FPN
from .gap_neck import GlobalAveragePooling
from .posewarper_neck import PoseWarperNeck

__all__ = ['GlobalAveragePooling', 'PoseWarperNeck', 'FPN']
```

##### 1.1.3.4 预测头（Head）

通常来说，预测头是模型算法实现的核心，用于控制模型的输出，并进行损失函数计算。
MMPose 中 Head 相关的模块定义在 $MMPOSE/mmpose/models/heads 目录下，开发者在自定义预测头时需要继承我们提供的基类 BaseHead，并重载以下三个方法对应模型推理的三种模式：
forward()
predict()
loss()
$MMPOSE/mmpose/models/heads/init.py的内容如下：

```python
from .base_head import BaseHead
from .heatmap_heads import (CPMHead, HeatmapHead, MSPNHead, SimCCHead,
                            ViPNASHead)
from .regression_heads import (DSNTHead, IntegralRegressionHead,
                               RegressionHead, RLEHead)

__all__ = [
    'BaseHead', 'HeatmapHead', 'CPMHead', 'MSPNHead', 'ViPNASHead',
    'RegressionHead', 'IntegralRegressionHead', 'SimCCHead', 'RLEHead',
    'DSNTHead'
]
```


predict：
输出：返回的是输入图片尺度下的结果，因此需要调用 self.decode() 对网络输出进行解码，这一过程实现在 BaseHead 中已经实现，它会调用编解码器提供的 decode() 方法来完成解码。另外，在 predict() 中进行测试时增强。在进行预测时，一个常见的测试时增强技巧是进行翻转集成。即，将一张图片先进行一次推理，再将图片水平翻转进行一次推理，推理的结果再次水平翻转回去，对两次推理的结果进行平均。这个技巧能有效提升模型的预测稳定性。
下面是在 RegressionHead 中定义 predict() 的例子：

```python
    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from outputs."""

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords = self.forward(_feats)
            _batch_coords_flip = flip_coordinates(
                self.forward(_feats_flip),
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)
            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        return preds

```

2. **编解码器**
在关键点检测任务中，根据算法的不同，需要利用标注信息，生成不同格式的训练目标，比如归一化的坐标值、一维向量、高斯热图等。同样的，对于模型输出的结果，也需要经过处理转换成标注信息格式。我们一般将标注信息到训练目标的处理过程称为编码，模型输出到标注信息的处理过程称为解码。
编码和解码是一对紧密相关的互逆处理过程。在 MMPose 早期版本中，编码和解码过程往往分散在不同模块里，使其不够直观和统一，增加了学习和维护成本。
MMPose 1.0 中引入了新模块编解码器（Codec） ，将关键点数据的编码和解码过程进行集成，以增加代码的友好度和复用性。
编解码器在工作流程中所处的位置如下所示：
3. ![img](https://img-blog.csdnimg.cn/540e0186915545008fcaf6b3c80f7df8.png)

一个编解码器主要包含两个部分：
编码器
解码器





##### 2.1 编码器

编码器主要负责将处于输入图片尺度的坐标值，编码为模型训练所需要的目标格式，主要包括：
归一化的坐标值：用于 Regression-based 方法 （直接预测每个关键点的位置坐标）
一维向量：用于 SimCC-based 方法
高斯热图：用于 Heatmap-based 方法（针对每个关键点预测一张热力图，预测出现在每个位置上的分数）
以 Regression-based 方法的编码器为例：
@abstractmethod
def encode(
    self,
    keypoints: np.ndarray,
    keypoints_visible: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Encoding keypoints from input image space to normalized space.

```python
Args:
    keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
    keypoints_visible (np.ndarray): Keypoint visibilities in shape
        (N, K)

Returns:
    tuple:
    - reg_labels (np.ndarray): The normalized regression labels in
        shape (N, K, D) where D is 2 for 2d coordinates
    - keypoint_weights (np.ndarray): The target weights in shape
        (N, K)
"""

if keypoints_visible is None:
    keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

w, h = self.input_size
valid = ((keypoints >= 0) &
         (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
             keypoints_visible > 0.5)

reg_labels = (keypoints / np.array([w, h])).astype(np.float32)
keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)

return reg_labels, keypoint_weights
```
参数说明：
N：instance number
K：keypoint number
D：keypoint dimension
L：embedding tag dimension
[w, h]：image size
[W, H]：heatmap size
sigma：The sigma value of the Gaussian heatmap





##### 2.1.1 Heatmap-based

Heatmap-based方法为每个关节生成似然热图（likelihood heatmap），并使用argmax 或 soft-argmax 操作把关节定位到一个点。
2D heatmap生成为一个二维高斯分布，其中心为标注的关节位置，通过为每个位置分配概率值来抑制false positive并平滑训练过程。
量化误差来源：通过2D高斯分布生成高斯热图作为标签，监督模型输出，通过L2 loss来进行优化。而这种方法下得到的Heatmap尺寸往往是小于图片原尺寸的，因而最后通过argmax得到的坐标放大回原图，会承受不可避免的量化误差。
heatmap-based的不足：
计算量大
存储量大
扩展到3D或4D（空间+时间）成本高
难以把heatmap布署到one-state方法中
低分辨率输入的性能受到限制：即在低分辨率图片上掉点严重：对于HRNet-W48，当输入分辨率从256x256降到64x64，AP会从75.1掉到48.5
为了提高特征图分辨率以获得更高的定位精度，需要多个计算量大的上采样层：为了提升精度，需要多个上采样层来将特征图分辨率由低向高进行恢复：通常来说上采样会使用转置卷积来获得更好的性能，但相应的计算量也更大，骨干网络输出的特征图原本通道数就已经很高了，再上采样带来的开销是非常庞大的
需要额外的后处理来减小尺度下降带来的量化误差：如DARK修正高斯分布，用argmax获取平面上的极值点坐标等
