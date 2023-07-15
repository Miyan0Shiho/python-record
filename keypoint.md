# keypoint

#### 公有属性：

angle:角度，表示为关键点的方向，初值为-1，float量

class_id：对class_id对每个特征点进行分类，初值为-1，int量

octave：代表是从金字塔哪一层提取到的数据，int量

pt：关键点的点坐标，opencv中的Point2f量

response:相应强度，代表该点角点的程度，float量

size：该点直径的大小，float量



#### 公有成员函数

##### 		构造函数：

​		keypoint(): 默认构造函数

```python
KeyPoint(pt, size, angle = -1, response = 0, octave = 0, class_id = -1)
参数
    x	     ： 关键点的x坐标
    y	     ： 关键点的y坐标
    _size	 ： 关键点直径
    _angle	 ： 关键点方向
    _response：关键点上的关键点检测器响应（即关键点的强度）
    _octave	 ： 已检测到关键点的pyramid octave
    _class_id： 关键点ID

```

​		

```python
KeyPoint(x, y, size, angle = -1, response = 0, octave = 0, class_id = -1)
参数
    x	     ： 关键点的x坐标
    y	     ： 关键点的y坐标
    _size	 ： 关键点直径
    _angle	 ： 关键点方向
    _response：关键点上的关键点检测器响应（即关键点的强度）
    _octave	 ： 已检测到关键点的pyramid octave
    _class_id： 关键点ID

```

##### 静态公有成员函数：

```python
static void convert(const std :: vector <KeyPoint> &keypoints, std :: vector <Point2f> &points2f, size = 1, response = 1, octave = 0, class_id = -1)
参数
points2f  : 每个关键点的（x，y）坐标数组
keypoints : 从任何特征检测算法（如SIFT / SURF / ORB）获得的关键点
size	  : 关键点直径
response  : 关键点上的关键点检测器响应（即关键点的强度）
octave	  : 已检测到关键点的pyramid octave
class_id  : 关键点 id

```

```python
static float overlap (const KeyPoint &kp1, const KeyPoint &KP2)
参数
    kp1	: First keypoint
    kp2	: Second keypoint

```

