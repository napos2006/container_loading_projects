"""
此文件定义与算法交互的领域模型

如集装箱类、货物类、托盘类等

长度单位为毫米(mm), 重量单位为千克(kg)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class ContainerSpec:
    """集装箱规格"""
    id: int
    name: str
    length: float
    width: float
    height: float
    max_weight: float
    door_reserve: float = 50    # 门预留空间

@dataclass
class ProductsSpec:
    """货物规格"""
    id: int
    sku: str
    frgn_name: str
    item_name: str
    length: float
    width: float
    height: float
    weight: float
    fragility: int                                      # 易碎等级
    allowed_rotations: List[Tuple[int, int, int]]       # 允许的旋转姿态

@dataclass
class PalletSpec:
    """托盘规格"""
    id: int
    length: float
    width: float
    height: float
    max_weight: float

@dataclass
class LoadingPoint:
    """装载点数据结构"""
    x: float                             # X 坐标
    y: float                             # Y 坐标
    # z: float                           # Z 坐标
    width: float                         # 可用区域宽度
    height: float                        # 可用区域高度
    active: bool = True                  # 是否可用

@dataclass
class Solution:
    """装载方案结果"""
    container_id: int
    items: List[Tuple[int, Tuple[int, int, int]]]
    volume_utilization: float           # 容积利用率
    weight_utilization: float           # 载重利用率