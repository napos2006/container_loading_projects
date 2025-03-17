"""
此文件包含用于将数据从一种格式转换为另一种格式的转换器

将数据库模型转换为领域模型
"""

from core.domain import ContainerSpec, ProductsSpec, PalletSpec
from database.models import Container, Product, Pallet
from typing import List, Tuple

def convert_container_to_spec(db_container: Container) -> ContainerSpec:
    """将数据库Container对象转换为算法规格"""
    return ContainerSpec(
        id=db_container.container_id,
        name=db_container.name,
        length=db_container.length,
        width=db_container.width,
        height=db_container.height,
        max_weight=float(db_container.max_weight),
    )

def convert_product_to_spec(db_product: Product) -> ProductsSpec:
    """将数据库Product对象转换为算法规格"""
    return ProductsSpec(
        id=db_product.product_id,
        sku=db_product.sku,
        frgn_name=db_product.frgn_name,
        item_name=db_product.item_name,
        length=db_product.length,
        width=db_product.width,
        height=db_product.height,
        weight=float(db_product.weight),
        fragility=db_product.fragile,
        allowed_rotations=decode_rotations(db_product.direction),       # 允许的旋转姿态，编码函数待实现
    )

def convert_pallet_to_spec(db_pallet: Pallet) -> PalletSpec:
    """将数据库Pallet对象转换为算法规格"""
    return PalletSpec(
        id=db_pallet.pallet_id,
        length=db_pallet.length,
        width=db_pallet.width,
        height=db_pallet.height,
        max_weight=float(db_pallet.max_weight),
    )

def decode_rotations(direction: int) -> List[Tuple[int, int, int]]:
    """将方向编码转换为允许的旋转姿态列表
    
    Args:
        direction: 方向编码
            0 - 自由旋转(6种姿态)
            1 - 原始状态下只允许正面朝上, 绕Z轴旋转(2种姿态)
        
    Returns:
        允许的旋转姿态列表，格式为 (x轴对应维度, y轴对应维度, z轴对应维度)
    """

    # 所有可能的维度组合（长、宽、高对应x、y、z轴的索引）
    # 索引说明: 0=length, 1=width, 2=height
    all_rotations = [
        (0, 1, 2),  # 原始方向：长 → x，宽 → y，高 → z
        (0, 2, 1),  # 长 → x，高 → y，宽 → z
        (1, 0, 2),  # 宽 → x，长 → y，高 → z
        (1, 2, 0),  # 宽 → x，高 → y，长 → z
        (2, 0, 1),  # 高 → x，长 → y，宽 → z
        (2, 1, 0)   # 高 → x，宽 → y，长 → z
    ]
    
    # 根据业务规则返回允许的旋转姿态
    if direction == 0:
        return all_rotations  # 允许所有6种旋转
    elif direction == 1:
        # 仅允许绕Z轴旋转（保持高度方向不变，交换长和宽）
        return [all_rotations[0], all_rotations[2]]
    else:
        raise ValueError(f"无效方向编码: {direction}")