"""
此文件用于从数据库加载数据
"""

from typing import Tuple, List
from database.db_manager import DatabaseManager
from database.converters import convert_container_to_spec, convert_pallet_to_spec, convert_product_to_spec
from database.models import Container, Product, Pallet

def load_data(container_id: int) -> Tuple[Container, List[Product], List[Pallet]]:
    db = DatabaseManager()
    session = db.get_session()

    # 加载数据库对象
    db_container = session.query(Container).get(container_id)
    db_products = session.query(Product).all()
    db_pallets = session.query(Pallet).all()

    # 转换为算法模型
    container_spec = convert_container_to_spec(db_container)
    products = [convert_product_to_spec(prod) for prod in db_products]
    pallets = [convert_pallet_to_spec(p) for p in db_pallets]

    return container_spec, products, pallets