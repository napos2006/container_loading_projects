import numpy as np
import random
from typing import Union, Tuple, List, Dict
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from collections import defaultdict
from dataclasses import dataclass

from config.constants import AlgorithmParams, BusinessRules
from core.domain import ContainerSpec, ProductsSpec, PalletSpec, LoadingPoint
from database.converters import decode_rotations


# ----------------------------------- 三大指标衡量函数 -----------------------------------
def calculate_volume_utilization(products: list, contaniner: object, use_pallet: bool) -> float:
    """计算货箱利用率(拆分排拖情况和不排拖情况)"""
    if not use_pallet:
        # 不排拖情况，直接计算货物体积占比
        total_volume = sum(product.length * product.width * product.height for product in products)
        container_volume = contaniner.length * contaniner.width * contaniner.height
        return total_volume / container_volume
    else:
        # 排拖情况，需扣除托盘高度（托盘高度统一为150mm）
        pallent_height = 150
        effective_container_height = contaniner.height - pallent_height
        if effective_container_height <= 0:
            return 0.0

        # 利用率 = (货物体积 / 集装箱总体积(扣除托盘高度))
        total_volume = sum(product.length * product.width * product.height for product in products)
        effective_container_volume = contaniner.length * contaniner.width * effective_container_height

        return total_volume / effective_container_volume

def calculate_weight_utilization(products: list, container: object) -> float:
    """计算载重利用率"""
    total_weight = sum(products.weight for product in products)
    return total_weight / container.max_weight

def calculate_center_offset(products: list, container: object, positions: list, return_total: bool = False) -> Union[float, Tuple[float, float, float]]:
    # position: 货物左前下角坐标列表, 每个元素为 (x, y, z)
    # return_total：是否返回总偏移量，默认为 False
    """计算重心偏移量（动态传入坐标）"""
    total_weight = sum(item.weight for item in products)
    if total_weight <= 0:
        return float('inf')  # 避免除零错误
    
    weighted_x, weighted_y, weighted_z = 0.0, 0.0, 0.0

    for item, (x, y, z) in zip(products, positions):
        # 计算货物的几何中心坐标
        center_x = item.length / 2 + x
        center_y = item.width / 2 + y
        center_z = item.height / 2 + z

        # 累加加权坐标
        weighted_x += center_x * item.weight
        weighted_y += center_y * item.weight
        weighted_z += center_z * item.weight
    
    # 计算整体重心
    center_gx = weighted_x / total_weight
    center_gy = weighted_y / total_weight
    center_gz = weighted_z / total_weight

    # 安全范围约束
    x_min, x_max = container.length * 0.45, container.length * 0.55
    y_min, y_max = container.width * 0.45, container.width * 0.55
    z_min, z_max = container.height * 0.45, container.height * 0.55

    # 计算各轴偏移量
    offset_x = max(0, x_min - center_gx, center_gx - x_max)
    offset_y = max(0, y_min - center_gy, center_gy - y_max)
    offset_z = max(0, z_min - center_gz, center_gz - z_max)

    if return_total:
        return offset_x + offset_y + offset_z
    else:
        return (offset_x, offset_y, offset_z)


# ----------------------------------- 托盘选择，基于遗传算法 -----------------------------------
@dataclass
class PallentSolution:
    """托盘布局方案"""
    pallets: List[PalletSpec]
    positions: List[Tuple[float, float, float]]             #托盘在集装箱内的坐标(x, y)

class PallentOptimizerGA:
    """基于遗传算法的托盘优化器（单位统一为毫米）"""
    def __init__(
            self, 
            container: ContainerSpec, 
            products: List[ProductsSpec], 
            candidate_pallets: List[PalletSpec],
            params: dict = None):
        """
        参数：
        container: 集装箱规格
        items: 待装载货物列表
        candidate_pallets: 候选托盘规格列表
        """

        # 单位转换系数
        self.mm_to_m = 0.001

        # 配置参数
        self.container = self._convert_container(container)
        self.products = self._convert_products(products)
        self.candidate_pallets = self._convert_pallets(candidate_pallets)

        # 遗传算法参数
        self.pop_size = params.get('population_size', 50)
        self.max_gen = params.get('max_generations', 100)
        self.elite_ratio = params.get('elite_ratio', 0.2)
        self.mutation_rate = params.get('mutation_rate', 0.15)

        # 预处理货物数据(按面积降序)
        self.sorted_items = sorted(self.products, key=lambda x: x[0]*x[1], reverse=True)
    
    def _convert_container(self, container: ContainerSpec) -> dict:
        """单位转换"""
        return {
            'length': container.length,
            'width': container.width,
            'height': container.height,
            'max_weight': container.max_weight,
        }
    
    def _convert_products(self, products: List[ProductsSpec]) -> List[Tuple]:
        """单位转换"""
        return [(i.length, i.width) for i in products]
    
    def _convert_pallets(self, pallets: List[PalletSpec]) -> List[Tuple]:
        """单位转换"""
        return [(p.length, p.width) for p in pallets]
    
    def optimize(self) -> PallentSolution:
        """主优化流程"""
        # 初始化种群
        population = self._init_population()
        # 迭代优化
        for _ in range(self.max_gen):
            # 评估适应度
            evaluated = [(ind, self._fitness(ind)) for ind in population]
            evaluated.sort(key=lambda x: x[1], reverse=True)

            # 精英保留
            elites = [ind for ind, _ in evaluated[:int(self.elite_ratio * self.pop_size)]]

            # 生成新一代
            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(evaluated)
                p2 = self._tournament_select(evaluated)
                child = self._crossover(p1, p2)

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_pop.append(child)
            
            population = new_pop

        best = max(population, key=lambda x: self._fitness(x))
        return self._decode_solution(best)
    
    def _init_population(self) -> List:
        """初始化种群"""
        return [self._generate_individual() for _ in range(self.pop_size)]
    
    def _generate_individual(self) -> List:
        """生成随机个体"""
        solution = []
        used_length = 0
        max_length = self.container['length'] - self.container.get('door_reserve', 50)

        while used_length < max_length:
            # 随机选择托盘并添加间隙
            pallet = random.choice(self.candidate_pallets)
            pallet_length = pallet[0] + BusinessRules.PALLET_GAP['longitudinal']

            if used_length + pallet_length > max_length:
                break

            if len(solution) > 100:  # 防止无限循环
                break

            solution.append(pallet)
            used_length += pallet_length
        
        return solution
    
    def _fitness(self, individual: List) -> float:
        """计算适用度"""
        if not individual:          # 空方案保护
            return 0.0

        total_area = sum(p[0]*p[1] for p in individual)
        container_area = (self.container['length'] - self.container['door_reserve']) * self.container['width']
        area_ratio = total_area / container_area

        fill_ratio = self._calc_fill_ratio(individual)

        # 形状规整性评分(鼓励方形布局)
        shape_score = 0
        for p in individual:
            ratio = min(p[0] / p[1], p[1] / p[0])
            shape_score += ratio

        return 0.5 * area_ratio + 0.3 * fill_ratio + 0.2 * shape_score

    
    def _calc_fill_ratio(self, solution: List) -> float:
        """计算填充率（考虑货物间隙）"""
        total_filled = 0
        for pallet in solution:
            # 有效托盘区域（扣除边缘间隙）
            eff_length = pallet[0] - 2*BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET
            eff_width = pallet[1] - 2*BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET
            if eff_length <= 0 or eff_width <= 0:
                continue

            # 贪心填充
            filled = 0
            x, y = 0, 0
            max_row_height = 0

            for item in self.sorted_items:
                # 考虑货物间隙
                item_length = item[0] + BusinessRules.PALLET_GAP['lateral']
                item_width = item[1] + BusinessRules.PALLET_GAP['longitudinal']

                if x + item_length > eff_length:
                    x = 0
                    y += max_row_height
                    max_row_height = 0

                if y + item_width > eff_width:
                    break

                filled += item[0] * item[1]             # 实际货物面积
                x += item_length
                max_row_height = max(max_row_height, item_width)
            
            total_filled += filled
        
        return total_filled / sum(p[0]*p[1] for p in solution)
    
    def _decode_solution(self, individual: List) -> PallentSolution:                # 需要修改
        """解码为托盘布局方案"""
        positions = []
        x, y = 0, 0
        max_row_height = 0

        for pallet in individual:
            # 检查当前行剩余空间
            if x + pallet[0] > self.container['length'] - self.container.get('door_reserve', 50):
                # 换行处理
                y += max_row_height + BusinessRules.PALLET_GAP['lateral']
                x = 0
                max_row_height = 0

            positions.append((x, y))
            x += pallet[0] + BusinessRules.PALLET_GAP['longitudinal']
            max_row_height = max(max_row_height, pallet[1])
        
        return PallentSolution(pallets=[PalletSpec(id=0, length=p[0], width=p[1], height=150, max_weight=1000) for p in individual], positions=positions)
    
    def _tournament_select(self, evaluated: List, k=3) -> List:
        candidates = random.sample(evaluated, k)
        return max(candidates, key=lambda x: x[1])[0]
    
    def _crossover(self, p1: List, p2: List) -> List:
        cut = random.randint(1, min(len(p1), len(p2))-1)
        return p1[:cut] + p2[cut:]
    
    def _mutate(self, individual: List) -> List:
        idx = random.randint(0, len(individual)-1)                                                                                                           
        new_pallet = random.choice(self.candidate_pallets)
        return individual[:idx] + [new_pallet] + individual[idx+1:]

# ----------------------------------- 塔装载启发式算法 -----------------------------------
class TowerPackingAlgorithm:
    """塔装载启发式算法"""
    def __init__(
            self, 
            container: ContainerSpec, 
            products: List[ProductsSpec], 
            pallet: PalletSpec, 
            transport_type: str = 'sea',        # sea/air, 海运/空运区分标志
            cargo_type: str = 'bulk',           # pallet/bulk, 托盘/散货区分标志
            container_type: str = 'general',    # general/high, 普柜/高柜、超高柜区分标志
    ):
        """
        参数：
        container: 集装箱规格（单位：毫米）
        products: 待装载货物列表（单位：毫米）
        pallet: 托盘规格（单位：毫米）
        """
        self.container = container
        self.products = products
        self.pallet = pallet
        self.transport_type = transport_type
        self.cargo_type = cargo_type 

        # 初始化业务规则参数
        self._init_business_rules(transport_type, cargo_type, container_type)

        # 状态记录
        self.fragile_stack = defaultdict(int)
        self.positions = []
        self.current_height = 0         # 使用示例变量跟踪高度
    
    def _init_business_rules(self, transport_type: str, cargo_type: str, container_type: str):
        """根据运输场景初始化业务规则"""
        # 高度限制规则选择
        if transport_type == 'sea':
            if cargo_type == 'pallet':
                self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_SEA_TRANSPORT_IN_GENERAL_CONTAINER
                if container_type == 'high':
                    self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_SEA_TRANSPORT_IN_HIGH_CONTAINER
            else:
                self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_BULK_CARGO_OF_SEA_TRANSPORT_IN_GENERAL_CONTAINER
                if container_type == 'high':
                    self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_BULK_CARGO_OF_SEA_TRANSPORT_IN_HIGH_CONTAINER
        elif transport_type == 'air':
            self.max_height = BusinessRules.MAX_GOODS_ALTITUDE_OF_PLATOON_TOWAGE_OF_AIR_FREIGHT
        else:
            raise ValueError('运输类型无效')
        
        # 间隙限制规则选择
        self.edge_gap = BusinessRules.GAP_OF_GOODS_AND_THE_EDGE_OF_PALLET
        if cargo_type == 'bulk':
            self.pallet_gap = BusinessRules.PALLET_LIMIT_LATERAL_OF_BULK_CARGO
        
    def optimize(self) -> dict:
        """主优化流程"""
        # 按优先级排序(易碎 > 重量 > 高度)
        sorted_products = sorted(
            self.products,
            key=self._calculate_priority,
            reverse=True
        )

        for product in sorted_products:
            # 获取允许的摆放姿态
            vaild_orientations = self._get_valid_orientations(product)

            for dim in vaild_orientations:
                if self._can_place_product(product, dim):
                    self._place_product(product, dim)
                    break
        return self._build_result()
    
    def _get_valid_orientations(self, product: ProductsSpec) -> List[Tuple]:
        """生成有效货物姿态(优化性能)"""
        return [
            (dim.length, dim.width, dim.height)
            for dim in product.allowed_rotations
            if self._is_valid_dimension(dim.length, dim.width, dim.height)
        ]

    def _is_valid_dimension(self, l: int, w: int, h: int) -> bool:
        """检查尺寸是否符合业务规则（增加容差处理）"""
        eff_length = self.pallet.length - 2 * self.edge_gap + BusinessRules.SIZE_TOLERANCE
        eff_width = self.pallet.width - 2 * self.edge_gap + BusinessRules.SIZE_TOLERANCE
        return l <= eff_length and w <= eff_width

    def _can_place_product(self, product: ProductsSpec, dim: tuple) -> bool:
        """检查货物是否可以放置"""
        # 高度检查
        if self.current_height + dim[2] > self.max_height:
            return False
        # 易碎品堆叠限制
        if product.fragility in BusinessRules.FRAGILE_STACK_LIMIT:
            allowed_layers = BusinessRules.FRAGILE_STACK_LIMIT[product.fragility]

            # 特别处理非常易碎货物
            if product.fragility == 0:
                if self.fragile_stack[product.id] >= 0:
                    return False
            else:
                if self.fragile_stack[product.id] >= allowed_layers:
                    return False
        # 支撑面积检查
        if self.positions:
            last = self.positions[-1]
            min_overlap_x = min(dim[0], last['dimensions'][0]) / max(dim[0], last['dimensions'][0])
            min_overlap_y = min(dim[1], last['dimensions'][1]) / max(dim[1], last['dimensions'][1])     

            if min_overlap_x + min_overlap_y < BusinessRules.SUPPORT_AREA_MIN_LIMIT:
                return False
        return True
    
    def _place_product(self, product: ProductsSpec, dim: tuple):
        """执行放置操作(标准化数据结构)"""
        self.positions.append({
            'product_id': product.id,
            'sku': product.sku,
            'position': (self.edge_gap, self.edge_gap, self.current_height),
            'dimensions': dim,
            'fragility': product.fragility,
        })
        self.current_height += dim[2]
        if product.fragility > 0:
            self.fragile_stack[product.id] += 1

    def _calculate_priority(self, product: ProductsSpec) -> float:
        """计算货物优先级(易碎性 > 重量 > 高度)"""
        fragility = product.fragility / BusinessRules.MAX_FRAGILITY
        weight = 1 - (product.weight / max(p.weight for p in self.products))
        height = 1 - (product.height / self.max_height)
        return (fragility * 0.6) + (weight * 0.3) + (height * 0.1)
    
    def _build_result(self) -> dict:
        """构建结果（添加元数据）"""
        return {
            'transport_type': self.transport_type,
            'positions': self.positions,
            'utilization': self._calculate_utilization(),
            'stability': self._check_stability(),
            'fragile_counts': dict(self.fragile_stack),
            'total_height': self.current_height
        }
    
    def _calculate_utilization(self) -> float:
        """体积利用率计算（增加防零除保护）"""
        used = sum(d[0]*d[1]*d[2] for _,d in self.positions)
        available = (self.pallet.length-2*self.edge_gap) * (self.pallet.width-2*self.edge_gap) * self.max_height
        return used / available if available > 0 else 0
    
    def _check_stability(self) -> dict:
        """稳定性检查(包含海运/空运特殊规则)"""
        total_weight = sum(p['product'].weight for p in self.positions)
        if total_weight == 0:
            return {'stable': False, 'reason': 'no_products'}
        
        # 计算重心坐标
        weighted_x = sum((p['position'][0] + p['dimensions'][0]/2) * p['product'].weight for p in self.positions)
        weighted_y = sum((p['position'][1] + p['dimensions'][1]/2) * p['product'].weight for p in self.positions)
        center_x = weighted_x / total_weight
        center_y = weighted_y / total_weight
        
        # 动态阈值
        if self.transport_type == 'sea':
            x_limit = self.container.length * BusinessRules.SEA_OFFSET_LIMIT
            y_limit = self.container.width * BusinessRules.SEA_OFFSET_LIMIT
        else:
            x_limit = self.container.length * BusinessRules.AIR_OFFSET_LIMIT
            y_limit = self.container.width * BusinessRules.AIR_OFFSET_LIMIT
        
        return {
            'stable': abs(center_x) < x_limit and abs(center_y) < y_limit,
            'actual_offset': (center_x, center_y),
            'allowed_offset': (x_limit, y_limit),
            'threshold': BusinessRules.SEA_OFFSET_LIMIT if self.transport_type == 'sea' else BusinessRules.AIR_OFFSET_LIMIT
        }

# ----------------------------------- 二维装载点优化算法 -----------------------------------
class BinPacking2D:
    """改进的二维装载优化器（支持间隙规则和智能优化）"""
    def __init__(self, container_width: float, container_height: float, global_offset: Tuple[float, float] = (0, 0)):
        """
        参数：
        container_width: 装载区域宽度（单位：毫米）
        container_height: 装载区域高度（单位：毫米）
        global_offset: 全局全局坐标偏移量（单位：毫米, 用于结果转换）
        """
        self.width = container_width
        self.height = container_height
        self.global_offset = global_offset

        # 初始化装载点(考虑间隙规则)
        self.gap_left = BusinessRules.PALLET_GAP_CONTAINER['left']
        self.gap_right = BusinessRules.PALLET_GAP_CONTAINER['right']
        self.gap_front = BusinessRules.PALLET_GAP_CONTAINER['front']
        self.gap_back = BusinessRules.PALLET_GAP_CONTAINER['back']    
    def initalize_ponits(self):
        """步骤1: 初始化装载点"""
        # 初始左下角点(考虑货物放置间隙)

    def select_best_point(self) -> LoadingPoint:
        """步骤2: 选择最佳装载点"""

    def cluster_points(self):
        """步骤3: 基于密度的装载点聚类"""
    
    def _split_space(self, point: LoadingPoint, used_w: float, used_h: float):
        """分裂剩余空间生成新候选点"""










# ----------------------------------- 分层优化控制器 -----------------------------------
class HierarchicalOptimizer:
    """协同优化控制器(支持遗传算法+塔装载+二维优化的分层协调)"""
    def __init__(self, container: ContainerSpec, products: List[ProductsSpec], candidate_pallets: List[PalletSpec]):
        """
        参数：
        container: 集装箱规格（单位：毫米）
        products: 待装载货物列表（单位：毫米）
        candidate_pallets: 可选托盘类型列表（单位：毫米）
        """
        self.container = container
        self.products = products
        self.candidate_pallets = candidate_pallets

        # 优化状态记录
        self.best_solution = None
        self.optimization_history = []

    def optimize(self):
        """执行分层优化流程"""
        # 阶段1： 遗传算法生成初始托盘布局
        ga_optimizer = PallentOptimizerGA(
            container=self.container,
            products=self.products,
            candidate_pallets=self.candidate_pallets

        )
        pallet_solution = ga_optimizer.optimize()

        # 阶段2： 并行优化各托盘装载方案
        tower_solutions = {}
        for pallet, pos in zip(pallet_solution.pallets, pallet_solution.positions):
            # 分配货物到当前托盘
            assigned_products = self._assign_products(pallet, pos)

            # 塔装载优化
            tower_packer = TowerPackingAlgorithm(
                container=self.container,
                products=assigned_products,
                pallet=pallet,
                position=pos        # 托盘全局位置
            )
            tower_sol = tower_packer.optimize()

            # 二维装载优化
            bin_packer = BinPacking2D(
                container_width=pallet.length,
                container_height=pallet.width,
                global_offset=(pos[0], pos[1])              # 全局坐标偏移
            )
            bin_sol = bin_packer.hybrid_optimize(tower_sol['local_positions'])                                      # 插眼，参数可能存在问题

            tower_solutions[pallet.id] = {
                'tower': tower_sol,
                'bin': bin_sol
            }
        
        # 阶段3：全局优化调整
        return self._global_refinement(pallet_solution, tower_solutions)
    
    def _assign_products(self, pallet: PalletSpec, position: Tuple) -> List[ProductsSpec]:
        """智能分配货物到指定托盘"""
        max_w = pallet.length - 2 * BusinessRules.PALLET_GAP_CONTAINER['front']
        max_h = pallet.width - 2 * BusinessRules.PALLET_GAP_CONTAINER['left']

        return [p for p in self.products if p.length <= max_w and p.width <= max_h]

    def _global_refinement(self, pallet_solution, tower_solutions) -> dict:
        """全局优化调整(预留群智能算法接口)"""
        # 当前方案评估
        current_score = self._evaluate_solution(pallet_solution, tower_solutions)

        # 

        # 生成最终结果
        return self._build_final_result()
    
    def _evaluate_solution(self, pallet_solution, tower_solutions) -> float:
        """方案评估（体积利用率 + 稳定性）"""
        total_volume = sum(t['bin']['total_volume'] for t in tower_solutions.values())
        container_volume = self.container.volume()
        return total_volume / container_volume
    
    def _build_final_result(self, solution) -> dict:
        """构建全局解决方案"""
        return {
            'pallets': solution['pallet_layout'],
            'utilization': solution['utilization'],
            'stability': solution['stability'],
        }











# ----------------------------------- ACO + PSO + SA + NSGAII -----------------------------------
class ACO:
    def __init__(self):
        self,num_ants = AlgorithmParams.ACO_ANTS_NUM
    
    def optimize(self, containr: ContainerSpec, products: ProductsSpec, pallet: PalletSpec):
        pass


class PSO:
    pass



class SA:
    pass



class NSGAII:
    def __init__(self):
        pass

    def multi_objective_optimization(solutions: list, container: object, use_pallet: bool):
        """多目标优化：计算每个解的容积、载重、重心偏移目标值"""
        for sol in solutions:
            # 假设 sol 包含 items（货物属性）和 positions（坐标列表）
            sol.volume_util = calculate_volume_utilization(
                items=sol.items,
                container=container,
                use_pallet=use_pallet
            )
            sol.weight_util = calculate_weight_utilization(
                items=sol.items,
                container=container
            )
            sol.center_offset = calculate_center_offset(
                products=sol.items,
                positions=sol.positions,  # 关键：传入动态生成的坐标
                container=container,
                return_total=True
            )
        return solutions