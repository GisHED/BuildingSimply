import os,json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import shapefile

from .data_manager import DataManager
from .visualization import Visualizer
from regularization_bart import Regular_Bart
from polygon_evaluator import SimplePolygonEvaluator
from shapely.geometry import Polygon

def save_evaluation_results(save_path, evaluator_results):
    """保存评估结果到缓存文件
    
    Args:
        method: 评估方法名称
        evaluator_results: 评估器结果字典
    """
    # 创建缓存目录
    cache_dir = os.path.dirname(save_path)
    os.makedirs(cache_dir, exist_ok=True)
    
    # 保存评估结果到缓存文件
    evaluation_results = {
        'area_changes': evaluator_results['area_changes'],
        'area_ious': evaluator_results['area_ious'],
        'perimeter_ratios': evaluator_results['perimeter_ratios'], 
        'point_ratios': evaluator_results['point_ratios'],
        'hausdorffs': evaluator_results['hausdorffs'],
        'shape_regularity_percent': evaluator_results['shape_regularity_percent'],
        'shape_regularity_deviation': evaluator_results['shape_regularity_deviation'],
        'edge_fragmentation_percent': evaluator_results['edge_fragmentation_percent'],
        'edge_fragmentation_ratio': evaluator_results['edge_fragmentation_ratio'],
        'original_points': evaluator_results['original_points'],
        'simplified_points': evaluator_results['simplified_points']
    }
    
    print(f'保存评估结果到：{save_path}')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f)


class DatasetVisualizer:
    """数据集可视化的主类"""
    
    def __init__(self):
        # 初始化路径
        self.dataset_path = "./dataset/shp/"
        self.cache_dir = "./cache/regularization" 
        self.record_dir = "./records"
        
        # 创建必要的目录
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)
        
        # 初始化组件
        self.data_manager = DataManager(self.dataset_path)
        self.visualizer = Visualizer()
        
        # 初始化模型
        self.regular_bart = Regular_Bart()

        
        # 初始化状态
        self.current_data = {}
        self.cached_params = {
            'center_idx': None,
            'num_neighbors': None,
            'display_buildings': None
        }

    def update_visualization(self, group_name: str) -> Tuple[Optional[Image.Image], Optional[Dict]]:
        """更新可视化结果"""
        if group_name not in self.data_manager.dataset_groups:
            return None, None
            
        if (self.current_data.get('group_name') == group_name and 
            'visualization_results' in self.current_data):
            return (self.current_data['visualization_results']['original'],
                    self.current_data)
        
        group_files = self.data_manager.get_group_files(group_name)
        building_info = self.data_manager.get_building_info(group_name)
        
        self.current_data = {
            'group_name': group_name,
            'building_info': building_info
        }
        
        results = {
            'original': self.visualizer.visualize_shp(group_files['original'])
        }
        
        self.current_data['visualization_results'] = results
        return results['original'], self.current_data

    def create_merged_visualization(self, 
                                  group_name: str,
                                  center_idx: int = 0,
                                  num_neighbors: int = 50,
                                  iou_threshold: float = 0.5,
                                  regularity_threshold: float = 1.0,
                                  fragmentation_threshold: float = 1.0) -> Tuple:
        """创建合并的可视化结果"""
        building_info = None
        
        # 检查是否需要重新加载数据
        if not self.current_data or self.current_data.get('group_name') != group_name:
            building_info = self.data_manager.get_building_info(group_name)
            self.current_data = {
                'group_name': group_name,
                'building_info': building_info,
                'matched_buildings': None
            }
            # 清除缓存
            self.cached_params = {
                'center_idx': None,
                'num_neighbors': None,
                'display_buildings': None
            }
        else:
            building_info = self.current_data['building_info']
        
        if building_info is None:
            return None, None, None, None
        
        # 检查是否需要重新计算邻近建筑
        params_changed = (self.cached_params['center_idx'] != center_idx or 
                         self.cached_params['num_neighbors'] != num_neighbors)
        
        if self.current_data.get('matched_buildings') is None:
            # 创建匹配结果字典
            matched_buildings = {
                'original': building_info['original'],
                'regular_bart': [None] * len(building_info['original']),
                'regular_urban': [None] * len(building_info['original']),
                'regular_bart_10Types': [None] * len(building_info['original']),
            }
            
            # 处理每个建筑
            for orig_idx, orig_building in enumerate(building_info['original']):
                if orig_idx % 10 == 0:
                    print(f"Processing building {orig_idx + 1} of {len(building_info['original'])}")
                
                # 处理规则化结果
                for method in ['regular_bart']:
                    cached_results = self._load_regularization_results(group_name, method)
                    if cached_results is None:
                        results = [self._process_regularization(b, method) 
                                 for b in building_info['original']]
                        self._save_regularization_results(group_name, method, results,building_info['original'])
                        matched_buildings[method] = results
                    else:
                        matched_buildings[method] = cached_results
            
            # 创建评估器
            self.current_data['evaluators'] = {}
            for method in ['regular_bart']:
                if method in matched_buildings and matched_buildings[method][0] is not None:
                    print(f'评估器创建：{method}')
                    evaluator = SimplePolygonEvaluator(
                        matched_buildings['original'],
                        matched_buildings[method]
                    )
                    save_evaluation_path = f'./cache/regularization/{group_name}_{method}_evaluation_results.json'
                    save_evaluation_results(save_evaluation_path, evaluator.results)                    
                    self.current_data['evaluators'][method] = evaluator
            
            # 选择最佳结果
            matched_buildings['runtime_ev'] = [None] * len(matched_buildings['original'])
            method_list = []
            
            for idx, orig_building in enumerate(matched_buildings['original']):
                if orig_building is None:
                    continue
                
                best_method = None
                best_regularity = float('inf')
                best_building = None
                best_iou = 0
                
                for method, evaluator in self.current_data['evaluators'].items():
                    results = evaluator.results
                    if (idx < len(results['area_ious']) and 
                        idx < len(results['shape_regularity_percent']) and 
                        idx < len(results['edge_fragmentation_percent'])):
                        
                        iou = results['area_ious'][idx]
                        regularity = results['shape_regularity_percent'][idx]
                        fragmentation = results['edge_fragmentation_percent'][idx]
                        
                        if (iou > iou_threshold and 
                            regularity < regularity_threshold and 
                            fragmentation < fragmentation_threshold and 
                            iou > best_iou):
                            best_iou = iou
                            best_method = method
                            best_building = matched_buildings[method][idx]
                
                if best_building is not None:
                    method_list.append(best_method)
                    matched_buildings['runtime_ev'][idx] = best_building
            # 保存runtime_ev结果到shapefile
            self._save_to_shapefile(group_name, 'runtime_ev', matched_buildings['runtime_ev'])
            
            from collections import Counter
            print(f'方法选择：{Counter(method_list)}')
            
            # 为runtime_ev创建评估器
            print(f'评估器创建：runtime_ev')
            self.current_data['evaluators']['runtime_ev'] = SimplePolygonEvaluator(
                matched_buildings['original'],
                matched_buildings['runtime_ev']
            )
            
            # 保存匹配结果
            self.current_data['matched_buildings'] = matched_buildings
            
            # 记录结果
            self._save_evaluation_records(group_name, matched_buildings)
            
            # 清除缓存
            self.cached_params['display_buildings'] = None
            params_changed = True
        
        if params_changed:
            # 获取显示范围内的建筑
            total_buildings = len(self.current_data['matched_buildings']['original'])
            center_idx = min(max(0, center_idx), total_buildings - 1)
            
            # 计算距离并选择邻近建筑
            center_point = np.array(self.current_data['matched_buildings']['original'][center_idx]['center'])
            distances = [np.linalg.norm(np.array(b['center']) - center_point) 
                        for b in self.current_data['matched_buildings']['original']]
            neighbor_indices = np.argsort(distances)[:num_neighbors]
            
            # 创建显示子集
            self.cached_params['display_buildings'] = {
                method: [buildings[i] for i in neighbor_indices] 
                for method, buildings in self.current_data['matched_buildings'].items()
            }
            
            # 更新缓存参数
            self.cached_params['center_idx'] = center_idx
            self.cached_params['num_neighbors'] = num_neighbors
        
        display_buildings = self.cached_params['display_buildings']
        
        # 计算显示范围
        all_points = []
        for building in display_buildings['original']:
            if building is not None:
                all_points.extend(building['points'])
        all_points = np.array(all_points)
        
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        x1, y1 = min_coords
        shape_x = 2400
        shape_y = int(shape_x * (max_coords[1] - min_coords[1]) / 
                     (max_coords[0] - min_coords[0]))
        scale = (shape_x/(max_coords[0]-min_coords[0]), 
                shape_y/(max_coords[1]-min_coords[1]))
        
        # 创建可视化结果
        original_img = self.visualizer.create_visualization(
            display_buildings['original'],
            x1, y1, scale, shape_x, shape_y
        )
        
        merged_img = self.visualizer.create_visualization(
            display_buildings['runtime_ev'],
            x1, y1, scale, shape_x, shape_y
        )
        
        return (Image.fromarray(original_img),
                Image.fromarray(merged_img),
                display_buildings,
                self.current_data['evaluators'])

    def _process_regularization(self, building: Dict, method: str) -> Optional[Dict]:
        """处理单个建筑的规则化"""
        if building is None:
            return None
        
        try:
            # 序列化建筑点
            points = np.array(building['points'])
            from regularization_utitls import serialize
            serialize_data = serialize(points)
            vec = serialize_data['indexs']
            vec_str = ' '.join(map(str, vec))
            
            # 根据方法选择模型
            if method == 'regular_bart':
                result_str = self.regular_bart.generate(vec_str)
            elif method == 'regular_bart_10Types':
                result_str = self.regular_bart.generate(vec_str)
            elif method == 'regular_afpm':
                result_str = self.regular_bart.generate(vec_str)
            else:  # regular_urban
                result_str = self.regular_bart.generate(vec_str)
            
            # 转换结果
            result_vec = [int(i) for i in result_str.split()]
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            result_points = [[v % 96, v // 96] for v in result_vec]
            result_points = np.array(result_points) / 95 * [x_max - x_min, y_max - y_min] + [x_min, y_min]
            
            return {
                'points': result_points.tolist(),
                'center': building['center']
            }
        except Exception as e:
            print(f"Error processing building with {method}: {str(e)}")
            return None

    def _save_regularization_results(self, group_name: str, method: str, results: List[Dict], original_buildings: List[Dict]):
        """保存规则化结果到缓存和shp文件"""
        # 保存缓存JSON
        cache_path = os.path.join(self.cache_dir, f"{group_name}_{method}.json")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        serialized_results = []
        for building, original_building in zip(results, original_buildings):
            if building is not None:
                serialized_results.append({
                    'points': building['points'],
                    'center': building['center'],
                    'original_points': original_building['points'],
                    'original_center': original_building['center']
                })
            else:
                serialized_results.append(None)
        
        with open(cache_path, 'w') as f:
            json.dump(serialized_results, f)
        
        # 保存为shp文件
        self._save_to_shapefile(group_name, method, results)

    def _save_to_shapefile(self, group_name: str, method: str, results: List[Dict]):
        """将规则化结果保存为shp文件
        
        Args:
            group_name: 组名（用于构建文件路径）
            method: 规则化方法名称
            results: 规则化结果列表
        """
        try:
            # 构建保存路径
            # 从group_name中提取目录结构
            base_dir = "dataset/shp/remote_pic_shp/vector_files"
            save_dir = os.path.dirname(os.path.join(base_dir, group_name))
            
            # 获取原始文件名（不含扩展名）和新文件名
            base_name = os.path.splitext(os.path.basename(group_name))[0]
            new_name = f"{base_name}_{method}"
            
            # 构建完整的输出路径
            output_path = os.path.join(save_dir, new_name)
            print(f'Saving {method} result to: {output_path}')
            
            # 删除已存在的相关文件
            for ext in ['.shp', '.shx', '.dbf']:
                existing_file = output_path + ext
                if os.path.exists(existing_file):
                    os.remove(existing_file)
            
            # 创建新的shapefile
            w = shapefile.Writer(output_path)
            w.field('name', 'C')  # 添加一个名称字段
            
            # 写入建筑物数据
            for i, building in enumerate(results):
                if building is not None and 'points' in building:
                    # 确保多边形闭合
                    points = building['points']
                    if points[0] != points[-1]:
                        points = points + [points[0]]
                    
                    # 写入多边形
                    w.poly([points])
                    w.record(f'Building_{i}')
            
            w.close()
            print(f'Successfully saved {method} result to shapefile')
            
        except Exception as e:
            print(f"Error saving {method} result to shapefile: {str(e)}")

    def _load_regularization_results(self, group_name: str, method: str) -> Optional[List[Dict]]:
        """从缓存加载规则化结果"""
        cache_path = os.path.join(self.cache_dir, f"{group_name}_{method}.json")
        if os.path.exists(cache_path):
            import json
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def _save_evaluation_records(self, group_name: str, matched_buildings: Dict):
        """保存评估记录"""
        record_data = {"data": []}
        for orig, ev in zip(matched_buildings['original'], 
                           matched_buildings['runtime_ev']):
            if orig is not None and ev is not None:
                from regularization_utitls import serialize
                orig_serial = serialize(np.array(orig['points']))
                ev_serial = serialize(np.array(ev['points']))
                record_data["data"].append({
                    "text": " ".join(map(str, orig_serial['indexs'])),
                    "title": " ".join(map(str, ev_serial['indexs']))
                })
        
        record_path = os.path.join(self.record_dir, f"{group_name}_records.json")
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        import json
        with open(record_path, 'w') as f:
            json.dump(record_data, f, indent=2)
        

class DatasetVisualizerSingle:
    """处理单个shp文件的可视化类"""
    
    def __init__(self):
        # 初始化路径
        self.cache_dir = "./cache/regularization" 
        self.record_dir = "./records"
        
        # 创建必要的目录
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)
        
        # 初始化模型
        self.regular_bart = Regular_Bart()
        self.regular_bart_10Types = Regular_Bart()
        
        # 初始化状态
        self.current_data = {}
        self.cached_params = {
            'center_idx': None,
            'num_neighbors': None,
            'display_buildings': None
        }
        
        # 初始化可视化器
        self.visualizer = Visualizer()

    def _load_building_coordinates(self, shp_path):
        """加载shp文件中所有建筑的坐标"""
        try:
            sf = shapefile.Reader(shp_path)
            shapes = sf.shapes()
            coordinates = []
            
            for shape in shapes:
                points = shape.points
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                
                building_info = {
                    'points': points,
                    'center': (center_x, center_y)
                }
                coordinates.append(building_info)
                
            return coordinates
        except Exception as e:
            print(f"Error loading coordinates from {shp_path}: {str(e)}")
            return None

    def create_merged_visualization(self, 
                                  shp_path: str,
                                  gt_path: str = None,
                                  center_idx: int = 0,
                                  num_neighbors: int = 50,
                                  iou_threshold: float = 0.5,
                                  regularity_threshold: float = 1.0,
                                  fragmentation_threshold: float = 1.0) -> Tuple:
        """创建合并的可视化结果
        
        Args:
            shp_path: 原始shp文件路径
            gt_path: 真值shp文件路径（可选）
            center_idx: 中心建筑索引
            num_neighbors: 显示的邻近建筑数量
            iou_threshold: IOU阈值
            regularity_threshold: 形状规则度阈值
            fragmentation_threshold: 边缘碎片度阈值
            
        Returns:
            Tuple[Image, Image, Image, Dict, Dict]: 原始图像、真值图像、合并图像、显示建筑信息、评估器
        """
        # 使用文件名作为group_name
        # 获取shp_path相对位置
        rel_path = os.path.relpath(shp_path, "dataset/shp/remote_pic_shp/vector_files")
        group_name = rel_path #os.path.join(rel_path, os.path.splitext(os.path.basename(shp_path))[0])
        
        # 检查是否需要重新加载数据
        if not self.current_data or self.current_data.get('shp_path') != shp_path:
            building_info = self._load_building_coordinates(shp_path)
            self.current_data = {
                'shp_path': shp_path,
                'group_name': group_name,
                'building_info': {'original': building_info},
                'matched_buildings': None
            }
            # 清除缓存
            self.cached_params = {
                'center_idx': None,
                'num_neighbors': None,
                'display_buildings': None
            }
        
        if self.current_data['building_info']['original'] is None:
            return None, None, None, None, None
        
        # 检查是否需要重新计算邻近建筑
        params_changed = (self.cached_params['center_idx'] != center_idx or 
                         self.cached_params['num_neighbors'] != num_neighbors)
        
        if self.current_data.get('matched_buildings') is None:
            # 创建匹配结果字典
            matched_buildings = {
                'original': self.current_data['building_info']['original'],
                'regular_bart': [None] * len(self.current_data['building_info']['original']),
                'regular_afpm': [None] * len(self.current_data['building_info']['original']),
                'regular_urban': [None] * len(self.current_data['building_info']['original']),
                'regular_bart_10Types': [None] * len(self.current_data['building_info']['original']),
            }
            
            # 加载真值数据
            if gt_path and os.path.exists(gt_path):
                try:
                    gt_buildings = self._load_building_coordinates(gt_path)
                    # if gt_buildings and len(gt_buildings) == len(matched_buildings['original']):
                    if gt_buildings: #真值中可能有部分建筑没检测出来，也可能有误检因此需要合并
                        matched_buildings['gt'] = [None] * len(matched_buildings['original'])
                        # 对每个真值建筑找到最近的原始建筑并计算IoU
                        for gt_idx, gt_building in enumerate(gt_buildings):
                            if gt_building is None:
                                continue
                            gt_center = np.array(gt_building['center'])                            
                            # 计算到所有原始建筑的距离
                            distances = []
                            for orig_idx, orig_building in enumerate(matched_buildings['original']):
                                if orig_building is None:
                                    distances.append(float('inf'))
                                    continue
                                orig_center = np.array(orig_building['center'])
                                dist = np.linalg.norm(gt_center - orig_center)
                                distances.append(dist)
                            
                            # 获取最近的原始建筑
                            closest_orig_idx = np.argmin(distances)
                            closest_orig = matched_buildings['original'][closest_orig_idx]
                            
                            if closest_orig is not None:
                                # 计算IoU
                                source_poly = Polygon(np.array(gt_building['points']))
                                target_poly = Polygon(np.array(closest_orig['points']))
                                
                                if not source_poly.is_valid:
                                    source_poly = source_poly.buffer(0)
                                if not target_poly.is_valid:
                                    target_poly = target_poly.buffer(0)
                                    
                                if source_poly.is_valid and target_poly.is_valid:
                                    intersection = source_poly.intersection(target_poly).area
                                    union = source_poly.union(target_poly).area
                                    iou = intersection / union if union > 0 else 0
                               
                                # 如果IoU大于阈值，则认为匹配成功
                                if iou > 0.6:
                                    matched_buildings['gt'][closest_orig_idx] = gt_building
                        # matched_buildings['gt'] = [b for b in matched_buildings['gt'] if b is not None]
                        print(f'真值匹配成功：{len(matched_buildings["gt"])},matched_buildings["gt"]:{matched_buildings["gt"][:5]}')
                            
                        # 对每个原始建筑找到最近的真值建筑
                        # for orig_idx, orig_building in enumerate(matched_buildings['original']):
                        #     if orig_building is None:
                        #         continue
                        #     orig_center = np.array(orig_building['center'])
                        #     # 计算到所有真值建筑的距离
                        #     distances = [np.linalg.norm(orig_center - np.array(b['center'])) 
                        #                for b in gt_buildings]
                        #     best_match_idx = np.argmin(distances)
                        #     matched_buildings['gt'][orig_idx] = gt_buildings[best_match_idx]
                        print(f'Successfully loaded and matched ground truth from: {gt_path}')
                    else:
                        print(f'Warning: Ground truth building count mismatch in {gt_path}')
                except Exception as e:
                    print(f'Error loading ground truth: {str(e)}')
            
            # 处理每个建筑
            for orig_idx, orig_building in enumerate(matched_buildings['original']):
                if orig_idx % 10 == 0:
                    print(f"Processing building {orig_idx + 1} of {len(matched_buildings['original'])}")
                
                # 处理规则化结果
                for method in ['regular_bart']:
                    cached_results = self._load_regularization_results(group_name, method)
                    if cached_results is None:
                        results = [self._process_regularization(b, method) 
                                 for b in matched_buildings['original']]
                        self._save_regularization_results(group_name, method, results, matched_buildings['original'])
                        matched_buildings[method] = results
                    else:
                        matched_buildings[method] = cached_results
                
            # 处理本地方法结果
            local_methods = ['FrameField', 'DPNN']
            for method in local_methods:
                matched_buildings[method] = [None] * len(matched_buildings['original'])
                method_path = os.path.splitext(shp_path)[0] + f'_{method}.shp'
                if os.path.exists(method_path):
                    try:
                        method_buildings = self._load_building_coordinates(method_path)
                        if method_buildings and len(method_buildings) == len(matched_buildings['original']):
                            # 对每个原始建筑找到最近的匹配建筑
                            for orig_idx, orig_building in enumerate(matched_buildings['original']):
                                if orig_building is None:
                                    continue
                                orig_center = np.array(orig_building['center'])
                                # 计算到所有目标建筑的距离
                                distances = [np.linalg.norm(orig_center - np.array(b['center'])) 
                                           for b in method_buildings]
                                best_match_idx = np.argmin(distances)
                                matched_buildings[method][orig_idx] = method_buildings[best_match_idx]
                            print(f'Successfully loaded and matched {method} results from: {method_path}')
                        else:
                            print(f'Warning: {method} results count mismatch in {method_path}')
                    except Exception as e:
                        print(f'Error loading {method} results: {str(e)}')
            
            # 创建评估器
            self.current_data['evaluators'] = {}
            all_methods = ['regular_bart'] + local_methods
            # if 'gt' in matched_buildings:
            #     all_methods.append('gt')
            
            for method in all_methods:
                if method in matched_buildings and matched_buildings[method][0] is not None:
                    print(f'评估器创建：{method}')
                    gt_buildings = matched_buildings['gt'] if 'gt' in matched_buildings else matched_buildings['original']
                    if 'gt' in matched_buildings: print(f'!!!!!!!!!!!使用真值进行评估!!!!')
                    evaluator = SimplePolygonEvaluator(
                        gt_buildings,
                        matched_buildings[method]
                    )
                    save_evaluation_path = f'./cache/regularization/{group_name}_{method}_evaluation_results.json'
                    save_evaluation_results(save_evaluation_path, evaluator.results)
                    self.current_data['evaluators'][method] = evaluator
            
            # 选择最佳结果
            matched_buildings['runtime_ev'] = [None] * len(matched_buildings['original'])
            method_list = []
            
            for idx, orig_building in enumerate(matched_buildings['original']):
                if orig_building is None:
                    continue
                
                best_method = None
                best_regularity = float('inf')
                best_building = None
                best_iou = 0
                
                for method, evaluator in self.current_data['evaluators'].items():
                    results = evaluator.results
                    if (idx < len(results['area_ious']) and 
                        idx < len(results['shape_regularity_percent']) and 
                        idx < len(results['edge_fragmentation_percent'])):
                        
                        iou = results['area_ious'][idx]
                        regularity = results['shape_regularity_percent'][idx]
                        fragmentation = results['edge_fragmentation_percent'][idx]
                        
                        if (iou > iou_threshold and 
                            regularity < regularity_threshold and 
                            fragmentation < fragmentation_threshold and 
                            iou > best_iou):
                            best_iou = iou
                            best_method = method
                            best_building = matched_buildings[method][idx]
                
                if best_building is not None:
                    method_list.append(best_method)
                    matched_buildings['runtime_ev'][idx] = best_building
            
            # 保存runtime_ev结果到shapefile
            self._save_to_shapefile(group_name, 'runtime_ev', matched_buildings['runtime_ev'])
            
            from collections import Counter
            print(f'方法选择：{Counter(method_list)}')
            
            # 为runtime_ev创建评估器
            print(f'评估器创建：runtime_ev')
            self.current_data['evaluators']['runtime_ev'] = SimplePolygonEvaluator(
                gt_buildings,
                matched_buildings['runtime_ev']
            )
            # 保存runtime_ev评估结果
            save_evaluation_path = f'./cache/regularization/{group_name}_runtime_ev_evaluation_results.json'
            save_evaluation_results(save_evaluation_path, self.current_data['evaluators']['runtime_ev'].results)
            # 保存匹配结果
            self.current_data['matched_buildings'] = matched_buildings
            
            # 记录结果
            self._save_evaluation_records(group_name, matched_buildings)
            
            # 清除缓存
            self.cached_params['display_buildings'] = None
            params_changed = True
        
        if params_changed:
            self._update_display_buildings(center_idx, num_neighbors)
        
        display_buildings = self.cached_params['display_buildings']
        
        # 创建可视化结果
        original_img, gt_img, merged_img = self._create_visualization_images(display_buildings)
        
        return (Image.fromarray(original_img),
                Image.fromarray(gt_img),
                Image.fromarray(merged_img),
                display_buildings,
                self.current_data['evaluators'])

    def _update_display_buildings(self, center_idx: int, num_neighbors: int):
        """更新显示范围内的建筑"""
        total_buildings = len(self.current_data['matched_buildings']['original'])
        center_idx = min(max(0, center_idx), total_buildings - 1)
        
        # 计算距离并选择邻近建筑
        center_point = np.array(self.current_data['matched_buildings']['original'][center_idx]['center'])
        distances = [np.linalg.norm(np.array(b['center']) - center_point) 
                    for b in self.current_data['matched_buildings']['original']]
        neighbor_indices = np.argsort(distances)[:num_neighbors]
        
        # 创建显示子集
        self.cached_params['display_buildings'] = {
            method: [buildings[i] for i in neighbor_indices] 
            for method, buildings in self.current_data['matched_buildings'].items()
        }
        
        # 更新缓存参数
        self.cached_params['center_idx'] = center_idx
        self.cached_params['num_neighbors'] = num_neighbors

    def _create_visualization_images(self, display_buildings):
        """创建可视化图像"""
        # 计算显示范围
        all_points = []
        for building in display_buildings['original']:
            if building is not None:
                all_points.extend(building['points'])
        all_points = np.array(all_points)
        
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        x1, y1 = min_coords
        shape_x = 2400
        shape_y = int(shape_x * (max_coords[1] - min_coords[1]) / 
                     (max_coords[0] - min_coords[0]))
        scale = (shape_x/(max_coords[0]-min_coords[0]), 
                shape_y/(max_coords[1]-min_coords[1]))
        
        # 创建可视化结果
        original_img = self.visualizer.create_visualization(
            display_buildings['original'],
            x1, y1, scale, shape_x, shape_y
        )
        
        # 创建真值图像（如果有）
        gt_img = np.zeros((shape_y, shape_x, 3), dtype=np.uint8)
        if 'gt' in display_buildings and any(b is not None for b in display_buildings['gt']):
            gt_img = self.visualizer.create_visualization(
                display_buildings['gt'],
                x1, y1, scale, shape_x, shape_y
            )
        
        merged_img = self.visualizer.create_visualization(
            display_buildings['runtime_ev'],
            x1, y1, scale, shape_x, shape_y
        )
        
        return original_img, gt_img, merged_img

    # 保留其他必要的辅助方法，但简化为只处理单个shp文件
    _process_regularization = DatasetVisualizer._process_regularization
    _save_regularization_results = DatasetVisualizer._save_regularization_results
    _load_regularization_results = DatasetVisualizer._load_regularization_results
    _save_evaluation_records = DatasetVisualizer._save_evaluation_records

    def _save_to_shapefile(self, group_name: str, method: str, results: List[Dict]):
        """将规则化结果保存为shp文件
        
        Args:
            group_name: 组名（用于构建文件路径）
            method: 规则化方法名称
            results: 规则化结果列表
        """
        try:
            # 构建保存路径
            # 从group_name中提取目录结构
            base_dir = "dataset/shp/remote_pic_shp/vector_files"
            save_dir = os.path.dirname(os.path.join(base_dir, group_name))
            
            # 获取原始文件名（不含扩展名）和新文件名
            base_name = os.path.splitext(os.path.basename(group_name))[0]
            new_name = f"{base_name}_{method}"
            
            # 构建完整的输出路径
            output_path = os.path.join(save_dir, new_name)
            print(f'Saving {method} result to: {output_path}')
            
            # 删除已存在的相关文件
            for ext in ['.shp', '.shx', '.dbf']:
                existing_file = output_path + ext
                if os.path.exists(existing_file):
                    os.remove(existing_file)
            
            # 创建新的shapefile
            w = shapefile.Writer(output_path)
            w.field('name', 'C')  # 添加一个名称字段
            
            # 写入建筑物数据
            for i, building in enumerate(results):
                if building is not None and 'points' in building:
                    # 确保多边形闭合
                    points = building['points']
                    if points[0] != points[-1]:
                        points = points + [points[0]]
                    
                    # 写入多边形
                    w.poly([points])
                    w.record(f'Building_{i}')
            
            w.close()
            print(f'Successfully saved {method} result to shapefile')
            
        except Exception as e:
            print(f"Error saving {method} result to shapefile: {str(e)}")
        