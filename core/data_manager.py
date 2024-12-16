import os
import shapefile
import numpy as np
from typing import Dict, List, Optional, Any

class DataManager:
    """管理数据集加载和建筑信息"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset_groups = self._load_dataset_groups()
        self.building_coordinates = {}

    def _load_dataset_groups(self) -> Dict[str, Dict[str, str]]:
        """加载所有数据集组"""
        groups = {}
        
        for root, _, files in os.walk(self.dataset_path):
            shp_files = {f.split('.')[0]: f for f in files if f.endswith('.shp')}
            
            base_names = set()
            for name in shp_files.keys():
                if not any(suffix in name for suffix in 
                          ['_TemplateMatching', '_RecursiveApproach', 
                           '_RectangleTrans', '_AFPM', '_Arcmap',
                           '_MLRResult', '_SVMResult', '_Result']):
                    base_names.add(name)

            for base_name in base_names:
                rel_path = os.path.relpath(root, self.dataset_path)
                group_name = os.path.join(rel_path, base_name)
                if group_name.startswith('.'):
                    group_name = group_name[2:]
                    
                groups[group_name] = {
                    'original': os.path.join(root, shp_files[base_name])
                }
        
        return groups

    def _load_building_coordinates(self, shp_path: str) -> Optional[List[Dict]]:
        """加载shp文件中的建筑坐标"""
        if shp_path not in self.building_coordinates:
            try:
                sf = shapefile.Reader(shp_path)
                shapes = sf.shapes()
                # records = sf.records()
                fields = sf.fields[1:]
                field_names = [field[0] for field in fields]
                
                coordinates = []
                is_original = any(shp_path == group['original'] 
                                for group in self.dataset_groups.values())
                
                simp_method_index = (field_names.index('SimpMethod') 
                                   if is_original and 'SimpMethod' in field_names 
                                   else None)
                
                for shape, record in zip(shapes, shapes):
                    points = shape.points
                    center_x = sum(p[0] for p in points) / len(points)
                    center_y = sum(p[1] for p in points) / len(points)
                    
                    building_info = {
                        'points': points,
                        'center': (center_x, center_y)
                    }
                                        
                    coordinates.append(building_info)
                    
                self.building_coordinates[shp_path] = coordinates
                
            except Exception as e:
                print(f"Error loading coordinates from {shp_path}: {str(e)}")
                return None
                
        return self.building_coordinates[shp_path]

    def get_building_info(self, group_name: str) -> Optional[Dict[str, List[Dict]]]:
        """获取指定数据集组的建筑信息"""
        if group_name not in self.dataset_groups:
            return None
            
        group_files = self.dataset_groups[group_name]
        # print(group_name,self.dataset_groups)
        return {
            'original': self._load_building_coordinates(group_files['original'])
        }

    def get_group_names(self) -> List[str]:
        """获取所有数据集组名称"""
        return list(self.dataset_groups.keys())

    def get_group_files(self, group_name: str) -> Optional[Dict[str, str]]:
        """获取指定数据集组的文件信息"""
        return self.dataset_groups.get(group_name) 