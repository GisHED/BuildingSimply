import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
from viusual_shp import fullpart_show_shp_building
import shapefile
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

class Visualizer:
    """处理建筑可视化"""
    
    def visualize_shp(self, shp_path: str) -> Optional[Image.Image]:
        """可视化单个shp文件"""
        try:
            img = fullpart_show_shp_building(shp_path)
            return Image.fromarray(img)
        except Exception as e:
            print(f"Error visualizing {shp_path}: {str(e)}")
            return None

    def create_visualization(self, 
                           buildings: List[Dict],
                           x1: float,
                           y1: float,
                           scale: Tuple[float, float],
                           shape_x: int,
                           shape_y: int,
                           highlight_first: bool = True) -> np.ndarray:
        """创建建筑可视化图像"""
        img = np.full((shape_y, shape_x, 3), [250, 250, 250], dtype=np.uint8)
        
        for i, building in enumerate(buildings):
            if building is None:
                continue
                
            points = np.array(building['points'])
            polygon_points = np.array([
                self._normalize_pt(pt, x1, y1, scale) 
                for pt in points
            ])
            
            color = (0, 0, 255) if highlight_first and i == 0 else (173, 216, 230)
            cv2.fillPoly(img, [polygon_points], color)
            cv2.polylines(img, [polygon_points], True, (0, 0, 0), 1)
            
        return img

    def _normalize_pt(self, 
                     pt: Tuple[float, float], 
                     x: float, 
                     y: float, 
                     scale: Tuple[float, float]) -> Tuple[int, int]:
        """标准化坐标点"""
        return (int((pt[0]-x)*scale[0]), int((pt[1]-y)*scale[1])) 

def overlay_shp_files(base_image_path: str, fill_shp_paths: list, line_shp_paths: list) -> np.ndarray:
    """
    在底图上叠加显示多个shp文件，支持面填充和线条两种显示方式
    
    Args:
        base_image_path: str, 底图路径
        fill_shp_paths: list, 需要以面填充方式显示的shp文件路径列表
        line_shp_paths: list, 需要以线条方式显示的shp文件路径列表
    
    Returns:
        np.ndarray: 叠加显示后的图像
    """
    # 读取底图
    base_img = cv2.imread(base_image_path)
    if base_img is None:
        raise ValueError(f"无法读取底图: {base_image_path}")
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    
    # 获取图像尺寸
    height, width = base_img.shape[:2]
    
    # 创建matplotlib图形
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(base_img)
    
    # 设置颜色列表
    fill_colors = list(mcolors.TABLEAU_COLORS.values())  # 面填充颜色
    line_colors = ['red', 'blue', 'green', 'yellow', 'purple']  # 线条颜色
    
    # 处理面填充的shp文件
    for idx, shp_path in enumerate(fill_shp_paths):
        try:
            sf = shapefile.Reader(shp_path)
            color = fill_colors[idx % len(fill_colors)]
            
            patches = []
            for shape in sf.shapes():
                points = shape.points
                if len(points) > 2:  # 确保多边形至少有3个点
                    polygon = Polygon(points, True)
                    patches.append(polygon)
            
            if patches:
                p = PatchCollection(patches, alpha=0.3, color=color)
                ax.add_collection(p)
        except Exception as e:
            print(f"处理填充shp文件时出错 {shp_path}: {str(e)}")
    
    # 处理线条的shp文件
    for idx, shp_path in enumerate(line_shp_paths):
        try:
            sf = shapefile.Reader(shp_path)
            color = line_colors[idx % len(line_colors)]
            
            for shape in sf.shapes():
                points = np.array(shape.points)
                if len(points) > 1:  # 确保至少有2个点可以画线
                    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2)
        except Exception as e:
            print(f"处理线条shp文件时出错 {shp_path}: {str(e)}")
    
    # 设置坐标轴
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # 注意y轴方向
    ax.axis('off')  # 隐藏坐标轴
    
    # 调整布局
    plt.tight_layout(pad=0)
    
    # 将图形转换为numpy数组
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # 清理matplotlib图形
    plt.close(fig)
    
    return img_array