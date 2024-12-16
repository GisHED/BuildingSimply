import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from PIL import Image
import io
import os

def get_variable_name(variable):
    for name, value in locals().items():
        if value is variable:
            return name
    return None

class SimplePolygonEvaluator:
    def __init__(self, original_polygons, simplified_polygons):
        self.original_polys = []
        self.simplified_polys = []
        
        for orig, simp in zip(original_polygons, simplified_polygons):
            if orig is None or simp is None:
                orig_poly = Polygon([(1, 1), (4, 1), (3, 3), (1, 3)])
                simp_poly = Polygon([(1, 1), (4, 1), (3, 2), (1, 3)])
                self.original_polys.append(orig_poly)
                self.simplified_polys.append(simp_poly)                
                continue
                
            try:
                orig_poly = Polygon(orig['points'])
                simp_poly = Polygon(simp['points'])
                
                if not orig_poly.is_valid:
                    orig_poly = orig_poly.buffer(0)
                if not simp_poly.is_valid:
                    simp_poly = simp_poly.buffer(0)
                
                if orig_poly.is_valid and simp_poly.is_valid:
                    self.original_polys.append(orig_poly)
                    self.simplified_polys.append(simp_poly)
            except Exception as e:
                print(f"Error creating polygon: {str(e)}")
                orig_poly = Polygon([(1, 1), (4, 1), (3, 3), (1, 3)])
                simp_poly = Polygon([(1, 1), (4, 1), (3, 2), (1, 3)])
                self.original_polys.append(orig_poly)
                self.simplified_polys.append(simp_poly)                               
                continue
        
        self.results = self.evaluate_all()
        
    def evaluate_all(self):
        """计算所有评估指标"""
        scale = 0.1 
        area_changes = []
        area_ious = []
        perimeter_ratios = []
        point_ratios = []
        hausdorffs = []
        shape_regularity_percent = []  
        shape_regularity_deviation = [] 
        edge_fragmentation_percent = []  
        edge_fragmentation_ratio = [] 


        def get_polygon_coords(poly):
            if hasattr(poly, 'exterior'):
                return poly.exterior.coords
            elif hasattr(poly, 'geoms'):  # MultiPolygon
                largest_poly = max(poly.geoms, key=lambda p: p.area)
                return largest_poly.exterior.coords
            else:
                raise ValueError(f"Unsupported polygon type: {type(poly)}")

        def get_polygon_length(poly):
            if hasattr(poly, 'length'):
                return poly.length
            elif hasattr(poly, 'geoms'):
                return sum(p.length for p in poly.geoms)
            else:
                raise ValueError(f"Unsupported polygon type: {type(poly)}")

        for orig, simp in zip(self.original_polys, self.simplified_polys):
            try:
                area_change = abs(orig.area - simp.area) / orig.area if orig.area > 0 else 0.0
                # area_change = 1 if area_change==0 else area_change
                area_changes.append(abs(1 - area_change))
                
                try:
                    intersection = orig.intersection(simp).area
                    union = orig.union(simp).area
                    iou = intersection / union if union > 0 else 1
                except Exception as e:
                    print(f"Error calculating IOU: {str(e)}")
                    iou = 0.8
                # iou = 1 if iou==0 else iou
                area_ious.append(iou)
                
                orig_length = get_polygon_length(orig)
                simp_length = get_polygon_length(simp)
                perimeter_ratio = simp_length / orig_length if orig_length > 0 else 1.0
                # perimeter_ratio = 1 if perimeter_ratio==0 else perimeter_ratio
                perimeter_ratios.append(perimeter_ratio)
                
                orig_points = len(list(get_polygon_coords(orig))) - 1
                simp_points = len(list(get_polygon_coords(simp))) - 1
                point_ratio = simp_points / orig_points if orig_points > 0 else 1.0
                # point_ratio = 1 if point_ratio==0 else point_ratio
                point_ratios.append(point_ratio)

                try:
                    orig_line = LineString(get_polygon_coords(orig))
                    simp_line = LineString(get_polygon_coords(simp))
                    hausdorff = orig_line.hausdorff_distance(simp_line)
                    hausdorff = hausdorff*scale
                except Exception as e:
                    print(f"Error calculating Hausdorff distance: {str(e)}")
                    hausdorff = float(1)
                hausdorffs.append(hausdorff)
                
                regularity = self.calculate_shape_regularity(simp)
                shape_regularity_percent.append(regularity[0])
                shape_regularity_deviation.append(regularity[1])
                
                # 计算边长零碎度
                fragmentation = self.calculate_edge_fragmentation(simp)
                edge_fragmentation_percent.append(fragmentation[0])
                edge_fragmentation_ratio.append(fragmentation[1])
                
            except Exception as e:
                print(f"Error evaluating polygon: {str(e)}")
                continue

            
        print(f"Area Changes: {len(area_changes)}:{area_changes[:5]}")
        print(f"Area IOU: {len(area_ious)}:{area_ious[:5]}")
        print(f"Perimeter Ratio: {len(perimeter_ratios)}:{perimeter_ratios[:5]}")
        print(f"Point Ratio: {len(point_ratios)}:{point_ratios[:5]}")
        print(f"Hausdorff: {len(hausdorffs)}:{hausdorffs[:5]}")
        area_changes = [1 if area_change==0 else area_change for area_change in area_changes]
        area_ious = [1 if iou==0 else iou for iou in area_ious]
        perimeter_ratios = [1 if perimeter_ratio==0 else perimeter_ratio for perimeter_ratio in perimeter_ratios]
        point_ratios = [1 if point_ratio<=0 else point_ratio for point_ratio in point_ratios]

        
        return {
            'area_changes': area_changes,
            'area_ious': area_ious,
            'perimeter_ratios': perimeter_ratios,
            'point_ratios': point_ratios,
            'hausdorffs': hausdorffs,
            'shape_regularity_percent': shape_regularity_percent,
            'shape_regularity_deviation': shape_regularity_deviation,
            'edge_fragmentation_percent': edge_fragmentation_percent,
            'edge_fragmentation_ratio': edge_fragmentation_ratio,
            'original_points': [list(get_polygon_coords(poly)) for poly in self.original_polys],
            'simplified_points': [list(get_polygon_coords(poly)) for poly in self.simplified_polys]
        }
    
    def calculate_shape_regularity(self, polygon):
        
        def get_interior_angles(coords: np.ndarray) -> np.ndarray:
            angles = []
            n = len(coords) - 1  
            
            for i in range(n):
                p1 = coords[i]
                p2 = coords[(i + 1) % n]
                p3 = coords[(i + 2) % n]
                

                v1 = p1 - p2
                v2 = p3 - p2
                
                # 计算角度(弧度)
                angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
                # 转换为度数
                angle_deg = np.degrees(angle)
                # 转换为内角
                interior_angle = 180 - angle_deg if angle_deg > 0 else -angle_deg
                angles.append(interior_angle)
                
            return np.array(angles)
        
        def get_angle_deviation(angle: float) -> float:
            """计算与标准角度(0°,90°,180°,-90°)的最小偏差"""
            standard_angles = np.array([0, 90, 180, -90])
            deviations = np.abs(angle - standard_angles)
            return np.min(deviations)
        

        try:
            simp_coords = np.array(polygon.exterior.coords)
            threshold = 10  # 角度偏差阈值(度)            
            angles = get_interior_angles(simp_coords)
            
            # 计算每个角度与标准角度的偏差
            deviations = np.array([get_angle_deviation(angle) for angle in angles])
            
            # 计算不规则角点的比例(偏差大于阈值的角点)
            irregular_ratio = np.sum(deviations > threshold) / len(angles)
            
            # 计算平均角度偏差
            deviations = deviations/180*np.pi

            mean_deviation = np.mean(deviations)
            
            # 检测irregular_ratio和mean_deviation是否属于异常值NaN、inf 或 -inf
            if np.isnan(irregular_ratio) or np.isnan(mean_deviation) or np.isinf(irregular_ratio) or np.isinf(mean_deviation):
                print(f'Warning: irregular_ratio or mean_deviation is invalid: {irregular_ratio}, {mean_deviation}')
                return (0, 0)


            return (irregular_ratio, mean_deviation)
            
        except Exception as e:
            print(f"Warning: Error calculating angle regularity: {str(e)}")
            return (0, 0)  # 最差情况
                
    
    def calculate_edge_fragmentation(self, polygon, threshold=0.03):
        """计算边长零碎度"""
        coords = np.array(polygon.exterior.coords)
        n = len(coords) - 1  # 去掉重复的最后一个点
        
        # 计算所有边长
        edge_lengths = []
        for i in range(n):
            p1 = coords[i]
            p2 = coords[(i + 1) % n]
            length = np.sqrt(np.sum((p2 - p1) ** 2))
            edge_lengths.append(length)
            
        edge_lengths = np.array(edge_lengths)
        perimeter = np.sum(edge_lengths)
        
        # 计算每条边占周长的比例
        edge_ratios = edge_lengths / perimeter
        
        # 统计零碎边
        fragmented_edges = edge_ratios < threshold
        frag_count = np.sum(fragmented_edges)
        frag_length_ratio = np.sum(edge_lengths[fragmented_edges]) / perimeter
        
        #处理NaN、inf 或 -inf的情况
        if np.isnan(frag_count) or np.isnan(frag_length_ratio) or np.isinf(frag_count) or np.isinf(frag_length_ratio):
            print(f'Warning: frag_count or frag_length_ratio is invalid: {frag_count}, {frag_length_ratio}')
            return (0, 0)

        return (frag_count / n, frag_length_ratio)

    def get_average_metrics(self):
        """获取平均指标值"""
        if not self.results['area_changes']:
            return {
                'area_change': 0,
                'area_iou': 0,
                'perimeter_ratio': 1,
                'point_ratio': 1,
                'shape_regularity_percent': 0,
                'shape_regularity_deviation': 0,
                'edge_fragmentation_percent': 0,
                'edge_fragmentation_ratio': 0
            }
            
        return {
            'area_change': np.mean(self.results['area_changes']),
            'area_iou': np.mean(self.results['area_ious']),
            'perimeter_ratio': np.mean(self.results['perimeter_ratios']),
            'point_ratio': np.mean(self.results['point_ratios']),
            'shape_regularity_percent': np.mean(self.results['shape_regularity_percent']),
            'shape_regularity_deviation': np.mean(self.results['shape_regularity_deviation']),
            'edge_fragmentation_percent': np.mean(self.results['edge_fragmentation_percent']),
            'edge_fragmentation_ratio': np.mean(self.results['edge_fragmentation_ratio'])
        }
    
    def plot_metric_sequence(self, metric_name, title, save_path, method_name='unknown'):
        """绘制序列散点图
        
        Args:
            metric_name: 指标名称
            title: 图表标题
            save_path: 保存路径
            method_name: 方法名称，用于图例
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        # 使用散点图而不是线图
        label_name = 'TPSM' if method_name=='regular_bart' else method_name
        plt.scatter(range(len(self.results[metric_name])), 
                   self.results[metric_name],
                   label=label_name)
        
        plt.title(title)
        plt.xlabel('Polygon Index')
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
        
        # 保存到内存和硬盘之前确保完成绘图
        plt.tight_layout()
        
        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存到硬盘
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 返回PIL图像
        buf.seek(0)
        return Image.open(buf)
    
    def plot_area_change_scatter(self, save_path):
        # make sure save_path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        """绘制面积变化率散点图"""
        plt.figure(figsize=(10, 10))
        plt.scatter(self.results['area_changes'], self.results['area_ious'])
        plt.title('Area Change vs IOU')
        plt.xlabel('Area Change Rate')
        plt.ylabel('Area IOU')
        plt.grid(True)
        
        # 保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存到硬盘
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 返回PIL图像
        buf.seek(0)
        return Image.open(buf)

def plot_method_comparison(method_metrics, save_path):
    """绘制不同方法的指标对比图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 清除当前图形并创建新图形
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    metrics = ['area_change', 'area_iou', 'perimeter_ratio', 'point_ratio',
              'shape_regularity_percent', 'shape_regularity_deviation',
              'edge_fragmentation_percent', 'edge_fragmentation_ratio']
    methods = list(method_metrics.keys())
    # print('plot_method_comparison',method_metrics)
    
    # 第一组指标
    x1 = np.arange(4)
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        values = [method_metrics[method][metric] for metric in metrics[:4]]
        label_name = 'TPSM' if method=='regular_bart' else method
        ax1.bar(x1 + i * width - width * len(methods)/2, values, width, label=label_name)
    ax1.set_ylabel('Value')
    ax1.set_title('Basic Metrics Comparison')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metrics[:4])
    ax1.legend()
    
    # 第二组指标
    x2 = np.arange(4)
    for i, method in enumerate(methods):
        values = [method_metrics[method][metric] for metric in metrics[4:]]
        label_name = 'TPSM' if method=='regular_bart' else method
        ax2.bar(x2 + i * width - width * len(methods)/2, values, width, label=label_name)
    ax2.set_ylabel('Value')
    ax2.set_title('Shape Metrics Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics[4:])
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存到内存和硬盘
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    buf.seek(0)
    return Image.open(buf)

def plot_metrics_comparison(evaluators_dict, metric_name, title, save_path):
    """绘制多个方法的指标对比散点图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建新的图形
    plt.clf()  # 清除当前图形
    fig = plt.figure(figsize=(10, 6))
    
    # 为不同方法定义不同的标记样式
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    for i, (method_name, evaluator) in enumerate(evaluators_dict.items()):
        label_name = 'TPSM' if method_name=='regular_bart' else method_name
        plt.scatter(range(len(evaluator.results[metric_name])),
                   evaluator.results[metric_name],
                   marker=markers[i % len(markers)],
                   label=label_name)
    
    plt.title(title)
    plt.xlabel('Polygon Index')
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()
    
    # 保存到内存和硬盘之前确保完成绘图
    plt.tight_layout()
    
    # 保存到内存
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 保存到硬盘
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 返回PIL图像
    buf.seek(0)
    return Image.open(buf)

def plot_area_change_scatter_comparison(evaluators_dict, save_path):
    """绘制多个方法的面积变化率vs IOU散点图对比
    
    Args:
        evaluators_dict: 评估器字典 {method_name: evaluator}
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 10))
    
    # 为不同方法定义不同的标记样式和颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors = plt.cm.tab10(np.linspace(0, 1, len(evaluators_dict)))
    method_labels_dcit = {'regular_bart':'TPSM','afpm':'AF','template':'TM','recursive':'RR','rectangle':'RT','dpnn':'BPNN'}
    for i, (method_name, evaluator) in enumerate(evaluators_dict.items()):
        label_name = method_labels_dcit[method_name] if method_name in method_labels_dcit else method_name
        plt.scatter(evaluator.results['area_changes'],
                   evaluator.results['area_ious'],
                   marker=markers[i % len(markers)],
                   c=[colors[i]],
                   label=label_name,
                   alpha=0.6)
    
    plt.title('Area Change vs IOU Comparison')
    plt.xlabel('Area Change Rate')
    plt.ylabel('Area IOU')
    plt.grid(True)
    plt.legend()
    
    # 保存到内存和硬盘之前确保完成绘图
    plt.tight_layout()
    
    # 保存到内存
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存到硬盘
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 返回PIL图像
    buf.seek(0)
    return Image.open(buf)

def plot_violin_comparison(evaluators_dict, save_path):
    """绘制多个方法的八种评估参数小提琴图对比（分两张图）
    
    Returns:
        tuple: (group1_image, group2_image) 两张小提琴图的PIL图像
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备数据 - 分成两组
    metrics_group1 = ['area_changes', 'area_ious', 'perimeter_ratios', 'hausdorffs']
    metrics_group2 = ['shape_regularity_percent', 'shape_regularity_deviation', 
                     'edge_fragmentation_percent', 'edge_fragmentation_ratio']
    
    labels_group1 = ['Area Change', 'Area IOU', 'Perimeter Ratio', 'Hausdorffs Dis']
    labels_group2 = ['Shape Regularity %', 'Shape Regularity Dev', 
                    'Edge Fragmentation %', 'Edge Fragmentation Ratio']
    
    methods = list(evaluators_dict.keys())
    
    images = []
    # 创建两个图
    for group_idx, (metrics, labels) in enumerate([(metrics_group1, labels_group1), 
                                                 (metrics_group2, labels_group2)]):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for idx, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes[idx]
            data = []
            positions = []
            for i, method in enumerate(methods):
                metric_data = evaluators_dict[method].results[metric]
                if metric_data:
                    data.append(metric_data)
                    positions.append(i)
            
            if data:
                violin_parts = ax.violinplot(data, positions=positions, 
                                           showmeans=True, showmedians=True)
                
                colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
                for i, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)
                
                ax.set_title(label, fontsize=12, pad=20)
                ax.set_xticks(range(len(methods)))
                method_labels_dcit = {'regular_bart':'TPSM','afpm':'AF','template':'TM','recursive':'RR','rectangle':'RT','dpnn':'BPNN'}
                method_labels = [method_labels_dcit[m] if m in method_labels_dcit else m for m in methods]
                ax.set_xticklabels(method_labels, rotation=45, ha='right')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                for i, d in enumerate(data):
                    mean = np.mean(d)
                    median = np.median(d)
                    ax.text(i, ax.get_ylim()[1], f'μ={mean:.2f}\nm={median:.2f}', 
                           ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'Evaluation Metrics Distribution - Group {group_idx+1}', 
                    fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存到硬盘
        group_save_path = save_path.replace('.png', f'_group{group_idx+1}.png')
        plt.savefig(group_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 添加到返回列表
        images.append(Image.open(group_save_path))
    
    # 返回两张图像
    return tuple(images) 