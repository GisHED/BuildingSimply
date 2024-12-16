import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import shapefile
import io
import cv2
import numpy as np
import os

# 定义每个类别的颜色
CATEGORY_COLORS = {
    'I': (255, 165, 0),    # 橙色
    'L': (124, 252, 0),    # 绿色
    'F': (255, 20, 147),   # 深粉色
    'U': (0, 0, 255),       # 蓝色
    'Y': (255, 255, 0),    # 黄色
    'T': (160, 32, 240),   # 紫色
    'C': (255, 192, 203),  # 粉红色
    'Z': (0, 255, 255),     # 青色
    'B': (120, 200, 0),         # 黑色
    'E': (200, 160, 187),   # 不知道
    'N': (255, 0, 0),       # 红色
    'S': (255, 165, 0),    # 橙色
    'V': (124, 252, 0),    # 绿色
    'H': (200, 200, 147),   # 深粉色
    'K': (0, 0, 255),       # 蓝色
    'W': (255, 255, 0),    # 黄色
}

buffered_shp_data = {}
four_part_borders = {}
four_part_regulations = {}
four_part_shapeCategory = {}
def get_big_file_path_for_part8():
    root_dir = './database/shp_data/osm'

    shp_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.shp'):
                shp_files.append(os.path.join(root, file))
    return shp_files
shp_list_for_part8 = get_big_file_path_for_part8()

def fullpart_show_random_building_drop(shp_path,num,_):
    global buffered_shp_data
    shp_path = './database/shp_data/osm/org_osm.shp'
    if shp_path not in buffered_shp_data:
        print('reading file:',shp_path)
        buffered_shp_data[shp_path] = gpd.read_file(shp_path)
    buildings = buffered_shp_data[shp_path]
    # 随机选取一个建筑物
    buildings['type'] = np.random.randint(0, 11, size=len(buildings))
    random_building = buildings.sample(n=1)
    # 计算选中建筑物与所有建筑物的距离
    distances = buildings.geometry.distance(random_building.geometry.iloc[0])    
    # 获取距离最近的200个建筑物的索引
    nearest_buildings_idx = distances.argsort()[:num]
    # 获取距离最近的200个建筑物的数据
    nearest_buildings = buildings.iloc[nearest_buildings_idx]  
    fig, ax = plt.subplots(figsize=(40, 40))
    # buildings.plot(ax=ax, color='lightgrey')
    random_building.plot(ax=ax, color='red', markersize=50)
    nearest_buildings.plot(ax=ax, color='blue', markersize=20)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())#去掉x轴刻度
    plt.gca().yaxis.set_major_locator(plt.NullLocator())#去年y轴刻度
    # 将图形转换为 PIL 图像对象
    buf = io.BytesIO()  # 创建一个 BytesIO 对象
    plt.savefig(buf, format='png')  # 保存 Matplotlib 图形至内存中
    buf.seek(0)  # 将指针移动到对象起始位置
    pil_image = Image.open(buf)  # 使用 PIL 打开内存中的图像
    return None,None,pil_image,None,None

def get_base_pt(scale_x,scale_y):
    x1,y1 = scale_x
    x2,y2 = scale_y
    x,y = np.abs(x1-x2),np.abs(y1-y2)
    if x*y == 0:
        print('#'*60,'\n[get_base_pt][error]:',scale_x,scale_y)
        return x1,y1,1,1200,1200
    shape_x = 2400
    shape_y = int(y/x*shape_x)
    # print(shape_x,shape_y)
    scale = (shape_x/x,shape_y/y)
    # print(scale)
    return x1,y1,scale,shape_x,shape_y
def normlize_pt(pt,x,y,scale):
    p_x,p_y = pt
    # print(pt)
    return (((p_x-x)*scale[0]).astype(np.int),((p_y-y)*scale[1]).astype(np.int))

def visualize_buildings(buildings, x1, y1, scale, shape_x, shape_y, categories=None):
    # 背景为奶白色
    img = np.full((shape_y, shape_x, 3), [250, 250, 250], dtype=np.uint8)
    for idx, each_shape in enumerate(buildings):
        # print(f'visulize_buildings: {type(each_shape)},{type(each_shape) == shapefile.Shape}')
        all_points = np.array(each_shape.points) if type(each_shape) == shapefile.Shape else np.array(each_shape)
        if len(all_points) < 3:
            continue  # Skip if there are less than 3 points to form a polygon
        polygon_points = np.array([normlize_pt(pt, x1, y1, scale) for pt in all_points])
        color = (173, 216, 230)  # 默认颜色淡蓝色填充
        if categories is not None and idx < len(categories):
            category = categories[idx]
            color = CATEGORY_COLORS.get(category, (173, 216, 230))
        cv2.fillPoly(img, [polygon_points], color)  # 淡蓝色填充
        cv2.polylines(img, [polygon_points], isClosed=True, color=(0, 0, 0), thickness=1)  # 黑色轮廓

    if categories is not None:
        # 绘制类别图例
        legend_x, legend_y = 20, 20
        for category in set(categories):
            color = CATEGORY_COLORS.get(category, (173, 216, 230))
            cv2.rectangle(img, (legend_x, legend_y), (legend_x + 100, legend_y + 60), color, -1)
            cv2.putText(img, category, (legend_x + 120, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            legend_y += 55        
        # # 绘制分类标签
        # for idx, each_shape in enumerate(buildings):
        #     if idx >= len(categories):
        #         break
        #     category = categories[idx]
        #     x, y = normlize_pt(each_shape.points[0], x1, y1, scale)
        #     cv2.putText(img, category, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 黑色字体
    return img

def fullpart_show_shp_building(shp_path):
    if shp_path not in buffered_shp_data:
        print('reading file:', shp_path)
        border_shape = shapefile.Reader(shp_path)
        buffered_shp_data[shp_path] = border_shape.shapes()
    border = buffered_shp_data[shp_path]
    show_data = border
    
    all_points = [p for i in range(len(show_data)) for p in show_data[i].points]
    scale_x, scale_y = np.min(all_points, axis=0), np.max(all_points, axis=0)
    x1, y1, scale, shape_x, shape_y = get_base_pt(scale_x, scale_y)
    img_full = visualize_buildings(show_data, x1, y1, scale, shape_x, shape_y)
    return img_full
def fullpart_show_random_building(shp_path,building_num,base_index,part_building_num,dis_type_str,part_building_center_idx):
    global four_part_borders
    if shp_path not in buffered_shp_data:
        print('reading file:', shp_path)
        border_shape = shapefile.Reader(shp_path)
        buffered_shp_data[shp_path] = border_shape.shapes()
    border = buffered_shp_data[shp_path]

    show_logs = {'part_buildings_index':[]}
    
    try:
        base_index = int(base_index)
        part_index = [int(i) for i in part_building_center_idx.split(',')]
    except:
        base_index = -1
        part_index = [-1,-1,-1,-1]
    if base_index>=len(border) or base_index<0:
        base_index = np.random.randint(0, len(border))
    dis_type_convert = {'1':1,'2':2,'3':3,'inf':np.inf}
    dis_type = dis_type_convert[dis_type_str]
    ordered_list = sorted(range(len(border)), key=lambda k: np.linalg.norm(np.array(border[k].points[0]) - np.array(border[base_index].points[0]), ord=dis_type))
    show_data = [border[i] for i in ordered_list[:building_num]]
    
    all_points = [p for i in range(len(show_data)) for p in show_data[i].points]
    scale_x, scale_y = np.min(all_points, axis=0), np.max(all_points, axis=0)
    x1, y1, scale, shape_x, shape_y = get_base_pt(scale_x, scale_y)

    def select_buildings_in_area(x_min, y_min, width, height):
        area_buildings = [shape for shape in show_data if x_min <= shape.points[0][0] <= x_min + width and y_min <= shape.points[0][1] <= y_min + height]
        return area_buildings

    def visualize_area(area_buildings,building_num=100,center_index=-1):
        if not area_buildings:
            return None
        center_index = np.random.randint(0, len(area_buildings)) if (center_index == -1 or center_index >= len(area_buildings)) else center_index
        show_logs['part_buildings_index'].append(center_index)
        center_building = area_buildings[center_index]
        ordered_buildings = sorted(area_buildings, key=lambda b: np.linalg.norm(np.array(b.points[0]) - np.array(center_building.points[0]), ord=np.inf))
        selected_buildings = ordered_buildings[:building_num]
        all_points_in_area = [p for i in range(len(selected_buildings)) for p in selected_buildings[i].points]
        # print(f'all_points_in_area:{all_points_in_area}')
        area_scale_x, area_scale_y = np.min(all_points_in_area, axis=0), np.max(all_points_in_area, axis=0)
        x1_area, y1_area, scale_area, shape_x_area, shape_y_area = get_base_pt(area_scale_x, area_scale_y)
        return visualize_buildings(selected_buildings, x1_area, y1_area, scale_area, shape_x_area, shape_y_area), (area_scale_x, area_scale_y),selected_buildings

    area_width, area_height = (scale_y - scale_x) / 2

    top_left_buildings = select_buildings_in_area(x1, y1, area_width, area_height)
    top_right_buildings = select_buildings_in_area(x1 + area_width, y1, area_width, area_height)
    bottom_left_buildings = select_buildings_in_area(x1, y1 + area_height, area_width, area_height)
    bottom_right_buildings = select_buildings_in_area(x1 + area_width, y1 + area_height, area_width, area_height)

    img_top_left, box_top_left, top_left_border_data = visualize_area(top_left_buildings, part_building_num,part_index[0])
    img_top_right, box_top_right, top_right_border_data = visualize_area(top_right_buildings, part_building_num,part_index[1])
    img_bottom_left, box_bottom_left, bottom_left_border_data = visualize_area(bottom_left_buildings, part_building_num,part_index[2])
    img_bottom_right, box_bottom_right, bottom_right_border_data = visualize_area(bottom_right_buildings, part_building_num,part_index[3])

    # 保存四个区域的建筑坐标信息
    for key,value in zip(['top_left','top_right','bottom_left','bottom_right'],[top_left_border_data,top_right_border_data,bottom_left_border_data,bottom_right_border_data]):
        four_part_borders[key] = value

    # 在完整图中显示子区域的位置
    img_full = visualize_buildings(show_data, x1, y1, scale, shape_x, shape_y)
    box_color = (255, 0, 0)
    thickness = 3
    for box in [box_top_left, box_top_right, box_bottom_left, box_bottom_right]:
        if box is not None:
            (x_min, y_min), (x_max, y_max) = box
            top_left = normlize_pt((x_min, y_min), x1, y1, scale)
            bottom_right = normlize_pt((x_max, y_max), x1, y1, scale)
            cv2.rectangle(img_full, top_left, bottom_right, box_color, thickness)
            
    show_logs['base_index'] = base_index
    show_logs['dis_type'] = dis_type
    show_logs['total_buildings'] = building_num
    show_logs['part_building_num'] = part_building_num

    return img_top_left, img_bottom_left, img_full, img_top_right, img_bottom_right,show_logs 

def visualize_overlapped_buildings(method_buildings_dict, selected_methods, 
                                 x1, y1, scale, shape_x, shape_y):
    """可视化叠加的建筑轮廓
    method_buildings_dict: 不同方法的建筑数据字典
    {
        'original': 原始建筑列表,
        'rectangle': 矩形变换结果列表,
        'template': 模板匹配结果列表,
        'recursive': 递归匹配结果列表,
        'afpm': AFPM结果列表,
        'merged': 融合结果列表
    }    
    """
    # 方法对应的轮廓颜色
    METHOD_COLORS = {
        'rectangle': (255, 0, 0),     # 红色
        'template': (0, 255, 0),      # 绿色
        'recursive': (0, 0, 255),     # 蓝色
        'afpm': (255, 165, 0),        # 橙色
        'merged': (160, 32, 240),     # 紫色
        'regular_bart': (255, 20, 147), # 深粉色
        'regular_urban': (0, 255, 255),  # 青色
        'regular_bart_10Types': (219, 112, 147),  # 浅粉红色
        'runtime_ev': (128, 0, 128),    # 深紫色
        'arcmap': (139, 69, 19),      # 棕色
        'mlr': (0, 128, 128),         # 青绿色
        'svm': (255, 140, 0),         # 深橙色
        'dpnn': (148, 0, 211)         # 深紫罗兰色
    }
    
    # 增加图例区域的高度
    legend_height = int(shape_y*0.08)  # 增加图例区域高度
    legend_scale = legend_height/150
    img = np.full((shape_y + legend_height, shape_x, 3), [250, 250, 250], dtype=np.uint8)

    
    # 绘制原始建筑（浅灰色填充，无轮廓）
    for building in method_buildings_dict['original']:
        if building is None:
            continue
        points = np.array(building['points'])
        polygon_points = np.array([
            (int((pt[0]-x1)*scale[0]), int((pt[1]-y1)*scale[1])) 
            for pt in points
        ])
        cv2.fillPoly(img, [polygon_points], (200, 200, 100))
    
    # 绘制选中方法的建筑轮廓
    for index,method in enumerate(selected_methods):
        if method not in method_buildings_dict:
            continue
        
        buildings = method_buildings_dict[method]
        color = METHOD_COLORS.get(method, (0, 0, 0))
        
        for building in buildings:
            if building is None:
                continue
            points = np.array(building['points'])
            polygon_points = np.array([
                (int((pt[0]-x1)*scale[0]), int((pt[1]-y1)*scale[1])) 
                for pt in points
            ])
            line_width = 2 if index == 0 else 1
            cv2.polylines(img, [polygon_points], True, color, line_width)
    
    # 绘制图例背景
    legend_margin = 20  # 图例边距
    legend_padding = 25  # 图例内边距
    legend_start_y = shape_y + 50
    legend_height = 100
    
    # 计算图例总宽度
    legend_width_per_item = 180  # 每个图例项的宽度
    total_items = len(selected_methods) + 1  # +1 for original
    total_legend_width = total_items * legend_width_per_item
    
    # 居中图例
    legend_start_x = (shape_x - total_legend_width) // 2
    
    # 绘制图例背景框
    cv2.rectangle(img, 
                 (legend_start_x - legend_padding, 
                  legend_start_y - legend_padding),
                 (legend_start_x + total_legend_width + legend_padding, 
                  legend_start_y + legend_height + legend_padding),
                 (100, 100, 100), 1)  # 灰色边框
    
    # 绘制图例标题
    title_y = legend_start_y - 5
    cv2.putText(img, "Legend", 
                (legend_start_x, title_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 绘制图例项
    current_x = legend_start_x + legend_padding
    item_y = legend_start_y + 30
    
    # 原始建筑图例
    cv2.rectangle(img, 
                 (current_x, item_y - 15),
                 (current_x + 30, item_y + 15),
                 (200, 200, 200), -1)  # 填充色
    cv2.rectangle(img, 
                 (current_x, item_y - 15),
                 (current_x + 30, item_y + 15),
                 (100, 100, 100), 1)  # 边框
    cv2.putText(img, "Original", 
                (current_x + 40, item_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    current_x += legend_width_per_item
    
    # 其他方法的图例
    for method in selected_methods:
        if method in METHOD_COLORS:
            color = METHOD_COLORS[method]
            # 绘制示例图形（空心矩形）
            cv2.rectangle(img, 
                         (current_x, item_y - 15),
                         (current_x + 30, item_y + 15),
                         color, 2)
            # 添加方法名称
            method_name = method.replace('_', ' ').title()
            cv2.putText(img, method_name,
                       (current_x + 40, item_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            current_x += legend_width_per_item
    if not os.path.exists('save_hd_img'):
        return img
    
    # 创建高分辨率版本
    hd_width = 100000  # 10万像素宽度
    hd_height = int(hd_width * (shape_y + legend_height) / shape_x)  # 保持原始宽高比
    hd_img = np.full((hd_height, hd_width, 3), [250, 250, 250], dtype=np.uint8)
    
    # 计算缩放比例
    hd_scale = (hd_width/shape_x, hd_height/(shape_y + legend_height))
    
    # 绘制原始建筑（高清版本）
    for building in method_buildings_dict['original']:
        if building is None:
            continue
        points = np.array(building['points'])
        polygon_points = np.array([
            (int((pt[0]-x1)*scale[0]*hd_scale[0]), int((pt[1]-y1)*scale[1]*hd_scale[1])) 
            for pt in points
        ])
        cv2.fillPoly(hd_img, [polygon_points], (200, 200, 200))
    
    # 绘制选中方法的建筑轮廓（高清版本）
    for index, method in enumerate(selected_methods):
        if method not in method_buildings_dict:
            continue
        
        buildings = method_buildings_dict[method]
        color = METHOD_COLORS.get(method, (0, 0, 0))
        
        for building in buildings:
            if building is None:
                continue
            points = np.array(building['points'])
            polygon_points = np.array([
                (int((pt[0]-x1)*scale[0]*hd_scale[0]), int((pt[1]-y1)*scale[1]*hd_scale[1])) 
                for pt in points
            ])
            line_width = 3 if index == 0 else 3  # 高清版本增加线宽
            cv2.polylines(hd_img, [polygon_points], True, color, line_width)



    # 绘制高清版本的图例
    legend_start_y = int(shape_y * hd_scale[1]) + 60
    legend_height = int(100 * hd_scale[1])
    legend_width_per_item = int(180 * hd_scale[0])
    total_items = len(selected_methods) + 1
    total_legend_width = total_items * legend_width_per_item
    legend_start_x = (hd_width - total_legend_width) // 2
    legend_padding = int(25 * hd_scale[0])
    
    # 绘制图例背景框
    cv2.rectangle(hd_img, 
                 (legend_start_x - legend_padding, 
                  legend_start_y - legend_padding),
                 (legend_start_x + total_legend_width + legend_padding, 
                  legend_start_y + legend_height + legend_padding),
                 (100, 100, 100), 3)
    
    # 绘制图例标题
    title_y = legend_start_y - 15
    cv2.putText(hd_img, "Legend", 
                (legend_start_x, title_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.1, (0, 0, 0), 6)
    
    # 绘制图例项
    current_x = legend_start_x + legend_padding
    item_y = legend_start_y + 90
    
    # 原始建筑图例
    box_size = int(30 * hd_scale[0])
    cv2.rectangle(hd_img, 
                 (current_x, item_y - box_size//2),
                 (current_x + box_size, item_y + box_size//2),
                 (200, 200, 200), -1)
    cv2.rectangle(hd_img, 
                 (current_x, item_y - box_size//2),
                 (current_x + box_size, item_y + box_size//2),
                 (100, 100, 100), 3)
    cv2.putText(hd_img, "Original", 
                (current_x + box_size + 10, item_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    current_x += legend_width_per_item
    
    # 其他方法的图例
    for method in selected_methods:
        if method in METHOD_COLORS:
            color = METHOD_COLORS[method]
            cv2.rectangle(hd_img, 
                         (current_x, item_y - box_size//2),
                         (current_x + box_size, item_y + box_size//2),
                         color, 6)
            method_name = method.replace('_', ' ').title()
            cv2.putText(hd_img, method_name,
                       (current_x + box_size + 10, item_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
            current_x += legend_width_per_item
    

    method_name = '_'.join(selected_methods)    
    # 保存高清TIFF图像
    hd_tiff_path = f'./evaluation_results/{method_name}_overlapped_buildings_HD.png'
    print(f'保存高清图像到：{hd_tiff_path}')
    cv2.imwrite(hd_tiff_path, hd_img)
    
    return img
