
import matplotlib.pyplot as plt
import numpy as np
from line_intersection import line, intersection, par_line_dist, point_in_line, cal_dist, azimuthAngle,Nrotation_angle_get_coor_coordinates, Srotation_angle_get_coor_coordinates
import json
import cv2,random
from PIL import Image,ImageFont,ImageDraw
from torchvision import transforms as T

#################################################################################################
#
#                                         显示部分
#
#################################################################################################
# opencv 显示图像
def show_img(img):
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_json(json_path,num=1000):
    vec_list = []
    filename = json_path
    with open(filename,encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)['indexs']
            vec_list.append(l)
            if len(vec_list)>num: break
    show_vec_list(vec_list)

# 将vector list显示为图像
def show_vec_list(vec_list,fig_name='fungis'):
    vec_size = len(vec_list)
    img_size = np.ceil(np.sqrt(vec_size)).astype(np.int32)
    plt.rcParams['figure.figsize'] = (16,16)
    for i in range(img_size):
        for j in range(img_size):  
            index = i*img_size+j
            if index >= vec_size:
                return
            vec = vec_list[index]
            pred_points = [[t%96,t//96] for t in vec]
            x, y = zip(*pred_points)
            plt.subplot(img_size, img_size, index+1)#当前画在第一行第2列图上
            # plt.axis('off')
            # print('{}[{}] : {}'.format(index,len(vec),vec))
            plt.plot(x, y, color='#6666ff', label=fig_name)  # x横坐标 y纵坐标 ‘k-’线性为黑色
            plt.grid()  

# 将点集的list显示为图像
def show_pt_list(pt_list,fig_name='fungis'):
    vec_size = len(pt_list)
    img_size = np.ceil(np.sqrt(vec_size)).astype(np.int32)
    plt.rcParams['figure.figsize'] = (16,16)
    for i in range(img_size):
        for j in range(img_size):  
            index = i*img_size+j
            if index >= vec_size:
                return
            vec = pt_list[index]
            x, y = zip(*vec)
            plt.subplot(img_size, img_size, index+1)#当前画在第一行第2列图上
            plt.plot(x, y, color='#6666ff', label=fig_name)  # x横坐标 y纵坐标 ‘k-’线性为黑色
            plt.grid()   


#################################################################################################
#
#                                         读取文件部分
#
#################################################################################################
# 显示shp文件的基本信息
def show_shp_info(file):
    print(str(file.shapeType))  # 输出shp类型
    print(file.encoding)# 输出shp文件编码
    print(file.bbox)  # 输出shp的文件范围（外包矩形）
    print(file.numRecords)  # 输出shp文件的要素数据
    print(file.fields)# 输出所有字段信息    

# 从json文件中返回所有的vec
# ways：0默认返回，只返回原始vec，1返回simpfy后的vec,2返回 veclist,simpfy_vec_list
def read_json(json_path,ways=0):
    vec_list = []
    simpfy_vec_list = []
    filename = json_path
    with open(filename,encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            vec_list.append(l['indexs'])
            if 'simpfied_indexs' in l.keys():
                simpfy_vec_list.append(l['simpfied_indexs'])
    if ways==0: return vec_list
    if ways==1: return simpfy_vec_list
    if ways==2: return vec_list,simpfy_vec_list
    return vec_list

def read_txt(txt_path):
    all_vec = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')[0]
            line = [int(i) for i in line.split(' ')]
            all_vec.append(line)
    return all_vec

# 将vec转换为np.array的point的格式
def vec2point(vec):
    if vec == None:
        return None
    p = []
    for v in vec:
        p.append([v%96,v//96])
    return np.array(p)

# 从json文件中返回所有的vec
def read_json_points(json_path,ways=0):
    vec_list = read_json(json_path,ways)
    if ways==2:
        points_list = []
        for vec in vec_list:
            points_list.append([vec2point(v) for v in vec])
    else:
        points_list = [vec2point(v) for v in vec_list]
    return points_list

# 从序列化好的json中生成vec
def some_vecs(json_path):
    filename = json_path
    with open(filename,encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)['indexs']
            yield l  



#################################################################################################
#
#                                     序列化vec 及 后处理部分
#
#################################################################################################
# 序列化中获取范围
def one_shape_scale(points):
    x,y = zip(*points)
    x_min,x_max = np.min(x),np.max(x)
    y_min,y_max = np.min(y),np.max(y)
    x_range = x_max-x_min
    y_range = y_max-y_min
    return np.array([x_range,y_range])

# 将点集进行序列化，返回vec
def serialize_with_scale(all_points, max_scale=None, map_scale=(95, 95)):
    x, y = zip(*all_points)
    if max_scale is None:
        max_scale = one_shape_scale(all_points)

    # Calculate the center point
    center_point = [ (np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2 ]
    serialized_border = {
        'points': [],
        'indexs': [],
        'params': [center_point, max_scale, map_scale]
    }

    for each_point in all_points:
        try:
            x, y = each_point
            # Scale and translate the coordinates
            x = int((x - center_point[0]) * map_scale[0] / max_scale[0] + map_scale[0] / 2)
            y = int((y - center_point[1]) * map_scale[1] / max_scale[1] + map_scale[1] / 2)

            if x < 0 or y < 0 or x > map_scale[0] or y > map_scale[1]:
                print("Point outside map scale:", (x,y))
                return None
            
            serialized_border['points'].append((x, y))
            serialized_border['indexs'].append(int(x + y * (map_scale[0] + 1)))

        except Exception as e:
            print(f"Error processing point: {each_point}")
            print(e)
            print('Data:', all_points)
            return None
    
    return serialized_border

# 将点集进行序列化，返回vec
def serialize_with_smallscale(all_points,max_scale=None,map_scale=(95,95)):
    x,y = zip(*all_points)
    if max_scale is None:
        max_scale = one_shape_scale(all_points)*0.9
    base_point = [np.min(x),np.min(y)]
    serialized_border = {}
    serialized_border['points'] = []
    serialized_border['indexs'] = []
    serialized_border['params'] = [base_point,max_scale,map_scale]
    for each_point in all_points:
        try:
            x,y = each_point
            x = int((x-base_point[0])*map_scale[0]/max_scale[0])
            y = int((y-base_point[1])*map_scale[1]/max_scale[1])
            serialized_border['points'].append((x,y))
            serialized_border['indexs'].append(int(x+y*(map_scale[0]+1)))
            if x<0 or y<0 or x>map_scale[0] or y>map_scale[1]:
                return None 
        except Exception as e:
            print(e)
            print('Data:',all_points)
            return None
    return serialized_border   

# 将点集进行序列化，返回vec
def serialize(all_points,map_scale=(95,95)):
    x,y = zip(*all_points)
    max_scale = one_shape_scale(all_points)
    base_point = [np.min(x),np.min(y)]
    serialized_border = {}
    serialized_border['points'] = []
    serialized_border['indexs'] = []
    serialized_border['params'] = [base_point,max_scale,map_scale]
    for each_point in all_points:
        try:
            x,y = each_point
            x = int((x-base_point[0])*map_scale[0]/max_scale[0])
            y = int((y-base_point[1])*map_scale[1]/max_scale[1])
            serialized_border['points'].append((x,y))
            serialized_border['indexs'].append(int(x+y*(map_scale[0]+1)))
            if x<0 or y<0 or x>map_scale[0] or y>map_scale[1]:
                return None 
        except Exception as e:
            print(e)
            print('Data:',all_points)
            return None
    return serialized_border   


# 过滤 序列化后的vec 剔除距离十分接近的点和共线（包括基本共线）的点
def filter_points(np_array):
    #删除过近的点
    distances = np.sqrt(np.sum(np.diff(np_array, axis=0)**2, axis=1))
    to_delete = np.where(distances < 0.1)[0] + 1
    np_array = np.delete(np_array, to_delete, axis=0)
    #删除平行的点
    angles = get_angels(np_array)
    to_delete = []
    for i, angle in enumerate(angles):
        # print(angle)
        if (angle > 178 and angle < 182) or (angle > -2 and angle < 2):
            # print('delete')
            to_delete.append(i+1)
    np_array = np.delete(np_array, to_delete, axis=0)
    final_pt = [np_array[-2],np_array[-1],np_array[0],np_array[1]]
    
    angels = get_angels(final_pt)
    angle = angels[0]
    if (angle > 178 and angle < 182) or (angle > -2 and angle < 2):
        np_array = np.delete(np_array, [-1], axis=0)
        return np_array
    angle = angels[1]
    if (angle > 178 and angle < 182) or (angle > -2 and angle < 2):
        np_array[0] = np_array[-1]
        np_array = np.delete(np_array, [-1], axis=0)
        return np_array
    return np_array


# 使用dp算法化简 序列化后的vec
def simpfyTrans(vec):
    # 简化图像
    target_vec = vec
    pred_points = np.array([[i%96,i//96] for i in target_vec])
    pred_points = cv2.approxPolyDP(pred_points,4,True).squeeze()
    pred_points = np.concatenate((pred_points,[pred_points[0]]),axis=0)
    return_vec = [int(i[0]+i[1]*96) for i in pred_points]
    return return_vec

# 仅删除里的近的和共线的点 返回序列化后的vec
def simpfyTrans1(vec):
    # 简化图像
    target_vec = vec
    pred_points = np.array([[i%96,i//96] for i in target_vec])
    final_points = filter_points(pred_points)
    final_points = np.concatenate((final_points,[final_points[0]]),axis=0)
    return_vec = serialize(final_points)
    return None if return_vec==None else return_vec['indexs']


# simpfyTrans2专用的工具函数
def get_angels(np_array):
    angles = []
    for i in range(1, len(np_array)-1):
        v1 = np_array[i] - np_array[i-1]
        v2 = np_array[i+1] - np_array[i]
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            angle = 0
        else:
            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            if cos_angle < -1:
                cos_angle = -1
            elif cos_angle > 1:
                cos_angle = 1
            angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)
    return angles



####################################################################################################################
# 先用dp算法，然后计算主方向，根据主方向，将所有的直线进行旋转，如果和主方向夹角小于45度 旋转到水平，大于则旋转为垂直方向；
# 然后校正旋转过后的直线，如果是平行方向的，判断直线距离接近，则移动到共线，如果直线距离较大则在垂直方向进行连接
# 如果是垂直方向的，则把直线交叉点作为新的点。
# 校正后的图像基本能保持角为90度左右
# 使用方法示例：
'''
   for vec in some_vecs('org_osm1.json'):
      if len(vec)<10:
          continue
      plt.figure()
      show_vec_list([vec,simpfyTrans2(vec,6)])
      break
'''
def simpfyTrans2(vec,epsilon=2):
    pred_points = np.array([[i%96,i//96] for i in vec[:-1]])
    # contours = rdp(pred_points, epsilon=epsilon)
    contours = cv2.approxPolyDP(pred_points,epsilon,True).squeeze()
    # contours[:, 1] = h - contours[:, 1]
    # 轮廓规则化
    dists = []
    azis = []
    azis_index = []

    # 获取每条边的长度和方位角
    for i in range(contours.shape[0]):
        cur_index = i
        next_index = i+1 if i < contours.shape[0]-1 else 0
        prev_index = i-1
        cur_point = contours[cur_index]
        nest_point = contours[next_index]
        prev_point = contours[prev_index]

        dist = cal_dist(cur_point, nest_point)
        azi = azimuthAngle(cur_point, nest_point)

        dists.append(dist)
        azis.append(azi)
        azis_index.append([cur_index, next_index])
        
    # 以最长的边的方向作为主方向
    longest_edge_idex = np.argmax(dists)
    main_direction = azis[longest_edge_idex]

    # 方向纠正，绕中心点旋转到与主方向垂直或者平行
    correct_points = []
    para_vetr_idxs = []  # 0平行 1垂直
    for i, (azi, (point_0_index, point_1_index)) in enumerate(zip(azis, azis_index)):

        if i == longest_edge_idex:
            correct_points.append([contours[point_0_index], contours[point_1_index]])
            para_vetr_idxs.append(0)
        else:
            # 确定旋转角度
            rotate_ang = main_direction - azi

            if np.abs(rotate_ang) < 180/4:
                rotate_ang = rotate_ang
                para_vetr_idxs.append(0)
            elif np.abs(rotate_ang) >= 90-180/4:
                rotate_ang = rotate_ang + 90
                para_vetr_idxs.append(1)                 

            # 执行旋转任务
            point_0 = contours[point_0_index]
            point_1 = contours[point_1_index]
            point_middle = (point_0 + point_1) / 2

            if rotate_ang > 0:
                rotate_point_0 = Srotation_angle_get_coor_coordinates(point_0, point_middle, np.abs(rotate_ang))
                rotate_point_1 = Srotation_angle_get_coor_coordinates(point_1, point_middle, np.abs(rotate_ang))
            elif rotate_ang < 0:
                rotate_point_0 = Nrotation_angle_get_coor_coordinates(point_0, point_middle, np.abs(rotate_ang))
                rotate_point_1 = Nrotation_angle_get_coor_coordinates(point_1, point_middle, np.abs(rotate_ang))
            else:
                rotate_point_0 = point_0
                rotate_point_1 = point_1
            correct_points.append([rotate_point_0, rotate_point_1])  
    correct_points = np.array(correct_points)

    # 相邻边校正，垂直取交点，平行平移短边或者加线
    final_points = []
    final_points.append(correct_points[0][0])
    for i in range(correct_points.shape[0]):
        cur_index = i
        next_index = i + 1 if i < correct_points.shape[0] - 1 else 0

        cur_edge_point_0 = correct_points[cur_index][0]
        cur_edge_point_1 = correct_points[cur_index][1]
        next_edge_point_0 = correct_points[next_index][0]
        next_edge_point_1 = correct_points[next_index][1]

        cur_para_vetr_idx = para_vetr_idxs[cur_index]
        next_para_vetr_idx = para_vetr_idxs[next_index]

        if cur_para_vetr_idx != next_para_vetr_idx:
            # 垂直取交点
            L1 = line(cur_edge_point_0, cur_edge_point_1)
            L2 = line(next_edge_point_0, next_edge_point_1)

            # print(L1,L2)

            point_intersection = intersection(L1, L2)
            final_points.append(point_intersection)

        elif cur_para_vetr_idx == next_para_vetr_idx:
            # 平行分两种，一种加短线，一种平移，取决于距离阈值
            # print(cur_edge_point_0,cur_edge_point_1,next_edge_point_1)
            L1 = line(cur_edge_point_0, cur_edge_point_1)
            L2 = line(next_edge_point_0, next_edge_point_1)
            marg = par_line_dist(L1, L2)

            if marg < 0.5:
                # 平移
                point_move = point_in_line(next_edge_point_0[0], next_edge_point_0[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])
                final_points.append(point_move)
                # 更新平移之后的下一条边
                correct_points[next_index][0] = point_move
                correct_points[next_index][1] = point_in_line(next_edge_point_1[0], next_edge_point_1[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])


            else:
                # 加线
                add_mid_point = (cur_edge_point_1 + next_edge_point_0) / 2
                add_point_1 = point_in_line(add_mid_point[0], add_mid_point[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])
                add_point_2 = point_in_line(add_mid_point[0], add_mid_point[1], next_edge_point_0[0], next_edge_point_0[1], next_edge_point_1[0], next_edge_point_1[1])
                final_points.append(add_point_1)
                final_points.append(add_point_2)

    final_points = np.array(final_points)
    # final_points = filter_points(final_points)
    final_points = np.concatenate((final_points,[final_points[0]]),axis=0)
    return_vec = serialize(final_points)
    return None if return_vec==None else return_vec['indexs'] 



def show_vec(vec):
    pred_points = [[i%96,i//96] for i in vec]
    points_str = ['CLS']+[' '.join([str(j) for j in i]) for i in pred_points]+['SEP']
    x, y = zip(*pred_points)
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.cla()
    plt.plot(x, y, color='#6666ff', label='fungis')  # x横坐标 y纵坐标 ‘k-’线性为黑色
    plt.grid()  
    
def show_vec_list(vec_list,file_name='result.jpg'):
    vec_size = len(vec_list)
    img_size = np.ceil(np.sqrt(vec_size)).astype(np.int32)
    plt.rcParams['figure.figsize'] = (16,16)
    for i in range(img_size):
        for j in range(img_size):  
            index = i*img_size+j
            if index >= vec_size:
                return
            vec = vec_list[index]
            pred_points = [[t%96,t//96] for t in vec]
            points_str = ['CLS']+[' '.join([str(j) for j in i]) for i in pred_points]+['SEP']
            x, y = zip(*pred_points)
            plt.subplot(img_size, img_size, index+1)#当前画在第一行第2列图上
            plt.axis('off')
            # print('{}[{}] : {}'.format(index,len(vec),vec))
            plt.plot(x, y, color='#6666ff', label='fungis')  # x横坐标 y纵坐标 ‘k-’线性为黑色
            plt.grid() 
    plt.savefig(file_name)    


def show_json(json_path,num=1000):
    vec_list = []
    filename = json_path
    with open(filename,encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)['indexs']
            vec_list.append(l)
            if len(vec_list)>num: break
    show_vec_list(vec_list)


def get_vec_area(vec,draw=False):
    target_vec = vec
    pred_points = [[i%96,i//96] for i in target_vec]
    img = 250*np.ones([120,120,1],np.uint8)
    for index in range(len(pred_points)-1):
        cv2.line(img,pred_points[index],pred_points[index+1],100,1)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 按照面积将所有轮廓逆序排序
    contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    c = contours2[0]
    area = cv2.contourArea(c)
    if draw:
        img2 = 250*np.ones([120,120,1],np.uint8)   
        cv2.drawContours(img2, [c],0, 50, 1)
        cv2.imshow("New image",img) 
        cv2.imshow("Counter",img2)    
        cv2.waitKey(2000)
        cv2.destroyAllWindows() 
    return area            


def get_vec_area(vec,draw=False):
    target_vec = vec
    pred_points = [[i%96,i//96] for i in target_vec]
    img = 250*np.ones([120,120,1],np.uint8)
    for index in range(len(pred_points)-1):
        cv2.line(img,pred_points[index],pred_points[index+1],100,1)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 按照面积将所有轮廓逆序排序
    contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    c = contours2[0]
    area = cv2.contourArea(c)
    if draw:
        img2 = 250*np.ones([120,120,1],np.uint8)   
        cv2.drawContours(img2, [c],0, 50, 1)
        cv2.imshow("New image",img) 
        cv2.imshow("Counter",img2)    
        cv2.waitKey(2000)
        cv2.destroyAllWindows() 
    return area   


# def one_shape_scale(points):
#     x,y = zip(*points)
#     x_min,x_max = np.min(x),np.max(x)
#     y_min,y_max = np.min(y),np.max(y)
#     x_range = x_max-x_min
#     y_range = y_max-y_min
#     return [x_range,y_range]
    
# def serialize(vec_border,max_scale,map_scale=(95,95)):
#     all_points = vec_border
#     x,y = zip(*all_points)
#     base_point = [np.min(x),np.min(y)]
#     serialized_border = {}
#     serialized_border['points'] = []
#     serialized_border['indexs'] = []
#     for each_point in vec_border:
#         x,y = each_point
#         x = int((x-base_point[0])*map_scale[0]/max_scale[0])
#         y = int((y-base_point[1])*map_scale[1]/max_scale[1])
#         serialized_border['points'].append((x,y))
#         serialized_border['indexs'].append((x+y*(map_scale[0]+1)))
#         if x<0 or y<0 or x>map_scale[0] or y>map_scale[1]:
#             return None 
#     return serialized_border   


class data_args:
    def __init__(self,rotate,rotate_value,translate,scale,shear,resize,hflip,vflip):
        self.rotate = rotate
        self.rotate_value = rotate_value
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resize = resize
        self.HFlip = hflip
        self.VFlip = vflip

# 从1中复制过来的
# 这里是读取vec(格式为[4,123,3,2,5,,88,99,4]类似的)，然后进行仿射变换等一系列变换，返回的也是此种格式
class img_trans():
    def __init__(self,img_size,args=data_args(1,45,1,1,1,1,0,0)):
        self.img_size = img_size
        self.args = args


    def apply_img_trans(self,vec,args=data_args(1,45,1,1,1,1,0,0)):
        self.args = args
        img = self._gen_pic_from_vec(vec)
        img_aug = self._augment(img)
        img_vec = self._img2vec(img_aug)
        return img_vec


    def _gen_pic_from_vec(self,vec):
        im = np.ones([320,320,3],np.uint8)
        img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        fillColor = (255,0,0)
        draw = ImageDraw.Draw(img_PIL)
        for index in range(len(vec)-1):
            draw.line((vec[index][0]+120,vec[index][1]+120,vec[index+1][0]+120,vec[index+1][1]+120),fill=fillColor,width=4)
        # img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
        # kernel = np.ones((5, 5), np.uint8)
        # img_dilate = cv2.dilate(img, kernel, 2)
        # # show_cv_img(img) 
        # img_PIL = Image.fromarray(cv2.cvtColor(img_dilate,cv2.COLOR_BGR2RGB))
        self.img = img_PIL  
        return img_PIL


    def _img2vec(self,img):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 10,255,cv2.THRESH_BINARY)
        # 寻找二值化图中的轮廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # img = cv2.cvtColor(np.asarray(img4),cv2.COLOR_RGB2BGR)
        cnt = contours[0]
        approx = cv2.approxPolyDP(cnt,4,True)
        # print(approx.shape)
        cv2.polylines(img, [approx], True, (255, 255, 0), 2)
        # show_cv_img(img) 
        return_array = np.squeeze(approx) 
        return_array = np.concatenate((return_array,[return_array[0]]))
        return_vec = serialize(return_array)
        return_vec = return_vec['indexs']
        return  return_vec          



    def _augment(self,img):        
        img0 = img.copy()
        if random.random()>0.8 and self.args.rotate:
            img1 = T.RandomAffine(degrees=(self.args.rotate_value-15,self.args.rotate_value+15))(img0)
        else:
            img1 = img0
        if random.random()>1 and self.args.translate:
            img2 = T.RandomAffine(degrees=0,translate=(0.2,0.2))(img1)
        else:
            img2 = img1
        if random.random()>0.5 and self.args.scale:
            img3 = T.RandomAffine(degrees=0,scale=(0.9,1.1))(img2)
        else:
            img3 = img2
        if random.random()>0.5 and self.args.shear:
            img4 = T.RandomAffine(degrees=0,shear=(-10,10))(img3)
        else:
            img4 = img3
        if random.random()>1 and self.args.resize:
            scale_size = [int(self.img_size[0]*(random.random()+0.5)),int(self.img_size[1]*(random.random()+0.5))]
            img5=T.Resize(scale_size)(img4)
        else:
            img5 = img4
        if self.args.HFlip:
            img5 = T.RandomHorizontalFlip(p=1)(img5)
        if self.args.VFlip:
            img5 = T.RandomVerticalFlip(p=1)(img5)            
        return img5    

#参数格式为(rotate,rotate_value,translate,scale,shear,resize,hflip,vflip)
params = [(1,45,1,1,1,1,0,1),(1,45,1,1,1,1,1,0),(1,45,1,1,1,1,0,0),(1,45,1,1,1,1,1,1),(1,-45,1,1,1,1,0,1),(1,-45,1,1,1,1,1,0),(1,-45,1,1,1,1,0,0),(1,-45,1,1,1,1,1,1),\
          (1,90,1,1,1,1,0,1),(1,90,1,1,1,1,1,0),(1,90,1,1,1,1,0,0),(1,90,1,1,1,1,1,1),(1,-90,1,1,1,1,0,1),(1,-90,1,1,1,1,1,0),(1,-90,1,1,1,1,0,0),(1,-90,1,1,1,1,1,1)]


class generate_similar_vec:
    def __init__(self,Scale=True,Move=True,Tuning=True,Delete=True,Simpfy=True,FangShe=True):
        self.scale_trans = Scale
        self.move_trans = Move
        self.tuning_trans = Tuning
        self.del_trans = Delete
        self.simpfy_trans = Simpfy
        self.apply_imgtrans = FangShe
        self.apply_img_trans = img_trans((300,300),)

    def scaleTrans(self,vec):
        # 缩放图像
        target_vec = vec
        scale = np.random.uniform(0.75,0.999)
        pred_points = [[int(i%96*scale),int(i//96*scale)] for i in target_vec]
        return_vec = [i[0]+i[1]*96 for i in pred_points]
        return return_vec
        # show_vec_list([target_vec,handled_vec])    

    def simpfyTrans(self,vec):
        # 简化图像
        target_vec = vec
        pred_points = np.array([[i%96,i//96] for i in target_vec])
        pred_points = cv2.approxPolyDP(pred_points,6,True).squeeze()
        pred_points = np.concatenate((pred_points,[pred_points[0]]),axis=0)
        return_vec = [i[0]+i[1]*96 for i in pred_points]
        return return_vec

    def applyImgTrans(self,vec):
        # 使用图像的方法实现旋转和仿射变换
        img_trans_param = random.choice(params)
        img_trans_param = data_args(*img_trans_param)
        pred_points = np.array([[i%96,i//96] for i in vec])
        vec = self.apply_img_trans.apply_img_trans(pred_points,img_trans_param)
        return vec

    def moveTrans(self,vec):
        # 平移图像
        target_vec = vec
        pred_points = [[i%96,i//96] for i in target_vec]
        x, y = zip(*pred_points)
        x_range = (-1*min(x),96-max(x))
        y_range = (-1*min(y),96-max(y))
        x_shift = np.random.randint(x_range[0],x_range[1])
        y_shift = np.random.randint(y_range[0],y_range[1])
        return_point = [[i%96+x_shift,i//96+y_shift] for i in target_vec]
        return_vec = [i[0]+i[1]*96 for i in return_point]
        return return_vec
        # show_vec_list([target_vec,return_vec]) 

    def tuningTrans(self,vec):
        # 微调个别点
        target_vec = vec
        pred_points = [[i%96,i//96] for i in target_vec]
        shift_index = random.sample(range(1,len(target_vec)-1),min(8,len(target_vec)-2))
        for index in shift_index:
            if pred_points[index][0]<4 or pred_points[index][0]>92 or pred_points[index][1]<4 or pred_points[index][1]>92:
                continue
            else:
                pred_points[index] = [pred_points[index][0]+np.random.randint(-1,1),pred_points[index][1]+np.random.randint(-1,1)]
        return_vec = [i[0]+i[1]*96 for i in pred_points]        
        return return_vec
        # show_vec_list([target_vec,return_vec])

    def deleteTrans(self,vec):
        # 删除个别点
        target_vec = vec
        base_area = get_vec_area(target_vec)
        if base_area == 0:
            return target_vec
        deleteable_index = []
        for i in range(len(target_vec)):
            delete_vec = target_vec.copy()
            del delete_vec[i]
            area = get_vec_area(delete_vec)
            if np.abs(area/base_area-1)<0.005:
                deleteable_index.append(i)

        delete_vec = target_vec.copy()
        if len(deleteable_index)>0:
            del_index = random.sample(deleteable_index,min(12,np.random.randint(len(deleteable_index))))
            del_index.sort(reverse=True)
            for i in del_index:
                del delete_vec[i]    
        return delete_vec

    def generate_similar_vec(self,vec):
        target_vec = vec
        simpfy_flag = False

        if self.simpfy_trans: #首先进行化简
            if np.random.rand()>0.1:
                simpfy_flag = True
                target_vec = self.simpfyTrans(target_vec)
                
        if self.apply_imgtrans: #接着进行仿射变换和旋转
            if np.random.rand()>0.5:
                repair_vec = target_vec.copy()
                try:
                    target_vec = self.applyImgTrans(target_vec)
                except:
                    target_vec = repair_vec

        if self.simpfy_trans: # 再次应用化简，去掉多余的点
            if np.random.rand()>0.5:
                simpfy_flag = True
                target_vec = self.simpfyTrans(target_vec)


        if self.scale_trans:
            if np.random.rand()>0.6:
                target_vec = self.scaleTrans(target_vec)
        if self.move_trans:
            if np.random.rand()>0.6:
                target_vec = self.moveTrans(target_vec)   
        if self.tuning_trans:
            if np.random.rand()>0.6:
                target_vec = self.tuningTrans(target_vec)
        if self.del_trans and simpfy_flag==False: # 如果已经进行了化简，就不再进行删除个别不重要的点的操作
            if np.random.rand()>0.6:
                target_vec = self.deleteTrans(target_vec)   
        return target_vec

import math
def get_counters_angle(p1,p2,p3):
    v1 = p2 - p1
    v2 = p3 - p2
    ag1 = math.atan2(v1[1],v1[0])
    ag2 = math.atan2(v2[1],v2[0])
    angle = ag2 - ag1
    angle = math.degrees(angle)
    # angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
    return angle

def get_counter_index(vec):
    append_vec = np.concatenate([vec,[vec[1]]])
    ags = []
    index_counter = []
    for index in range(len(append_vec)-2):
        ag = get_counters_angle(append_vec[index],append_vec[index+1],append_vec[index+2])
        if ag > 180:
            ag = ag-360
        if ag < -180:
            ag = ag+360
        ags.append(ag)
        if ag>0 and ag<120:
            # if index >= len(append_vec)-3: index = -1
            index_counter.append(index+1)
    return index_counter 

def argument_vec_add(vec,contours_index):
    append_vec = np.concatenate([vec,[vec[1]]])
    p1,p2,p3 = append_vec[contours_index-1],append_vec[contours_index],append_vec[contours_index+1]
    
    vec_a = p2-p1
    vec_a_norm = vec_a/np.linalg.norm(vec_a)
    vec_b = p3-p2
    vec_b_norm = vec_b/np.linalg.norm(vec_b)
    # print(p1,p2,p3,np.linalg.norm(vec_a),np.linalg.norm(vec_b))
    if np.linalg.norm(vec_a)<6 or np.linalg.norm(vec_b)<6:
        return vec,0
    alptha = min(np.linalg.norm(vec_a)/2,(np.random.random()*5+10))
    beta= min(0.1*np.linalg.norm(vec_b),2+np.random.random()*8)
    gama = alptha + min(0.1*np.linalg.norm(vec_a),3+np.random.random()*8)
    delta = beta*alptha/(gama-alptha)
    insert_p1 = p2 - alptha*vec_a_norm
    insert_p2 = insert_p1 + beta * vec_b_norm
    insert_p3 = insert_p2 + gama * vec_a_norm
    insert_p4 = insert_p3 + delta * vec_b_norm
    insert_p5 = p2 + (beta+delta) * vec_b_norm
    argument_vec = np.concatenate([vec[:contours_index],[insert_p1,insert_p2,insert_p3,insert_p4,insert_p5],vec[contours_index+1:]])
    argument_vec = argument_vec.astype(np.int32)
    return argument_vec,4

def argument_vec_minus(vec,contours_index):
    append_vec = np.concatenate([vec,[vec[1]]])
    p1,p2,p3 = append_vec[contours_index-1],append_vec[contours_index],append_vec[contours_index+1]
    vec_a = p2-p1
    vec_a_norm = vec_a/np.linalg.norm(vec_a)
    vec_b = p3-p2
    vec_b_norm = vec_b/np.linalg.norm(vec_b)
    # print(p1,p2,p3,np.linalg.norm(vec_a),np.linalg.norm(vec_b))
    if np.linalg.norm(vec_a)<6 or np.linalg.norm(vec_b)<6:
        return vec,0
    alptha = min(np.linalg.norm(vec_a)/2,(np.random.random()*5+10))
    beta= min(0.3*np.linalg.norm(vec_b),2+np.random.random()*8)
    gama = alptha - max(0.1*np.linalg.norm(vec_a),0.3*alptha+np.random.random()*0.2*alptha)
    delta = beta*gama/(alptha-gama)+beta      # beta*alptha/(gama-alptha)
    insert_p1 = p2 - alptha*vec_a_norm
    insert_p2 = insert_p1 - beta * vec_b_norm
    insert_p3 = insert_p2 + gama * vec_a_norm
    insert_p4 = insert_p3 + delta * vec_b_norm
    insert_p5 = p2 + (delta - beta) * vec_b_norm
    argument_vec = np.concatenate([vec[:contours_index],[insert_p1,insert_p2,insert_p3,insert_p4,insert_p5],vec[contours_index+1:]])
    argument_vec = argument_vec.astype(np.int32)
    return argument_vec,4

def argument_by_one_index(vec,contour_index):
    # print(len(vec),contour_index)
    if np.random.random()<0.5:
        returninfo = argument_vec_add(vec,contour_index)
    else:
        returninfo = argument_vec_minus(vec,contour_index)
    return returninfo

def argument_by_all_index(vec,contours_index):
    contours_index_repaired = contours_index.copy()
    vec_t = vec.copy()
    for c_range in range(len(contours_index_repaired)):
        vec_t,repair_num = argument_by_one_index(vec_t,contours_index_repaired[c_range])
        # print(c_range,contours_index_repaired,repair_num)
        for i,value in enumerate(contours_index_repaired):
            if i == c_range:
                contours_index_repaired[i] = contours_index_repaired[i] + int(repair_num/2)
            if i > c_range:
                contours_index_repaired[i] = contours_index_repaired[i] + repair_num
        # print(c_range,contours_index_repaired,repair_num)
    return vec_t
# argument_vec = argument_by_all_index(vec,list(range(len(vec))[1:]))
# show_pt_list([argument_vec,vec])

def argument_by_PIL(vec,inout=0,show=False):
    vec = np.array(vec)
    if len(vec.shape)==2 and vec.shape[1]==2:
        vec = vec
    else:
        vec = vec2point(vec)
    if np.max(vec)<120:
        vec = vec * 2    
    im = np.ones([320,320,3],np.uint8)
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    fillColor = (255,0,0)
    draw = ImageDraw.Draw(img_PIL)
    for index in range(len(vec)-1):
        draw.line((vec[index][0]+60,vec[index][1]+60,vec[index+1][0]+60,vec[index+1][1]+60),fill=fillColor,width=4)
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    img_gray = cv2.dilate(img_gray, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    img_gray = cv2.erode(img_gray, kernel, iterations=3)
    # for _ in range(3):
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #     img_gray = cv2.dilate(img_gray, kernel, iterations=3)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    #     img_gray = cv2.erode(img_gray, kernel, iterations=3)
    # for _ in range(3):
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    #     img_gray = cv2.erode(img_gray, kernel, iterations=1)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    #     img_gray = cv2.dilate(img_gray, kernel, iterations=1)
    ret, thresh = cv2.threshold(img_gray, 10,255,cv2.THRESH_BINARY)
    # 寻找二值化图中的轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.cvtColor(np.asarray(img4),cv2.COLOR_RGB2BGR)
    # print(len(contours),contours[0].shape)
    if len(contours)>1:
        cnt = contours[1] if inout==1 else contours[0]
    else:
        cnt = contours[0]
    approx = cv2.approxPolyDP(cnt,2,True)
    # print(approx.shape)
    cv2.polylines(img, [approx], True, (255, 255, 0), 2)

    return_array = np.squeeze(approx) 
    return_array = np.concatenate((return_array,[return_array[0]]))
    return_array = (return_array-60)
    if show: show_img(img)
    return return_array/2

def argument_by_counter_PIL(vec,circle=1):
    vec_t = vec.copy()
    for c in range(circle):
        all_point_argument_vec = argument_by_all_index(vec_t,list(range(len(vec_t))[1:]))
        counter_PIL_argument_vec = argument_by_PIL(all_point_argument_vec,inout=c%2,show=False)
        vec_t = counter_PIL_argument_vec
    return vec_t