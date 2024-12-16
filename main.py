import gradio as gr
import geopandas as gpd
import os
import numpy as np
from viusual_shp import fullpart_show_shp_building, visualize_overlapped_buildings
from PIL import Image
import shapefile
import cv2
from regularization_bart import Regular_Bart
# from regularization_urban import Regular
from polygon_evaluator import SimplePolygonEvaluator, plot_method_comparison, plot_metrics_comparison, plot_area_change_scatter_comparison, plot_violin_comparison
import json

from core.dataset_visualizer import DatasetVisualizer


def create_interface():
    visualizer = DatasetVisualizer()
    
    with gr.Blocks() as interface:
        gr.Markdown("# Building Matching Results Visualization")
        
        # 数据集选择
        group_names = list(visualizer.data_manager.dataset_groups.keys())
        dataset_dropdown = gr.Dropdown(
            choices=group_names,
            label="Select Dataset Group",
            value=group_names[0] if group_names else None
        )
        
        # 原始图像
        original_image = gr.Image(label="Original Building Layout")
        
        # 标签数据
        label_text = gr.TextArea(label="Label Data", interactive=False)
        
        # 控制滑块
        with gr.Row():
            center_idx_slider = gr.Slider(
                minimum=0,
                maximum=1000,  # 将在更新时动态调整
                step=5,
                value=100,
                label="Center Building Index"
            )
            num_neighbors_slider = gr.Slider(
                minimum=10,
                maximum=1000,
                step=10,
                value=50,
                label="Number of Neighbors"
            )
        
        # 融合结果可视化（1x2网格）
        with gr.Row():
            merged_original = gr.Image(label="Original Buildings")
            merged_result = gr.Image(label="Merged Result")
        
        # 在融合结果可视化后添加叠加显示部分
        gr.Markdown("### Overlapped Visualization")
        
        with gr.Row():
            with gr.Column(scale=2):
                checkboxes = gr.CheckboxGroup(
                    choices=['regular_bart', 'regular_urban', 
                            'regular_bart_10Types', 'runtime_ev'],  # 添加新选项
                    label="Select Methods to Display",
                    value=['regular_bart']
                )
                display_mode = gr.Radio(
                    choices=['Show Matched', 'Show All'],
                    label="Display Mode",
                    value='Show Matched'
                )                
            with gr.Column(scale=1):
                with gr.Row():
                    iou_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.0,
                        label="IOU Threshold"
                    )
                with gr.Row():
                    regularity_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        label="Shape Regularity Threshold"
                    )
                with gr.Row():
                    fragmentation_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        label="Edge Fragmentation Threshold"
                    )
            with gr.Column(scale=1):
                force_refresh = gr.Button("Force Refresh")
                refresh = gr.Button("Refresh")
        
        # 叠加显示结果
        overlapped_image = gr.Image(label="Overlapped Results")
        
        # 添加评估区域
        gr.Markdown("### Evaluation Results")
        
        # 评估图表显示
        with gr.Row():
            area_change_plot = gr.Image(label="Area Change Rate")
            area_iou_plot = gr.Image(label="Area IOU")
            perimeter_plot = gr.Image(label="Perimeter Ratio")
            
        
        with gr.Row():
            point_ratio_plot = gr.Image(label="Point Count Ratio")
            shape_regularity_plot = gr.Image(label="Shape Regularity")
            edge_fragmentation_plot = gr.Image(label="Edge Fragmentation")

        with gr.Row():
            hausdorff_plot = gr.Image(label="Hausdorff Distance")


        with gr.Row():
            scatter_plot = gr.Image(label="Area Change vs IOU")
            method_comparison = gr.Image(label="Method Comparison")
        with gr.Row():
            violin_plot1 = gr.Image(label="Violin Plot 1")
            violin_plot2 = gr.Image(label="Violin Plot 2")
        
        # 更新函数
        def update_all(group_name, center_idx, num_neighbors, selected_methods, 
                      display_mode, iou_threshold, regularity_threshold, 
                      fragmentation_threshold, force_refresh_clicked=False):
            # 获取原始结果
            if force_refresh_clicked:
                # 清除内存缓存，但保留硬盘缓存
                visualizer.current_data = {}
                visualizer.cached_params = {
                    'center_idx': None,
                    'num_neighbors': None,
                    'display_buildings': None
                }
            
            orig, data = visualizer.update_visualization(group_name)
            # 获取融合结果、显示范围内建筑信息和evaluators
            merged_orig, merged_res, display_buildings, evaluators = visualizer.create_merged_visualization(
                group_name, int(center_idx), int(num_neighbors), float(iou_threshold),
                float(regularity_threshold), float(fragmentation_threshold))
            
            # 根据显示模式选择要显示的建筑
            display_dict = {}
            for method in display_buildings.keys():
                if method == 'original':
                    display_dict[method] = display_buildings[method]
                elif display_mode == 'Show Matched':
                    # 使用单一方法的结果
                    if method in ['todo']:
                        display_dict[method] = display_buildings[f'matched_{method}']
                    else:
                        display_dict[method] = display_buildings[method]
                else:  # Show Matched
                    display_dict[method] = display_buildings[method]
            
            # 计算显示范围
            all_points = []
            for building in display_dict['original']:
                if building is not None:
                    all_points.extend(building['points'])
            all_points = np.array(all_points)
            
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            x1, y1 = min_coords
            shape_x = 2500
            shape_y = int(shape_x * (max_coords[1] - min_coords[1]) / 
                         (max_coords[0] - min_coords[0]))
            scale = (shape_x/(max_coords[0]-min_coords[0]), 
                    shape_y/(max_coords[1]-min_coords[1]))
            
            overlapped = visualize_overlapped_buildings(
                display_dict,
                selected_methods,
                x1, y1, scale, shape_x, shape_y
            )
            overlapped_img = Image.fromarray(overlapped)
            
            # 使用已有的evaluators生成评估图表
            plots = {}
            if evaluators:
                # 检查是否需要为新选择的方法创建evaluator
                for method in selected_methods:
                    if method not in evaluators and method in display_buildings:
                        # 为新方法创建evaluator
                        print(f'评估器创建：{method}')
                        new_evaluator = SimplePolygonEvaluator(
                            display_buildings['original'],
                            display_buildings[method]
                        )

                        evaluators[method] = new_evaluator
                        # 更新data中的evaluators
                        data['evaluators'] = evaluators
                
                # 筛选选中的方法的evaluators
                selected_evaluators = {method: evaluator 
                                     for method, evaluator in evaluators.items() 
                                     if method in selected_methods}
                
                if selected_evaluators:
                    # 使用所有评估器生成对比图
                    plots['area_change'] = plot_metrics_comparison(
                        selected_evaluators,
                        'area_changes',
                        'Area Change Rate',
                        f'./evaluation_results/{group_name}_area_change.png'
                    )
                    plots['area_iou'] = plot_metrics_comparison(
                        selected_evaluators,
                        'area_ious',
                        'Area IOU',
                        f'./evaluation_results/{group_name}_area_iou.png'
                    )
                    plots['perimeter'] = plot_metrics_comparison(
                        selected_evaluators,
                        'perimeter_ratios',
                        'Perimeter Ratio',
                        f'./evaluation_results/{group_name}_perimeter.png'
                    )
                    plots['point_ratio'] = plot_metrics_comparison(
                        selected_evaluators,
                        'point_ratios',
                        'Point Count Ratio',
                        f'./evaluation_results/{group_name}_point_ratio.png'
                    )

                    plots['hausdorff'] = plot_metrics_comparison(
                        selected_evaluators,
                        'hausdorffs',
                        'Hausdorff Distance',
                        f'./evaluation_results/{group_name}_hausdorff.png'
                    )
                    
                    # 使用所有评估器生成散点图对比
                    plots['scatter'] = plot_area_change_scatter_comparison(
                        selected_evaluators,
                        f'./evaluation_results/{group_name}_scatter.png'
                    )
                    
                    # 生成两张小提琴图
                    violin_images = plot_violin_comparison(
                        selected_evaluators,
                        f'./evaluation_results/{group_name}_violin.png'
                    )
                    plots['violin1'] = violin_images[0]
                    plots['violin2'] = violin_images[1]
                    
                    plots['comparison'] = plot_method_comparison(
                        {method: evaluator.get_average_metrics() 
                         for method, evaluator in selected_evaluators.items()},
                        f'./evaluation_results/{group_name}_comparison.png'
                    )
                    
                    # 添加形状规则度的序列图
                    plots['shape_regularity'] = plot_metrics_comparison(
                        selected_evaluators,
                        'shape_regularity_percent',
                        'Shape Regularity',
                        f'./evaluation_results/{group_name}_shape_regularity.png'
                    )
                    
                    # 添加边长零碎度的序列图
                    plots['edge_fragmentation'] = plot_metrics_comparison(
                        selected_evaluators,
                        'edge_fragmentation_percent',
                        'Edge Fragmentation',
                        f'./evaluation_results/{group_name}_edge_fragmentation.png'
                    )
                    
                    plots['comparison'] = plot_method_comparison(
                        {method: evaluator.get_average_metrics() 
                         for method, evaluator in selected_evaluators.items()},
                        f'./evaluation_results/{group_name}_comparison.png'
                    )
            
            return [orig, data, 
                    merged_orig, merged_res, overlapped_img,
                    plots.get('area_change'), plots.get('area_iou'),
                    plots.get('perimeter'), plots.get('point_ratio'),
                    plots.get('shape_regularity'), plots.get('edge_fragmentation'),
                    plots.get('hausdorff'), plots.get('scatter'), plots.get('violin1'), plots.get('violin2'),
                    plots.get('comparison')]
        
        # 更新输出列表
        outputs = [original_image, label_text,
                   merged_original, merged_result, overlapped_image,
                   area_change_plot, area_iou_plot, perimeter_plot,
                   point_ratio_plot, shape_regularity_plot, edge_fragmentation_plot,
                   hausdorff_plot, scatter_plot, violin_plot1, violin_plot2, method_comparison]
        
        # 更新所有事件处理器
        for component in [dataset_dropdown, center_idx_slider, num_neighbors_slider, 
                           display_mode]:
            if isinstance(component, gr.Slider):
                component.release(
                    fn=update_all,
                    inputs=[dataset_dropdown, center_idx_slider, num_neighbors_slider, 
                           checkboxes, display_mode, iou_threshold, regularity_threshold, 
                           fragmentation_threshold, gr.State(False)],
                    outputs=outputs
                )
            else:
                component.change(
                    fn=update_all,
                    inputs=[dataset_dropdown, center_idx_slider, num_neighbors_slider, 
                           checkboxes, display_mode, iou_threshold, regularity_threshold, 
                           fragmentation_threshold, gr.State(False)],
                    outputs=outputs
                )
        
        refresh.click(
            fn=update_all,
            inputs=[dataset_dropdown, center_idx_slider, num_neighbors_slider, 
                    checkboxes, display_mode, iou_threshold, regularity_threshold, 
                    fragmentation_threshold,
                    gr.State(False)],  # 强制刷新
            outputs=outputs
        )
        force_refresh.click(
            fn=update_all,
            inputs=[dataset_dropdown, center_idx_slider, num_neighbors_slider, 
                    checkboxes, display_mode, iou_threshold, regularity_threshold, 
                    fragmentation_threshold,
                    gr.State(True)],  # 强制刷新
            outputs=outputs
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False)  # 禁用共享
