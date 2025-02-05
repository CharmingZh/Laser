import open3d as o3d
import numpy as np

def create_plane_mesh(plane_model, pcd, mesh_size=10.0, resolution=1):
    """
    创建一个与拟合平面相符的矩形网格用于可视化。

    Parameters:
    - plane_model: 平面模型参数 [a, b, c, d]，对应平面方程 ax + by + cz + d = 0
    - pcd: 点云数据，用于计算平面的中心点
    - mesh_size: 平面网格的大小
    - resolution: 网格分辨率

    Returns:
    - plane_mesh: 创建的平面网格对象
    """
    a, b, c, d = plane_model
    # 计算平面的中心点（使用点云的边界框中心）
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    # 平面的法向量
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # 找到平面上的两个正交向量
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # 定义平面网格的四个角点
    half_size = mesh_size / 2
    corners = [
        center + u * half_size + v * half_size,
        center - u * half_size + v * half_size,
        center - u * half_size - v * half_size,
        center + u * half_size - v * half_size
    ]

    # 创建网格三角形
    triangles = [
        [0, 1, 2],
        [0, 2, 3]
    ]

    # 创建顶点和三角形
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    # 为网格赋予颜色
    mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 灰色
    # 移除以下行，因为 TriangleMesh 对象没有 'transformation' 属性
    # mesh.transformation = np.identity(4)

    return mesh

def main():
    # 加载点云
    # 注意：这里只保留一个加载点云的路径，根据实际需要选择
    pcd = o3d.io.read_point_cloud("../temp_results/colored_chest.ply")  # 替换为你的点云文件路径
    # pcd = o3d.io.read_point_cloud("../03_pipeline/output.ply")  # 替换为你的点云文件路径

    # 定义期望的法向量（与xoy平面平行）
    desired_normal = np.array([0, 0, 1])
    threshold_angle = 3  # 最大允许的角度偏差（单位：度）
    max_iterations = 10000  # 最大迭代次数

    plane_model = None
    inliers = []

    for i in range(max_iterations):
        # 使用 RANSAC 提取平面
        plane_model_candidate, inliers_candidate = pcd.segment_plane(distance_threshold=0.5,
                                                                      ransac_n=3,
                                                                      num_iterations=1000)
        [a, b, c, d] = plane_model_candidate
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)  # 单位化

        # 计算法向量与期望法向量之间的角度
        dot_product = np.clip(np.dot(normal, desired_normal), -1.0, 1.0)
        angle = np.arccos(dot_product) * 180 / np.pi

        if angle < threshold_angle:
            plane_model = plane_model_candidate
            inliers = inliers_candidate
            print(f"找到符合约束的平面，法向量与xoy平面的夹角为 {angle:.2f} 度。")
            break
        else:
            # 移除当前提取的平面内点，继续下一次迭代
            pcd = pcd.select_by_index(inliers_candidate, invert=True)
            print(f"第 {i+1} 次迭代：平面法向量夹角 {angle:.2f} 度，不满足约束，继续搜索。")
    else:
        print("未能在最大迭代次数内找到满足约束的平面。")
        return

    # 输出平面方程
    [a, b, c, d] = plane_model
    print(f"平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # 提取平面内的点
    plane_cloud = pcd.select_by_index(inliers)
    plane_cloud.paint_uniform_color([1.0, 0, 0])  # 红色表示平面

    # 提取非平面点
    non_plane_cloud = pcd.select_by_index(inliers, invert=True)

    # 计算每个点到平面的距离的符号（点在法向量正方向上为正）
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)  # 单位化

    points = np.asarray(non_plane_cloud.points)
    distances = (points @ normal) + d  # 点到平面的代数距离

    # 分离点：距离 < -1 和 >= -1
    dist_split = -1

    above_indices = np.where(distances < -dist_split)[0]
    below_indices = np.where(distances >= -dist_split)[0]

    above_cloud = non_plane_cloud.select_by_index(above_indices)
    below_cloud = non_plane_cloud.select_by_index(below_indices)

    print(f"保留的点数量（距离 < -1）：{len(above_indices)}")
    print(f"被移除的点数量（距离 >= -1）：{len(below_indices)}")

    # 对被移除的点（距离 >= -1）进行颜色标记
    if len(below_indices) > 0:
        below_cloud.paint_uniform_color([0, 0, 1.0])  # 蓝色表示距离 >= -1 的点
    else:
        below_cloud = None

    # 对通过距离过滤的点（距离 < -1）进行聚类分析，保留点最多的那个类别
    if len(above_indices) == 0:
        print("没有保留的点可供聚类。")
        largest_cluster = None
        other_clusters = None
    else:
        print("开始进行聚类分析...")
        # 使用 DBSCAN 进行聚类
        # 参数 eps 和 min_points 需要根据实际数据进行调整
        labels = np.array(above_cloud.cluster_dbscan(eps=5, min_points=50, print_progress=True))

        if labels.size == 0:
            print("未找到任何聚类。")
            largest_cluster = None
            other_clusters = above_cloud
        else:
            # 统计每个簇的点数（排除噪声点，标签为 -1）
            unique_labels, counts = np.unique(labels, return_counts=True)
            valid = unique_labels != -1

            if not np.any(valid):
                print("所有聚类都被视为噪声。")
                largest_cluster = None
                other_clusters = above_cloud
            else:
                # 找到点最多的簇的标签
                largest_cluster_label = unique_labels[valid][np.argmax(counts[valid])]
                largest_cluster_size = counts[valid][np.argmax(counts[valid])]
                print(f"最大的簇标签: {largest_cluster_label}，包含点数: {largest_cluster_size}")

                # 选择最大的簇
                largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
                largest_cluster = above_cloud.select_by_index(largest_cluster_indices)
                largest_cluster.paint_uniform_color([0, 1.0, 0])  # 绿色表示保留的点
                print(f"保留的最大簇点数量: {len(largest_cluster_indices)}")

                # 选择其他簇（包括噪声点）
                other_clusters_indices = np.where((labels != largest_cluster_label) & (labels != -1))[0]
                noise_indices = np.where(labels == -1)[0]
                if len(other_clusters_indices) > 0 and len(noise_indices) > 0:
                    other_clusters = above_cloud.select_by_index(np.concatenate((other_clusters_indices, noise_indices)) )
                elif len(other_clusters_indices) > 0:
                    other_clusters = above_cloud.select_by_index(other_clusters_indices)
                elif len(noise_indices) > 0:
                    other_clusters = above_cloud.select_by_index(noise_indices)
                else:
                    other_clusters = None

                if other_clusters and len(other_clusters.points) > 0:
                    other_clusters.paint_uniform_color([1.0, 1.0, 0])  # 黄色表示被移除的聚类点
                else:
                    other_clusters = None

    # 创建平面网格用于可视化
    plane_mesh = create_plane_mesh(plane_model, pcd, mesh_size=10.0, resolution=1)

    # 准备用于可视化的几何体列表
    geometries = [plane_mesh, plane_cloud]

    if largest_cluster is not None:
        geometries.append(largest_cluster)
    if below_cloud is not None:
        geometries.append(below_cloud)
    if other_clusters is not None:
        geometries.append(other_clusters)

    # 可视化结果
    o3d.visualization.draw_geometries(geometries, window_name="点云分类结果",
                                      point_show_normal=False,
                                      width=1200, height=800,
                                      left=50, top=50,
                                      mesh_show_back_face=True)

    # 保存结果点云（仅保存保留的最大簇）
    if largest_cluster is not None:
        o3d.io.write_point_cloud("filtered_point_cloud.ply", largest_cluster)
        print("已保存过滤后的点云为 'filtered_point_cloud.ply'")
    else:
        print("没有保留的最大簇点云可供保存。")

if __name__ == "__main__":
    main()
