import numpy as np
import open3d as o3d


def drawSphere(center, radius, color=[0.0, 0.0, 0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def drawCone(bottom_center, top_position, color=[0.6, 0.6, 0.9]):
    cone = o3d.geometry.TriangleMesh.create_cone(
        radius=0.007,
        height=np.linalg.norm(top_position - bottom_center) + 1e-6)
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center) + 1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4:  # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    T = bottom_center + 5e-3 * line2
    # print(R)
    cone.transform(
        np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    return cone


def draw_shifted_pts(mesh_name, pts, weights=None):
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    pred_joints = o3d.geometry.PointCloud()
    pred_joints.points = o3d.utility.Vector3dVector(pts)
    if weights is None:
        color_joints = [[1.0, 0.0, 0.0] for i in range(len(pts))]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('YlOrRd')
        # weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        # weights = 1 / (1 + np.exp(-weights))
        color_joints = cmap(weights.squeeze())
        color_joints = color_joints[:, :-1]
    pred_joints.colors = o3d.utility.Vector3dVector(color_joints)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    vis.add_geometry(pred_joints)

    param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    # vis.run()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    # param = ctr.convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def show_obj_skel(mesh_name, root):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    vis.add_geometry(drawSphere(root.pos, 0.01, color=[0.1, 0.1, 0.1]))
    this_level = root.children
    while this_level:
        next_level = []
        for p_node in this_level:
            vis.add_geometry(drawSphere(p_node.pos, 0.008, color=[1.0, 0.0, 0.0]))  # [0.3, 0.1, 0.1]
            vis.add_geometry(drawCone(np.array(p_node.parent.pos), np.array(p_node.pos)))
            next_level += p_node.children
        this_level = next_level

    vis.run()

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image
