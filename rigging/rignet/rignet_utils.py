import sys, os
import time
import shutil
import tempfile
import subprocess
import traceback

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

import open3d as o3d
import trimesh

import binvox_rw

__all__ = [
    'RigInfo',
    'TreeNode',
    'primMST_symmetry',
    'loadSkel_recur',
    'create_single_data',
    'inside_check',
    'sample_on_bone',
    'meanshift_cluster',
    'nms_meanshift',
    'flip',
    'increase_cost_for_outside_bone',
    'get_bones',
    'calc_geodesic_matrix',
    'post_filter',
    'assemble_skel_skin',
]

MESH_NORMALIZED = None


class Node(object):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos


class TreeNode(Node):
    def __init__(self, name, pos):
        super(TreeNode, self).__init__(name, pos)
        self.children = []
        self.parent = None


class RigInfo:
    """
    Wrap class for rig information
    """

    def __init__(self, filename=None):
        self.joint_pos = {}
        self.joint_skin = []
        self.root = None
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename, 'r') as f_txt:
            lines = f_txt.readlines()
        for line in lines:
            word = line.split()
            if word[0] == 'joints':
                self.joint_pos[word[1]] = [float(word[2]), float(word[3]), float(word[4])]
            elif word[0] == 'root':
                root_pos = self.joint_pos[word[1]]
                self.root = TreeNode(word[1], (root_pos[0], root_pos[1], root_pos[2]))
            elif word[0] == 'skin':
                skin_item = word[1:]
                self.joint_skin.append(skin_item)
        self.loadHierarchy_recur(self.root, lines, self.joint_pos)

    def loadHierarchy_recur(self, node, lines, joint_pos):
        for li in lines:
            if li.split()[0] == 'hier' and li.split()[1] == node.name:
                pos = joint_pos[li.split()[2]]
                ch_node = TreeNode(li.split()[2], tuple(pos))
                node.children.append(ch_node)
                ch_node.parent = node
                self.loadHierarchy_recur(ch_node, lines, joint_pos)

    def save(self, filename):
        with open(filename, 'w') as file_info:
            for key, val in self.joint_pos.items():
                file_info.write(
                    'joints {0} {1:.8f} {2:.8f} {3:.8f}\n'.format(key, val[0], val[1], val[2]))
            file_info.write('root {}\n'.format(self.root.name))

            for skw in self.joint_skin:
                cur_line = 'skin {0} '.format(skw[0])
                for cur_j in range(1, len(skw), 2):
                    cur_line += '{0} {1:.4f} '.format(skw[cur_j], float(skw[cur_j + 1]))
                cur_line += '\n'
                file_info.write(cur_line)

            this_level = self.root.children
            while this_level:
                next_level = []
                for p_node in this_level:
                    file_info.write('hier {0} {1}\n'.format(p_node.parent.name, p_node.name))
                    next_level += p_node.children
                this_level = next_level

    def save_as_skel_format(self, filename):
        fout = open(filename, 'w')
        this_level = [self.root]
        hier_level = 1
        while this_level:
            next_level = []
            for p_node in this_level:
                pos = p_node.pos
                parent = p_node.parent.name if p_node.parent is not None else 'None'
                line = '{0} {1} {2:8f} {3:8f} {4:8f} {5}\n'.format(hier_level, p_node.name, pos[0], pos[1], pos[2],
                                                                   parent)
                fout.write(line)
                for c_node in p_node.children:
                    next_level.append(c_node)
            this_level = next_level
            hier_level += 1
        fout.close()

    def normalize(self, scale, trans):
        for k, v in self.joint_pos.items():
            self.joint_pos[k] /= scale
            self.joint_pos[k] -= trans

        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                node.pos /= scale
                node.pos = (node.pos[0] - trans[0], node.pos[1] - trans[1], node.pos[2] - trans[2])
                for ch in node.children:
                    next_level.append(ch)
            this_level = next_level

    def get_joint_dict(self):
        joint_dict = {}
        this_level = [self.root]
        while this_level:
            next_level = []
            for node in this_level:
                joint_dict[node.name] = node.pos
                next_level += node.children
            this_level = next_level
        return joint_dict

    def adjacent_matrix(self):
        joint_pos = self.get_joint_dict()
        joint_name_list = list(joint_pos.keys())
        num_joint = len(joint_pos)
        adj_matrix = np.zeros((num_joint, num_joint))
        this_level = [self.root]
        while this_level:
            next_level = []
            for p_node in this_level:
                for c_node in p_node.children:
                    index_parent = joint_name_list.index(p_node.name)
                    index_children = joint_name_list.index(c_node.name)
                    adj_matrix[index_parent, index_children] = 1.
                next_level += p_node.children
            this_level = next_level
        adj_matrix = adj_matrix + adj_matrix.transpose()
        return adj_matrix


def normalize_obj(mesh_v):
    dims = [
        max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
        max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
        max(mesh_v[:, 2]) - min(mesh_v[:, 2])
    ]
    scale = 1.0 / max(dims)
    pivot = np.array([
        (min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
        (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2
    ])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = set()
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.add(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(neighbor_ids)
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def get_geo_edges(surface_geodesic, remesh_obj_v):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= 0.06).squeeze(1)
        if len(geodesic_ball_samples) > 10:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, 10, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def add_self_loops(
        edge_index, num_nodes=None):
    N = np.max(edge_index) + 1 if num_nodes is None else num_nodes
    loop_index = np.arange(N)
    loop_index = np.repeat(np.expand_dims(loop_index, 0), 2, axis=0)
    edge_index = np.concatenate([edge_index, loop_index], axis=1)
    return edge_index


def calc_surface_geodesic(mesh):
    # We denselu sample 4000 points to be more accuracy.
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    time1 = time.time()
    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, _] = dijkstra(conn_matrix, directed=False, indices=range(N),
                         return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    time2 = time.time()

    print('surface geodesic calculation: {} seconds'.format((time2 - time1)))
    return surface_geodesic


def create_single_data(mesh_filaname, vox_file=None):
    """
    create input data for the network. The data is wrapped by Data structure in pytorch-geometric library
    :param mesh_filaname: name of the input mesh
    :return: wrapped data, voxelized mesh, and geodesic distance matrix of all vertices
    """
    print("creating data for model {:s}".format(mesh_filaname))

    mesh = o3d.io.read_triangle_mesh(mesh_filaname)
    mesh.compute_triangle_normals()
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    mesh_v, translation_normalize, scale_normalize = normalize_obj(mesh_v)
    mesh_normalized = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh_v),
        triangles=o3d.utility.Vector3iVector(mesh_f))
    global MESH_NORMALIZED
    MESH_NORMALIZED = mesh_normalized

    # vertices
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = v.astype(np.float32)

    # topology edges
    print("     gathering topological edges.")
    tpl_e = get_tpl_edges(mesh_v, mesh_f).T
    tpl_e = add_self_loops(tpl_e, num_nodes=v.shape[0])
    tpl_e = tpl_e.astype(np.int64)

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = calc_surface_geodesic(mesh)

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = add_self_loops(geo_e, num_nodes=v.shape[0])
    geo_e = geo_e.astype(np.int64)

    # batch
    batch = np.zeros(len(v), dtype=np.int64)

    # voxel
    if vox_file is None:
        fo_normalized = tempfile.NamedTemporaryFile(suffix='_normalized.obj')
        fo_normalized.close()
        path = fo_normalized.name
        vox_file = os.path.splitext(path)[0] + '.binvox'

        o3d.io.write_triangle_mesh(path, mesh_normalized)
        try:
            if sys.platform.startswith("win"):
                binvox_exe = "binvox.exe"
            else:
                binvox_exe = "./binvox"

            if not os.path.isfile(binvox_exe):
                raise FileNotFoundError(
                    "binvox executable not found in {0}, please check RigNet path in the addon preferences")

            ret = subprocess.call([binvox_exe, "-d", "88", path])
            if ret == 0:
                with open(vox_file, 'rb') as fvox:
                    vox = binvox_rw.read_as_3d_array(fvox)
            else:
                no_file = mesh_filaname.replace('.obj', '_normalized.obj')
                shutil.copyfile(path, no_file)
                print("===============")
                print("The normalized file is saved to '%s'." % no_file)
                print("The binvox file should be created by run 'binvox -d 88 %s'." % no_file)
                print("===============")
                raise RuntimeError("failed to execute binvox")
        finally:
            os.unlink(path)
    else:
        with open(vox_file, 'rb') as fvox:
            vox = binvox_rw.read_as_3d_array(fvox)

    data = dict(
        batch=batch, pos=v[:, 0:3],
        tpl_edge_index=tpl_e, geo_edge_index=geo_e,
    )
    return data, vox, surface_geodesic, translation_normalize, scale_normalize


def inside_check(pts, vox):
    """
    Check where points are inside or outside the mesh based on its voxelization.
    :param pts: points to be checked
    :param vox: voxelized mesh
    :return: internal points, and index of them in the input array.
    """
    vc = (pts - vox.translate) / vox.scale * vox.dims[0]
    vc = np.round(vc).astype(int)
    ind1 = np.logical_and(np.all(vc >= 0, axis=1), np.all(vc < 88, axis=1))
    vc = np.clip(vc, 0, 87)
    ind2 = vox.data[vc[:, 0], vc[:, 1], vc[:, 2]]
    ind = np.logical_and(ind1, ind2)
    pts = pts[ind]
    return pts, np.argwhere(ind).squeeze()


def sample_on_bone(p_pos, ch_pos):
    """
    sample points on a bone
    :param p_pos: parent joint position
    :param ch_pos: child joint position
    :return: a array of samples on this bone.
    """
    ray = ch_pos - p_pos
    bone_length = np.sqrt(np.sum((p_pos - ch_pos) ** 2))
    num_step = np.round(bone_length / 0.01)
    i_step = np.arange(1, num_step + 1)
    unit_step = ray / (num_step + 1e-30)
    unit_step = np.repeat(unit_step[np.newaxis, :], num_step, axis=0)
    res = p_pos + unit_step * i_step[:, np.newaxis]
    return res


def meanshift_cluster(pts_in, bandwidth, weights=None, max_iter=20):
    """
    Meanshift clustering
    :param pts_in: input points
    :param bandwidth: bandwidth
    :param weights: weights per pts indicting its importance in the clustering
    :return: points after clustering
    """
    diff = 1e10
    num_iter = 1
    while diff > 1e-3 and num_iter < max_iter:
        Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
        K = np.maximum(bandwidth ** 2 - Y, np.zeros(Y.shape))
        if weights is not None:
            K = K * weights
        row_sums = K.sum(axis=0, keepdims=True)
        P = K / (row_sums + 1e-10)
        P = P.transpose()
        pts_in_prim = 0.3 * (np.matmul(P, pts_in) - pts_in) + pts_in
        diff = np.sqrt(np.sum((pts_in_prim - pts_in) ** 2))
        pts_in = pts_in_prim
        num_iter += 1
    return pts_in


def nms_meanshift(pts_in, density, bandwidth):
    """
    NMS to extract modes after meanshift. Code refers to sci-kit-learn.
    :param pts_in: input points
    :param density: density at each point
    :param bandwidth: bandwidth used in meanshift. Used here as neighbor region for NMS
    :return: extracted clusters.
    """
    Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
    sorted_ids = np.argsort(density)[::-1]
    unique = np.ones(len(sorted_ids), dtype=np.bool)
    dist = np.sqrt(Y)
    for i in sorted_ids:
        if unique[i]:
            neighbor_idxs = np.argwhere(dist[:, i] <= bandwidth)
            unique[neighbor_idxs.squeeze()] = 0
            unique[i] = 1  # leave the current point as unique
    pts_in = pts_in[unique]
    return pts_in


def minKey(key, mstSet, nV):
    # Initilaize min value
    min = sys.maxsize
    for v in range(nV):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v
    return min_index


def primMST_symmetry(graph, init_id, joints):
    """
    my modified prim algorithm to generate a tree as symmetric as possible.
    Not guaranteed to be symmetric. All heuristics.
    :param graph: pairwise cost matrix
    :param init_id: init node ID as root
    :param joints: joint positions J*3
    :return:
    """
    joint_mapping = {}
    left_joint_ids = np.argwhere(joints[:, 0] < -2e-2).squeeze(1).tolist()
    middle_joint_ids = np.argwhere(np.abs(joints[:, 0]) <= 2e-2).squeeze(1).tolist()
    right_joint_ids = np.argwhere(joints[:, 0] > 2e-2).squeeze(1).tolist()
    for i in range(len(left_joint_ids)):
        joint_mapping[left_joint_ids[i]] = right_joint_ids[i]
    for i in range(len(right_joint_ids)):
        joint_mapping[right_joint_ids[i]] = left_joint_ids[i]

    if init_id not in middle_joint_ids:
        # find nearest joint in the middle to be root
        if len(middle_joint_ids) > 0:
            nearest_id = np.argmin(
                np.linalg.norm(joints[middle_joint_ids, :] - joints[init_id, :][np.newaxis, :], axis=1))
            init_id = middle_joint_ids[nearest_id]

    nV = graph.shape[0]
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = [False] * nV
    # Make key init_id so that this vertex is picked as first vertex
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    while not all(mstSet):
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet, nV)
        # left cases
        if u in left_joint_ids and parent[u] in middle_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = parent[u]
                key[u2] = graph[u2, parent[u2]]
        elif u in left_joint_ids and parent[u] in left_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
        elif u in middle_joint_ids and parent[u] in left_joint_ids:
            # form loop
            u2 = None
        # right cases
        elif u in right_joint_ids and parent[u] in middle_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = parent[u]
                key[u2] = graph[u2, parent[u2]]
        elif u in right_joint_ids and parent[u] in right_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
        elif u in middle_joint_ids and parent[u] in right_joint_ids:
            # form loop
            u2 = None
        # middle case
        else:
            u2 = None

        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(nV):
            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if graph[u, v] > 0 and mstSet[v] == False and key[v] > graph[u, v]:
                key[v] = graph[u, v]
                parent[v] = u
            if u2 is not None and graph[u2, v] > 0 and mstSet[v] == False and key[v] > graph[u2, v]:
                key[v] = graph[u2, v]
                parent[v] = u2

    return parent, key


def loadSkel_recur(p_node, parent_id, joint_name, joint_pos, parent):
    """
    Converst prim algorithm result to our skel/info format recursively
    :param p_node: Root node
    :param parent_id: parent name of current step of recursion.
    :param joint_name: list of joint names
    :param joint_pos: joint positions
    :param parent: parent index returned by prim alg.
    :return: p_node (root) will be expanded to linked with all joints
    """
    for i in range(len(parent)):
        if parent[i] == parent_id:
            if joint_name is not None:
                ch_node = TreeNode(joint_name[i], tuple(joint_pos[i]))
            else:
                ch_node = TreeNode('joint_{}'.format(i), tuple(joint_pos[i]))
            p_node.children.append(ch_node)
            ch_node.parent = p_node
            loadSkel_recur(ch_node, i, joint_name, joint_pos, parent)


def increase_cost_for_outside_bone(cost_matrix, joint_pos, vox):
    """
    increase connectivity cost for bones outside the meshs
    """
    for i in range(len(joint_pos)):
        for j in range(i + 1, len(joint_pos)):
            bone_samples = sample_on_bone(joint_pos[i], joint_pos[j])
            bone_samples_vox = (bone_samples - vox.translate) / vox.scale * vox.dims[0]
            bone_samples_vox = np.round(bone_samples_vox).astype(int)

            ind1 = np.logical_and(
                np.all(bone_samples_vox >= 0, axis=1),
                np.all(bone_samples_vox < vox.dims[0], axis=1))
            bone_samples_vox = np.clip(bone_samples_vox, 0, vox.dims[0] - 1)
            ind2 = vox.data[bone_samples_vox[:, 0], bone_samples_vox[:, 1], bone_samples_vox[:, 2]]
            in_flags = np.logical_and(ind1, ind2)
            outside_bone_sample = np.sum(in_flags == False)

            if outside_bone_sample > 1:
                cost_matrix[i, j] = 2 * outside_bone_sample
                cost_matrix[j, i] = 2 * outside_bone_sample
            if np.abs(joint_pos[i, 0]) < 2e-2 and np.abs(joint_pos[j, 0]) < 2e-2:
                cost_matrix[i, j] *= 0.5
                cost_matrix[j, i] *= 0.5
    return cost_matrix


def flip(pred_joints):
    """
    symmetrize the predicted joints by reflecting joints on the left half space to the right
    :param pred_joints: raw predicted joints
    :return: symmetrized predicted joints
    """
    pred_joints_left = pred_joints[np.argwhere(pred_joints[:, 0] < -2e-2).squeeze(), :]
    pred_joints_middle = pred_joints[np.argwhere(np.abs(pred_joints[:, 0]) <= 2e-2).squeeze(), :]

    if pred_joints_left.ndim == 1:
        pred_joints_left = pred_joints_left[np.newaxis, :]
    if pred_joints_middle.ndim == 1:
        pred_joints_middle = pred_joints_middle[np.newaxis, :]

    pred_joints_middle[:, 0] = 0.0
    pred_joints_right = np.copy(pred_joints_left)
    pred_joints_right[:, 0] = -pred_joints_right[:, 0]
    pred_joints_res = np.concatenate((pred_joints_left, pred_joints_middle, pred_joints_right), axis=0)
    side_indicator = np.concatenate(
        (-np.ones(len(pred_joints_left)), np.zeros(len(pred_joints_middle)), np.ones(len(pred_joints_right))), axis=0)
    return pred_joints_res, side_indicator


def get_bones(skel):
    """
    extract bones from skeleton struction
    :param skel: input skeleton
    :return: bones are B*6 array where each row consists starting and ending points of a bone
             bone_name are a list of B elements, where each element consists starting and ending joint name
             leaf_bones indicate if this bone is a virtual "leaf" bone.
             We add virtual "leaf" bones to the leaf joints since they always have skinning weights as well
    """
    bones = []
    bone_name = []
    leaf_bones = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            p_pos = np.array(p_node.pos)
            next_level += p_node.children
            for c_node in p_node.children:
                c_pos = np.array(c_node.pos)
                bones.append(np.concatenate((p_pos, c_pos))[np.newaxis, :])
                bone_name.append([p_node.name, c_node.name])
                leaf_bones.append(False)
                if len(c_node.children) == 0:
                    bones.append(np.concatenate((c_pos, c_pos))[np.newaxis, :])
                    bone_name.append([c_node.name, c_node.name + '_leaf'])
                    leaf_bones.append(True)
        this_level = next_level
    bones = np.concatenate(bones, axis=0)
    return bones, bone_name, leaf_bones


def pts2line(pts, lines):
    '''
    Calculate points-to-bone distance. Point to line segment distance refer to
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    :param pts: N*3
    :param lines: N*6, where [N,0:3] is the starting position and [N, 3:6] is the ending position
    :return: origins are the neatest projected position of the point on the line.
             ends are the points themselves.
             dist is the distance in between, which is the distance from points to lines.
             Origins and ends will be used for generate rays.
    '''
    l2 = np.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, axis=1)
    origins = np.zeros((len(pts) * len(lines), 3))
    ends = np.zeros((len(pts) * len(lines), 3))
    dist = np.zeros((len(pts) * len(lines)))
    for l in range(len(lines)):
        if np.abs(l2[l]) < 1e-8:  # for zero-length edges
            origins[l * len(pts):(l + 1) * len(pts)] = lines[l][0:3]
        else:  # for other edges
            t = np.sum((pts - lines[l][0:3][np.newaxis, :]) * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :], axis=1) / \
                l2[l]
            t = np.clip(t, 0, 1)
            t_pos = lines[l][0:3][np.newaxis, :] + t[:, np.newaxis] * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :]
            origins[l * len(pts):(l + 1) * len(pts)] = t_pos
        ends[l * len(pts):(l + 1) * len(pts)] = pts
        dist[l * len(pts):(l + 1) * len(pts)] = np.linalg.norm(
            origins[l * len(pts):(l + 1) * len(pts)] - ends[l * len(pts):(l + 1) * len(pts)], axis=1)
    return origins, ends, dist


def calc_pts2bone_visible_mat(mesh, origins, ends):
    '''
    Check whether the surface point is visible by the internal bone.
    Visible is defined as no occlusion on the path between.
    :param mesh:
    :param surface_pts: points on the surface (n*3)
    :param origins: origins of rays
    :param ends: ends of the rays, together with origins, we can decide the direction of the ray.
    :return: binary visibility matrix (n*m), where 1 indicate the n-th surface point is visible to the m-th ray
    '''
    ray_dir = ends - origins
    RayMeshIntersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = RayMeshIntersector.intersects_location(origins, ray_dir + 1e-15)
    locations_per_ray = [locations[index_ray == i] for i in range(len(ray_dir))]
    min_hit_distance = []
    for i in range(len(locations_per_ray)):
        if len(locations_per_ray[i]) == 0:
            min_hit_distance.append(np.linalg.norm(ray_dir[i]))
        else:
            min_hit_distance.append(
                np.min(np.linalg.norm(locations_per_ray[i] - origins[i], axis=1))
            )
    min_hit_distance = np.array(min_hit_distance)
    distance = np.linalg.norm(ray_dir, axis=1)
    vis_mat = (np.abs(min_hit_distance - distance) < 1e-4)
    return vis_mat


def calc_geodesic_matrix(
        bones, mesh_v, surface_geodesic,
        use_sampling=False, decimation=3000, sampling=1500, mesh_normalized=None):
    """
    calculate volumetric geodesic distance from vertices to each bones
    :param bones: B*6 numpy array where each row stores the starting and ending joint position of a bone
    :param mesh_v: V*3 mesh vertices
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: an approaximate volumetric geodesic distance matrix V*B, were (v,b) is the distance from vertex v to bone b
    """

    if use_sampling:
        mesh0 = mesh_normalized if mesh_normalized else MESH_NORMALIZED
        mesh0 = mesh0.simplify_quadric_decimation(decimation)

        fo_simplified = tempfile.NamedTemporaryFile(suffix='_simplified.obj')
        fo_simplified.close()
        o3d.io.write_triangle_mesh(fo_simplified.name, mesh0)
        mesh_trimesh = trimesh.load(fo_simplified.name)
        os.unlink(fo_simplified.name)

        subsamples_ids = np.random.choice(len(mesh_v), np.min((len(mesh_v), sampling)), replace=False)
        subsamples = mesh_v[subsamples_ids, :]
        surface_geodesic = surface_geodesic[subsamples_ids, :][:, subsamples_ids]
    else:
        fo = tempfile.NamedTemporaryFile(suffix='.obj')
        fo.close()
        o3d.io.write_triangle_mesh(fo.name, mesh_normalized if mesh_normalized else MESH_NORMALIZED)
        mesh_trimesh = trimesh.load(fo.name)
        os.unlink(fo.name)
        subsamples = mesh_v

    origins, ends, pts_bone_dist = pts2line(subsamples, bones)
    pts_bone_visibility = calc_pts2bone_visible_mat(mesh_trimesh, origins, ends)
    pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(subsamples)).transpose()
    pts_bone_dist = pts_bone_dist.reshape(len(bones), len(subsamples)).transpose()

    # remove visible points which are too far
    for b in range(pts_bone_visibility.shape[1]):
        visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
        if len(visible_pts) == 0:
            continue
        threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
        pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False

    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                visible_matrix[r, c] = 8.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]

    if use_sampling:
        nn_dist = np.sum((mesh_v[:, np.newaxis, :] - subsamples[np.newaxis, ...]) ** 2, axis=2)
        nn_ind = np.argmin(nn_dist, axis=1)
        visible_matrix = visible_matrix[nn_ind, :]

    return visible_matrix


def post_filter(skin_weights, topology_edge, num_ring=1):
    skin_weights_new = np.zeros_like(skin_weights)
    for v in range(len(skin_weights)):
        adj_verts_multi_ring = []
        current_seeds = [v]
        for r in range(num_ring):
            adj_verts = []
            for seed in current_seeds:
                adj_edges = topology_edge[:, np.argwhere(topology_edge == seed)[:, 1]]
                adj_verts_seed = list(set(adj_edges.flatten().tolist()))
                adj_verts_seed.remove(seed)
                adj_verts += adj_verts_seed
            adj_verts_multi_ring += adj_verts
            current_seeds = adj_verts
        adj_verts_multi_ring = list(set(adj_verts_multi_ring))
        if v in adj_verts_multi_ring:
            adj_verts_multi_ring.remove(v)
        skin_weights_neighbor = [skin_weights[int(i), :][np.newaxis, :] for i in adj_verts_multi_ring]
        skin_weights_neighbor = np.concatenate(skin_weights_neighbor, axis=0)
        skin_weights_new[v, :] = np.mean(skin_weights_neighbor, axis=0)

    return skin_weights_new


def add_duplicate_joints(skel):
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            if len(p_node.children) > 1:
                new_children = []
                for dup_id in range(len(p_node.children)):
                    p_node_new = TreeNode(p_node.name + '_dup_{:d}'.format(dup_id), p_node.pos)
                    p_node_new.overlap = True
                    p_node_new.parent = p_node
                    p_node_new.children = [p_node.children[dup_id]]
                    # for user interaction, we move overlapping joints a bit to its children
                    p_node_new.pos = np.array(p_node_new.pos) + 0.03 * np.linalg.norm(
                        np.array(p_node.children[dup_id].pos) - np.array(p_node_new.pos))
                    p_node_new.pos = (p_node_new.pos[0], p_node_new.pos[1], p_node_new.pos[2])
                    p_node.children[dup_id].parent = p_node_new
                    new_children.append(p_node_new)
                p_node.children = new_children
            p_node.overlap = False
            next_level += p_node.children
        this_level = next_level
    return skel


def mapping_bone_index(bones_old, bones_new):
    bone_map = {}
    for i in range(len(bones_old)):
        bone_old = bones_old[i][np.newaxis, :]
        dist = np.linalg.norm(bones_new - bone_old, axis=1)
        ni = np.argmin(dist)
        bone_map[i] = ni
    return bone_map


def assemble_skel_skin(skel, attachment):
    bones_old, bone_names_old, _ = get_bones(skel)
    skel_new = add_duplicate_joints(skel)
    bones_new, bone_names_new, _ = get_bones(skel_new)
    bone_map = mapping_bone_index(bones_old, bones_new)
    skel_new.joint_pos = skel_new.get_joint_dict()
    skel_new.joint_skin = []

    for v in range(len(attachment)):
        vi_skin = [str(v)]
        skw = attachment[v]
        skw = skw / (np.sum(skw) + 1e-10)
        for i in range(len(skw)):
            if i == len(bones_old):
                break
            if skw[i] > 1e-5:
                bind_joint_name = bone_names_new[bone_map[i]][0]
                bind_weight = skw[i]
                vi_skin.append(bind_joint_name)
                vi_skin.append(str(bind_weight))
        skel_new.joint_skin.append(vi_skin)
    return skel_new
