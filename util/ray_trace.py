import trimesh
import numpy as np
from util.misc import timeit

FACE_PRESET = np.array([[2, 1, 0], [3, 2, 0],
                        [4, 5, 6], [4, 6, 7],
                        [0, 1, 4], [1, 5, 4],
                        [2, 3, 7], [7, 6, 2],
                        [3, 0, 4], [7, 3, 4],
                        [1, 2, 6], [6, 5, 1],
                        ])


def generate_rays_features(edge_features, target_tf):
    ones = np.ones(len(edge_features)).reshape(-1,1)
    edge_features = np.hstack([edge_features, ones])
    edge_features = (np.linalg.pinv(target_tf) @ edge_features.T).T[:, :-1]
    edge_features[:, 1] = -edge_features[:, 1]
    ray_origins = edge_features
    ray_directions = - edge_features / \
        np.linalg.norm(edge_features, axis=1).reshape(-1, 1)
    return ray_directions, ray_origins


def generate_rays():
    phi_vals = 16
    theta_vals = 50
    phi_range = np.linspace(-15, 15, phi_vals)  # Angle b/w xy plane and z axis
    theta_range = np.linspace(0, 359, theta_vals)  # planar angle (in xy plane)

    zcap = np.sin(phi_range * np.pi / 180)
    zcos = np.cos(phi_range * np.pi / 180)

    xcos = np.cos(theta_range * np.pi / 180)

    ycos = np.sin(theta_range * np.pi / 180)

    ray_directions = np.zeros((theta_vals*phi_vals, 3))
    ray_origins = np.zeros((theta_vals*phi_vals, 3))

    for i in range(phi_vals):
        start = i*theta_vals
        end = start + theta_vals
        ray_directions[start:end, 0] = xcos * zcos[i]
        ray_directions[start:end, 1] = ycos * zcos[i]
        ray_directions[start:end, 2] = zcap[i]

    return ray_origins, ray_directions


def get_mesh(traffic, lidar_tf):
    '''
    lidar_sen: The querry lidar position, might be different from actual lidar position
    '''
    all_vertices = None
    all_faces = None
    for itr, vehicle in enumerate(traffic):
        vehicle_tf = vehicle.get_transform().get_matrix()
        # Transformation of vehicle in lidar frame
        vehicle_lidar_tf = np.linalg.pinv(lidar_tf) @ vehicle_tf
        extent = vehicle.bounding_box.extent
        ex, ey, ez = extent.x, extent.y, extent.z  # Get from extent
        vertices = np.array([  # Homogenous
            [-ex, -ey, -ez, 1],
            [ex, -ey, -ez, 1],
            [ex, ey, -ez, 1],
            [-ex, ey, -ez, 1],

            [-ex, -ey, ez, 1],
            [ex, -ey, ez, 1],
            [ex, ey, ez, 1],
            [-ex, ey, ez, 1],
        ])
        vertices = (vehicle_lidar_tf @ vertices.T).T[:, :-1]

        if itr == 0:
            all_vertices = vertices
        else:
            all_vertices = np.vstack([all_vertices, vertices])

        if itr == 0:
            all_faces = FACE_PRESET
        else:
            all_faces = np.vstack([all_faces, FACE_PRESET + (itr)*8])

    mesh = trimesh.Trimesh(vertices=all_vertices,
                           faces=all_faces)
    return mesh


@timeit
def intersect_vis(mesh, ray_origins, ray_directions, vis=False):
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)

    print('The rays hit the mesh at coordinates:\n', len(locations))
    if vis:
        ray_visualize = trimesh.load_path(np.hstack((ray_origins,
                                                    ray_origins + ray_directions*10.0)).reshape(-1, 2, 3))
        # mesh.show()

        # unmerge so viewer doesn't smooth
        mesh.unmerge_vertices()
        # make mesh white- ish
        mesh.visual.face_colors = [255, 255, 255, 255]
        mesh.visual.face_colors[index_tri] = [255, 0, 0, 255]

        scene = trimesh.Scene([mesh,
                               ray_visualize])
        scene.show()

    return len(locations)
