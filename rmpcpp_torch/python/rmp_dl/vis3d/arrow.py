import numpy as np
import open3d as o3d

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/5
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=cone_radius,
                                                        cone_height=cone_height,
                                                        cylinder_radius=cylinder_radius,
                                                        cylinder_height=cylinder_height, 
                                                        resolution=4,
                                                        cylinder_split=2)
                                                                                                                    
    return(mesh_frame)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        # In this case a and b are pointing in the same direction. Return unit matrix
        return np.eye(3) if c > 0 else -np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_arrow_geometry(origin, direction, scale=0.075):
    base = np.array([0, 0, -1]) 
    
    rotation = rotation_matrix_from_vectors(base, -direction)

    mesh = create_arrow(scale)

    mesh.rotate(rotation) 
    mesh.translate(origin)

    return mesh
