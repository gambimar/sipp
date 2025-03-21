import numpy as np
import sys
from SKEL.skel import kin_skel
import trimesh

all_bodies = ['toes_r', 'calcn_r', 'tibia_r', 'femur_r', 'pelvis', 'lumbar_body', 'humerus_r','ulna_r', 'hand_r',
              'toes_l', 'calcn_l', 'tibia_l', 'femur_l', 'humerus_l', 'ulna_l', 'hand_l']


# We start at the edges of the kinematic tree and work our way inwards
# _m represents the other (mirrored) joint
cut_per_segment = {
    'toes': {
        0: {
            "joint": "toes",
            "normal": {
                "type": "two_point",
                "points": ["calcn", "toes"] # order matters, always pointing towards the inside
            }
        },
        1: {
            "joint": ["toes_m", "toes"], # List means that we take the middle point
            "normal": {
                "type": "two_point",
                "points": ["toes_m", "toes"]
            },
        },
        2: { # Remove everything above the knee - the head would come into frame otherwise
            "joint": "tibia", # List means that we take the middle point
            "normal": {
                "type": "two_point",
                "points": ["tibia", "toes"]
            }
        }
    },
    'calcn': { # We skip the talus later and make it massless (subtalar is locked anyways)
        0: { # Remove toes
            "joint": "toes",
            "normal": {
                "type": "two_point",
                "points": ["toes", "calcn"]
            }
        },
        1: { # Remove talus and upwards
            "joint": "talus",
            "normal": {
                "type": "two_point",
                "points": ["tibia", ["talus","toes"]] # btw. tibia is the origin of the tibia - so the knee
            }
        },
        2: { # Remove other foot
            "joint": ["talus_m", "talus"],
            "normal": {
                "type": "two_point",
                "points": ["talus_m", "talus"]
            },
        }
    },
    'tibia': {
        0: { # Remove calcn
            "joint": "talus",
            "normal": {
                "type": "two_point",
                "points": [["talus","toes"], "tibia"]
            }
        },
        1: { # Remove femur
            "joint": "tibia",
            "normal": {
                "type": "two_normal_avg",
                "points_outer": ["femur", "tibia"], # outside
                "points_inner": ["talus", "tibia"]
            }
        },
        2: { # Remove other leg
            "joint": ["tibia_m", "tibia"],
            "normal": {
                "type": "two_point",
                "points": ["tibia_m", "tibia"]
            }
        }
    },
    'femur': {
        0: { # Remove femur
            "joint": "tibia",
            "normal": {
                "type": "two_normal_avg",
                "points_outer": ["talus", "tibia"],
                "points_inner": ["femur", "tibia"]
            }
        },
        1: { # Remove pelvis
            "joint": "femur",
            "normal": {
                "type": "two_point",
                "points": [["femur_m", "femur", "lumbar_body"], "femur"] # a list of points means that we take the average point
            }
        },
        2: { # Remove other leg
            "joint": ["femur", "femur_m"],
            "normal": {
                "type": "two_point",
                "points": ["femur_m", "femur"]
            }
        },
        3: { # Remove lumbar and higher
            "joint": "lumbar_body",
            "normal": {
                "type": "two_point",
                "points": ["lumbar_body", ["femur_m", "femur"]],
            }
        },
        4: { # Remove arm
            "joint": ['hand','femur'],
            "normal": {
                "type": "two_point",
                "points": ["hand", "femur"]
            }
        }
    },
    'pelvis': {
        0: { # Remove femur
            "joint": "femur",
            "normal": {
                "type": "two_point",
                "points": ["femur", ["femur_m", "femur", "lumbar_body"]] # a list of points means that we take the average point
            }
        },
        1: { # Remove other femur
            "joint": "femur_m",
            "normal": {
                "type": "two_point",
                "points": ["femur_m", ["femur_m", "femur", "lumbar_body"]] # a list of points means that we take the average point
            }
        },
        2: { # Remove lumbar and higher
            "joint": ["lumbar_body", "femur_m", "femur", "lumbar_body", "lumbar_body"], # Lumbar in opensim is much lower
            "normal": {
                "type": "two_normal_avg",
                "points_outer": ["thorax", "lumbar_body"],
                "points_inner": [["femur_m", "femur"], "lumbar_body"],
            },
        },
    },
    'lumbar_body': {
        0: { # Remove lumbar and higher
            "joint": ["lumbar_body", "femur_m", "femur", "lumbar_body", "lumbar_body"], # Lumbar in opensim is much lower
            "normal": {
                "type": "two_normal_avg",
                "points_outer": [["femur_m", "femur"], "lumbar_body"],
                "points_inner": ["thorax", "lumbar_body"],
            },
        },
        1: { # Remove arm
            "joint": "humerus",
            "normal": {
                "type": "two_point",
                "points": ["humerus", ["lumbar_body","thorax","thorax","thorax"]] # take some
            },
        },
        2: { # Remove other arm
            "joint": "humerus_m",
            "normal": {
                "type": "two_point",
                "points": ["humerus_m", ["lumbar_body","thorax","thorax","thorax"]] # take some
            },
        },
        3:  {
            "joint": ["head","thorax"],
            "normal": {
                "type": "two_point",
                "points": ["head", "lumbar_body"]
            },
            "safety_cut": True # The safety cut is there to make sure that the head is not cut off, so we cut it off and back on
        },
        4: { # Remove femur to be safe
            "joint": "femur",
            "normal": {
                "type": "two_point",
                "points": ["femur", ["femur_m", "femur", "lumbar_body"]] # a list of points means that we take the average point
            }
        },
        5: { # Remove other femur
            "joint": "femur_m",
            "normal": {
                "type": "two_point",
                "points": ["femur_m", ["femur_m", "femur", "lumbar_body"]] # a list of points means that we take the average point
            }
        },
    },
    "humerus": {
        0: { # Remove other arm
            "joint": "humerus",
            "normal": {
                "type": "two_point",
                "points": [["lumbar_body","thorax","thorax","thorax"], "humerus"] # take some
            },
        },
        1:  {
            "joint": "head",
            "normal": {
                "type": "two_point",
                "points": ["head", "lumbar_body"]
            },
        },
        2:  {
            "joint": "ulna",
            "normal": {
                "type": "two_normal_avg",
                "points_outer": ["hand", "ulna"],
                "points_inner": ["humerus", "ulna"],
            },
        },
    },
    "ulna": {
        0: { # Remove hand
            "joint": "hand",
            "normal": {
                "type": "two_point",
                "points": ["hand", "ulna"]
            },
        },
        1: {
            "joint": "ulna",
            "normal": {
                "type": "two_normal_avg",
                "points_outer": ["humerus", "ulna"],
                "points_inner": ["hand", "ulna"],
            },
        },
        2: {
            "joint": ["ulna", "femur"],
            "normal": {
                "type": "two_point",
                "points": ["femur", "ulna"]
            },
        }
    },
    "hand": {
        0: { # Remove ulna
            "joint": "hand",
            "normal": {
                "type": "two_point",
                "points": ["ulna", "hand"]
            },
        },
        1: { # Remove everything that can be lower body
            "joint": ["hand","femur"],
            "normal": {
                "type": "two_point",
                "points": ["femur", "hand"]
            },
        }
    },
} # Time to test this before going further


def parse_entry(item_, joints, side):
    c_side = "r" if side == "l" else "l"
    if isinstance(item_, str):
        if item_[-2:] == "_m":
            return joints[kin_skel.skel_joints_name.index(f"{item_[:-2]}_{c_side}")]
        elif item_ in ['lumbar_body', 'thorax', 'head', 'torso']:
            return joints[kin_skel.skel_joints_name.index(item_)]
        else:
            return joints[kin_skel.skel_joints_name.index(f"{item_}_{side}")]
    elif isinstance(item_, list):
        # Average the points
        points = [parse_entry(p, joints, side) for p in item_]
        return np.mean(points, axis=0)
    else:
        raise ValueError("Unknown item type")


def parse_normal(normal, joints, side):
    if normal['type'] == "two_point":
        p1 = parse_entry(normal['points'][0], joints, side)
        p2 = parse_entry(normal['points'][1], joints, side)
        return p2 - p1
    elif normal['type'] == "two_normal_avg":
        # Get both normals as a normalized vector
        p1 = parse_entry(normal['points_outer'][0], joints, side)
        p2 = parse_entry(normal['points_outer'][1], joints, side)
        n1 = p2 - p1
        n1 = n1 / np.linalg.norm(n1)
        p1 = parse_entry(normal['points_inner'][0], joints, side)
        p2 = parse_entry(normal['points_inner'][1], joints, side)
        n2 = p2 - p1
        n2 = n2 / np.linalg.norm(n2)
        return (n1 - n2) / 2
    else:
        raise ValueError("Unknown normal type")


def parse_mask(mask, joints, side, mesh):
    normal = parse_normal(mask['normal'], joints, side)
    joint = parse_entry(mask['joint'], joints, side)
    # Find all vertices on the mesh that are on the side of the normal
    vertices = mesh.vertices
    mask = np.dot(vertices - joint, normal) < 0
    # Get all faces indices that have vertices on "our" side
    face_idx = np.nonzero(mask[mesh.faces].all(axis=1))[0]
    return face_idx


def cut_segment(segment_name, meshes, joints, allow_convex_hull="default"):
    """
        Inputs:
            segment_name: str, the name of the segment
            meshes: list of trimesh.Trimesh, the meshes to cut
            joints: torch.Tensor, the joints, shape (N, 3)
        Outputs:
            list of trimesh.Trimesh, the cut meshes - or False if the segment is not in the cut_per_segment
    """
    if segment_name[-2:] in ["_m", "_l", "_r"]:
        segment_name, side = segment_name.split("_")
    else:
        side = "r"  # We need to pick a side, but that is symmetric anyways
    if segment_name not in cut_per_segment:
        if segment_name != "torso":
            return False
        
    if allow_convex_hull == "default":
        allow_convex_hull = True if segment_name in ['toes','calcn'] else False
    # Get the planes
    cuts = {}
    cuts_dict = cut_per_segment[segment_name]
    for cut_idx, cut in cuts_dict.items():
        cuts[cut_idx] = {}
        cuts[cut_idx]['normal'] = parse_normal(cut['normal'], joints, side)
        cuts[cut_idx]['joint'] = parse_entry(cut['joint'], joints, side)
        # Check if mask is present
        if 'safety_cut' in cut:
            cuts[cut_idx]['safety_cut'] = cut['safety_cut']

    # Cut the meshes
    cut_meshes = []
    for mesh in meshes:
        safety_copy = mesh.copy()
        for cut_idx, cut in cuts.items():
            # Cut the mesh
            if 'safety_cut' in cut:
                cut_meshes.append(
                    safety_copy.slice_plane(plane_origin=cut['joint'], plane_normal=-cut['normal'], cap=True))
            mesh = mesh.slice_plane(plane_origin=cut['joint'], plane_normal=cut['normal'], cap=True)
        if mesh.is_watertight:
            cut_meshes.append(mesh)
        else:
            print(f"Warning: resulting mesh {segment_name} is not watertight")
            submeshes = mesh.split(only_watertight=False)
            for submesh in submeshes:
                if submesh.is_watertight:
                    cut_meshes.append(submesh)
                else:
                    submesh.fill_holes()
                    submesh.fix_normals()
                    trimesh.repair.fix_winding(submesh)
                    trimesh.repair.fix_inversion(submesh)
                    if submesh.is_watertight:
                        print('Repaired with trimesh')
                        cut_meshes.append(submesh)
                    else:
                        if allow_convex_hull:
                            print('Could not repair mesh, fallback to convex hull')
                            cut_meshes.append(trimesh.convex.convex_hull(submesh))
    if len(cut_meshes) == 0:
        return False
    complete_mesh = trimesh.util.concatenate(cut_meshes)
    return complete_mesh