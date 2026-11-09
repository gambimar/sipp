import trimesh
import tempfile
import subprocess
import os

def clean_mesh_with_marching_cubes(mesh: trimesh.Trimesh, resolution=64) -> trimesh.Trimesh:
    """
    Clean a mesh with self-intersections using Blender's marching cubes implementation.
    This converts the mesh to a volume and back to a mesh, removing self-intersections.
    """
    import bpy
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.stl")
        output_path = os.path.join(tmpdir, "cleaned.stl")
        
        # Export mesh
        mesh.export(input_path)
        
        # Clean up scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Import mesh
        bpy.ops.import_mesh.stl(filepath=input_path)
        obj = bpy.context.selected_objects[0]
        obj.name = "OriginalMesh"
        
        # Apply transforms
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # Convert mesh to volume
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Add remesh modifier with voxel mode (which uses marching cubes internally)
        remesh_mod = obj.modifiers.new(name="Remesh", type='REMESH')
        remesh_mod.mode = 'VOXEL'
        remesh_mod.voxel_size = 0.005  # Adjust based on your mesh scale
        remesh_mod.adaptivity = 0.0
        
        # Apply the modifier
        bpy.ops.object.modifier_apply(modifier=remesh_mod.name)
        
        # Export cleaned mesh
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.export_mesh.stl(filepath=output_path, use_selection=True)
        
        out = trimesh.load(output_path, force='mesh')

        print('Volume: Start:', mesh.volume, 'End:', out.volume)
        return out

def clean_mesh_with_geometry_nodes(mesh: trimesh.Trimesh, voxel_size=0.01) -> trimesh.Trimesh:
    """
    Alternative mesh cleaning using explicit geometry nodes (Blender 3.0+).
    This provides more control over the marching cubes process.
    """
    import bpy
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.stl")
        output_path = os.path.join(tmpdir, "cleaned.stl")
        
        # Export mesh
        mesh.export(input_path)
        
        # Clean up scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Import mesh
        bpy.ops.import_mesh.stl(filepath=input_path)
        obj = bpy.context.selected_objects[0]
        obj.name = "OriginalMesh"
        
        # Apply transforms
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # Create geometry nodes modifier
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        # Add geometry nodes modifier
        geo_mod = obj.modifiers.new(name="MeshToVolume", type='NODES')
        
        # Create a new geometry node group
        node_group = bpy.data.node_groups.new(name="MeshToVolumeClean", type='GeometryNodeTree')
        geo_mod.node_group = node_group
        
        # Set up proper input/output interfaces
        node_group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        
        # Add input and output nodes
        input_node = node_group.nodes.new('NodeGroupInput')
        output_node = node_group.nodes.new('NodeGroupOutput')
        
        # Position nodes
        input_node.location = (-300, 0)
        output_node.location = (300, 0)
        
        # Add mesh to volume node
        mesh_to_vol = node_group.nodes.new('GeometryNodeMeshToVolume')
        mesh_to_vol.location = (-100, 0)
        mesh_to_vol.resolution_mode = 'VOXEL_SIZE'
        mesh_to_vol.inputs['Voxel Size'].default_value = voxel_size
        
        # Add volume to mesh node (this is where marching cubes happens)
        vol_to_mesh = node_group.nodes.new('GeometryNodeVolumeToMesh')
        vol_to_mesh.location = (100, 0)
        vol_to_mesh.resolution_mode = 'VOXEL_SIZE'
        vol_to_mesh.inputs['Voxel Size'].default_value = voxel_size
        vol_to_mesh.inputs['Threshold'].default_value = 0.1
        vol_to_mesh.inputs['Adaptivity'].default_value = 0.0
        
        # Connect nodes
        node_group.links.new(input_node.outputs['Geometry'], mesh_to_vol.inputs['Mesh'])
        node_group.links.new(mesh_to_vol.outputs['Volume'], vol_to_mesh.inputs['Volume'])
        node_group.links.new(vol_to_mesh.outputs['Mesh'], output_node.inputs['Geometry'])
        
        # Apply the modifier
        bpy.ops.object.modifier_apply(modifier=geo_mod.name)
        
        # Export cleaned mesh
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.export_mesh.stl(filepath=output_path, use_selection=True)
        
        out = trimesh.load(output_path, force='mesh')
        print('Volume: Start:', mesh.volume, 'End:', out.volume)
        return out

def slice_mesh_with_blender(target_mesh: trimesh.Trimesh, cutter_mesh: trimesh.Trimesh, operation='DIFFERENCE', clean_mesh=False) -> trimesh.Trimesh:
    assert operation in {'DIFFERENCE', 'INTERSECT', 'UNION'}, "Invalid operation type"

    # Clean meshes with marching cubes if requested
    if clean_mesh:
        print("Cleaning target mesh with marching cubes...")
        target_mesh = clean_mesh_with_marching_cubes(target_mesh)

    with tempfile.TemporaryDirectory() as tmpdir:
        # File paths
        target_path = os.path.join(tmpdir, "target.stl")
        cutter_path = os.path.join(tmpdir, "cutter.stl")
        output_path = os.path.join(tmpdir, "result.stl")

        # Export both meshes
        target_mesh.export(target_path)
        cutter_mesh.export(cutter_path)

        # Create Blender script
        import bpy

        # Clean up
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Import target
        bpy.ops.object.select_all(action='DESELECT')
        before = set(bpy.data.objects)
        bpy.ops.import_mesh.stl(filepath=target_path)
        after = set(bpy.data.objects)
        target_obj = (after - before).pop()
        target_obj.name = "Target"

        # Import cutter
        bpy.ops.object.select_all(action='DESELECT')
        before = set(bpy.data.objects)
        bpy.ops.import_mesh.stl(filepath=cutter_path)
        after = set(bpy.data.objects)
        cutter_obj = (after - before).pop()
        cutter_obj.name = "Cutter"

        # Apply transforms
        for obj in [target_obj, cutter_obj]:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Boolean operation
        bpy.ops.object.select_all(action='DESELECT')
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        bool_mod = target_obj.modifiers.new(name="BooleanCut", type='BOOLEAN')
        bool_mod.object = cutter_obj
        bool_mod.operation = operation
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)

        # Export result
        bpy.ops.object.select_all(action='DESELECT')
        target_obj.select_set(True)
        bpy.ops.export_mesh.stl(filepath=output_path, use_selection=True)
        return trimesh.load(output_path)