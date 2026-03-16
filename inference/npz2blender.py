
#run with 
# ~/blender-4.1.1-linux-x64/blender  -t 8 --background --python ./inference/npz2blender.py -- --input_npz <NPZ_PATH> --out_path <OUTPUT_PATH> --export_alembic







import bpy
from bpy.app.handlers import persistent
import bpy_extras
import os
import numpy as np
# from gloss import *
# import trimesh
import json
import argparse
import sys
# import imageio.v3 as iio
from os import listdir
from os.path import isfile, join

import math
from mathutils import Matrix, Vector
import mathutils
import shutil
import time



path_cur_script=os.path.dirname(os.path.abspath(__file__))

def hex_to_rgb(hex_color):
    """Convert hex color (e.g., '#2b1b17') to normalized RGB tuple (0-1)"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, 1.0)

def apply_colorramp_colors(out_path, dark_color=None, light_color=None):
    """
    Apply dark and light colors to gradient nodes and variation intensity in hair material shader.
    
    Args:
        out_path: Output directory path
        dark_color: Hex color for dark/roots (e.g., '#2b1b17')
        light_color: Hex color for light/tips (e.g., '#d2b48c')
    """
    # Try to load from hair_colors.json if colors not provided
    if dark_color is None or light_color is None:
        hair_colors_path = os.path.join(out_path, "hair_colors.json")
        if os.path.exists(hair_colors_path):
            print(f"✓ Loading hair colors from: {hair_colors_path}")
            try:
                with open(hair_colors_path, 'r') as f:
                    color_data = json.load(f)
                dark_color = color_data.get("dark_root_color", dark_color)
                light_color = color_data.get("light_tip_color", light_color)
            except Exception as e:
                print(f"✗ Error reading hair_colors.json: {e}")
                return
    
    if not dark_color or not light_color:
        print("✗ No hair colors provided!")
        return
    
    print(f"\n{'='*80}")
    print("APPLYING HAIR COLORS TO MATERIAL NODES")
    print(f"{'='*80}")
    print(f"Dark (roots):  {dark_color}")
    print(f"Light (tips):  {light_color}")
    
    # Get hair material
    hair_mat = bpy.data.materials.get('Bgen_Hair_Shader')
    if not hair_mat or not hair_mat.use_nodes:
        print("✗ Bgen_Hair_Shader material not found or doesn't use nodes!")
        return
    
    node_tree = hair_mat.node_tree
    dark_rgb = hex_to_rgb(dark_color)
    light_rgb = hex_to_rgb(light_color)
    
    # Update Eevee Gradient
    if "Eevee Gradient" in node_tree.nodes:
        eevee_grad = node_tree.nodes["Eevee Gradient"]
        try:
            eevee_grad.color_ramp.elements[0].color = dark_rgb
            eevee_grad.color_ramp.elements[1].color = light_rgb
            print(f"✓ Eevee Gradient updated")
            print(f"    Dark: {dark_rgb}")
            print(f"    Light: {light_rgb}")
        except Exception as e:
            print(f"✗ Failed to update Eevee Gradient: {e}")
    else:
        print(f"✗ Eevee Gradient not found")
    
    # Update Eevee Variation (color input at index 7)
    if "Eevee Variation" in node_tree.nodes:
        eevee_var = node_tree.nodes["Eevee Variation"]
        try:
            # Set variation intensity to the light color as well
            eevee_var.inputs[7].default_value = light_rgb
            print(f"✓ Eevee Variation updated")
            print(f"    Color input[7]: {light_rgb}")
        except Exception as e:
            print(f"✗ Failed to update Eevee Variation: {e}")
    else:
        print(f"✗ Eevee Variation not found")
    
    # Update Cycles Gradient
    if "Cycles Gradient" in node_tree.nodes:
        cycles_grad = node_tree.nodes["Cycles Gradient"]
        try:
            cycles_grad.color_ramp.elements[0].color = dark_rgb
            cycles_grad.color_ramp.elements[1].color = light_rgb
            print(f"✓ Cycles Gradient updated")
            print(f"    Dark: {dark_rgb}")
            print(f"    Light: {light_rgb}")
        except Exception as e:
            print(f"✗ Failed to update Cycles Gradient: {e}")
    else:
        print(f"✗ Cycles Gradient not found")
    
    print(f"\n✓ Hair colors applied successfully!")

def export_alembic(out_alembic_path, resolution):
    print("-------------------------------------------")
    
    # bpy.ops.outliner.item_activate(deselect_all=True)
    bpy.data.objects["hair_01"].select_set(True)
    # hair = bpy.context.active_object
    hair = bpy.data.objects["hair_01"]
    bpy.context.view_layer.objects.active=hair


    start=time.time()
    for modif in hair.modifiers:
        print("applying",modif.name)
        bpy.context.view_layer.objects.active = hair
        bpy.ops.object.modifier_apply(modifier=modif.name)
    print("finished applying all geometry nodes")
    end=time.time()
    print("applying geometry nodes took", end-start)
    #shrinkwrap on the scalp (Wrong because it makes weird strands for the long hair)
    #default hair with t=8: 20s
    #default hair with t=16: 15s
    #50% strans with t=16: 6s

    #with shrinkwrap on the whole mesh
    #default hair with t=8: 43s
    #default hair with t=16: 34s
    #50% strans with t=16: 14s
    #50% strans, 50%points with t=16: 7s
    #50% strans, 25%points with t=16: 5s




    #conver particle
    bpy.ops.curves.convert_to_particle_system()

    # bpy.ops.outliner.item_activate(deselect_all=True)
    # bpy.context.space_data.context = 'PARTICLES'
    bpy.context.object.show_instancer_for_render = False
    bpy.context.object.show_instancer_for_viewport = False
    #I have no idea which one actually works to increase resolution so I change all
    # bpy.data.particles["ParticleSettings"].display_step = 7
    # bpy.data.particles["ParticleSettings"].hair_step = 7
    # bpy.data.particles["ParticleSettings"].render_step = 7
    bpy.data.particles["ParticleSettings"].display_step = resolution
    bpy.data.particles["ParticleSettings"].hair_step = resolution
    bpy.data.particles["ParticleSettings"].render_step = resolution
    

    #hide everything except scalp
    for obj in bpy.data.objects:
        print("obj", obj)
        if obj.name!="smplx_scalp_blender":
            obj.hide_render=True
            obj.hide_viewport=True
        else:
            print("smplx scalp blender doesn't get hidden")
    # for obj in bpy.scene.objects:
        # print("obj in scene", obj)



    bpy.data.objects['smplx_scalp_blender'].hide_render=False
    bpy.data.objects['smplx_scalp_blender'].hide_viewport=False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_render = False
    bpy.data.objects['smplx_scalp_blender'].show_instancer_for_viewport = False
    bpy.data.objects["smplx_scalp_blender"].select_set(True)



    bpy.ops.wm.alembic_export(filepath=out_alembic_path, check_existing=False, start=1, end=1,selected=True, visible_objects_only=True, uvs=False, packuv=False, normals=False, use_instancing=False, global_scale=1.0, export_hair=True, export_particles=False, as_background_job=False, evaluation_mode='VIEWPORT', init_scene_frame_range=True)

def update_existing_blend(blend_path, out_path, dark_color=None, light_color=None):
    """
    Update colors in an existing blend file without regenerating geometry.
    
    Args:
        blend_path: Path to existing .blend file
        out_path: Output directory path (for hair_colors.json)
        dark_color: Hex color for dark/roots
        light_color: Hex color for light/tips
    """
    print(f"\n{'='*80}")
    print(f"UPDATING EXISTING BLEND FILE: {blend_path}")
    print(f"{'='*80}")
    
    # Open existing blend file
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    print(f"✓ Opened blend file")
    
    # Apply color updates
    apply_colorramp_colors(out_path, dark_color, light_color)
    
    # Save the updated blend file
    print(f"\n✓ Saving updated blend file...")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"✓ Saved: {blend_path}")


def main():
    print("main")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True) #npz file to read and create a alembic from
    parser.add_argument('--out_path', required=True) #output path for the blender file and the alembic
    parser.add_argument('--export_alembic', action='store_true') #set it to true to also export an alembic file
    parser.add_argument('-ss', '--strands_subsample', type=float, default=1.0)  # perentage of strands we keep (1.0=keep all, 0.5=keep half, 0.25=keep quarter)
    parser.add_argument('-vs', '--vertex_subsample', type=float, default=1.0)  # perentage of vertices per strand to keep (1.0=keep all, 0.5=keep half, 0.25=keep quarter)
    parser.add_argument('-ar', '--alembic_resolution', type=int, default=7) #the resolution of the alembic, higher number means more points per strand (default=7 which is probably 2^7=128 points per strands)
    parser.add_argument('-sh', '--shrinkwrap', action='store_true') #set it to true to perform a shrinkwrap of the hair so that it avoids penetrating through the body
    parser.add_argument('--dark_color', type=str, default=None, help='Hex color for dark roots (e.g., #2b1b17)')
    parser.add_argument('--light_color', type=str, default=None, help='Hex color for light tips (e.g., #d2b48c)')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    print("strands_subsample", args.strands_subsample)
    print("vertex_subsample", args.vertex_subsample)
    print("alembic_resolution", args.alembic_resolution)
    print("shrinkwrap", args.shrinkwrap)

    # Always define output path early
    out_scene_path = os.path.join(args.out_path, "blender_scene.blend")

    #read npz 
    path_hair=args.input_npz
    do_export_alembic=args.export_alembic


    hair_geom=np.load(path_hair)
    points=hair_geom["positions"] #nr_strands x nr_points_per_strand x 3


    subsample_nr_strands=False
    #removes randomly x amount of strands or X nr of vertices
    if args.strands_subsample!=1.0 or args.vertex_subsample!=1.0:
        subsample_nr_strands=True
    if subsample_nr_strands:
        print("before ramoving random curves, points is ", points.shape) #nr_strands x nr_verts x3
        num_strands_to_keep = int(points.shape[0] * args.strands_subsample)
        strands_to_keep = np.random.choice(points.shape[0], num_strands_to_keep, replace=False)
        points = points[strands_to_keep, :, :].copy()
        print("after removing random curves, points is ", points.shape)

        #removing verts now 
        nr_verts_to_skip=int(np.floor(1.0/args.vertex_subsample))
        print("nr_verts_to_skip",nr_verts_to_skip)
        points = points[:, ::nr_verts_to_skip, :].copy()
        print("after removing consecurive vertices, points is ", points.shape)
    print("final points", points.shape)

    #open the blender file
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v24.blend")
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v25_with_shrinkwrap.blend")
    # if args.shrinkwrap:
    #     path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend")
    # else:
    #     path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v24.blend")
    # path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v27_blender36.blend")
    path_in_blend=os.path.join(path_cur_script,"./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend")
    bpy.ops.wm.open_mainfile(filepath=path_in_blend)


    #write new geometry
    print("creating geometry")
    bpy.data.objects["hair_01"].select_set(True)
    obj = bpy.data.objects.get("hair_01")
    bpy.context.view_layer.objects.active = obj
    # bpy.ops.object.mode_set(mode='EDIT')
    # #  Get the evaluated state of the object to account for geometry nodes and modifiers
    # depsgraph = bpy.context.evaluated_depsgraph_get()
    # depsgraph.update()
    # eval_obj = obj.evaluated_get(depsgraph)
    # curves_data=eval_obj.data
    # help(obj.data)
    curves_data=obj.data
    nr_strands=points.shape[0]
    # nr_strands=3000
    nr_points_per_strand=points.shape[1]

    #v4 faster
    points_per_curve = [nr_points_per_strand for i in range(nr_strands)]
    curves_data.add_curves(points_per_curve)
    # print("added curves")
    # exit(1)

    # Prepare a flat array for positions
    flat_points = points.reshape(-1, 3)  # Flatten points to a 2D array
    flat_points[:, [1, 2]] = flat_points[:, [2, 1]]  # Swap y and z
    flat_points[:, 1] *= -1  # Negate the y values

    # Assign the flat array directly
    curves_data.points.foreach_set("position", flat_points.flatten())
          


    if not args.shrinkwrap:
        bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")

   





    # Update the viewport to reflect changes
    obj.data.update_tag()
    obj.modifiers.update()
    # bpy.ops.object.mode_set(mode='OBJECT') 
    bpy.context.view_layer.update()

    # Apply hair colors to Color Ramp
    apply_colorramp_colors(args.out_path, args.dark_color, args.light_color)

    #save blend file
    print('saving .blend')
    bpy.ops.wm.save_as_mainfile(filepath=out_scene_path) 
    print('finished saving .blend')


    if do_export_alembic:
        out_path_alembic=os.path.join(args.out_path, "hair.abc")
        print("exporting hair to", out_path_alembic)
        export_alembic(out_path_alembic, args.alembic_resolution)

   
if __name__ == '__main__':
    main() 
