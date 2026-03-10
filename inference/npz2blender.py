
# Run with:
# ~/blender-4.1.1-linux-x64/blender -t 8 --background --python ./inference/npz2blender.py \
#   -- --input_npz <NPZ_PATH> --out_path <OUTPUT_PATH> --export_alembic

import bpy
import os
import numpy as np
import argparse
import sys
import time

path_cur_script = os.path.dirname(os.path.abspath(__file__))

# Objects in the base blend that are NOT hair-related and should be removed
NON_HAIR_TYPES  = {'CAMERA', 'LIGHT', 'LIGHT_PROBE'}
NON_HAIR_NAMES  = {'smplx_body', 'smplx_mesh', 'body', 'Body', 'SMPLX-mesh-neutral',
                   'smplx_neutral', 'Armature', 'armature',
                   'smplx_base_blender',
                   'cameraorbitlookat', 'cameraorbitslight',
                   'Plane'}
HAIR_OBJECTS    = {'hair_01', 'smplx_scalp_blender'}  # always keep these

# Collections whose contents should be excluded from the scene
EXCLUDE_COLLECTIONS = {'collision', 'collisionz_inflated', 'tmp'}


def purge_non_hair_objects():
    """Remove cameras, lights, planes, body meshes — keep only hair curves and scalp."""
    to_remove = []
    for obj in bpy.data.objects:
        if obj.name in HAIR_OBJECTS:
            continue
        if obj.type in NON_HAIR_TYPES:
            to_remove.append(obj)
            continue
        for keyword in NON_HAIR_NAMES:
            if keyword.lower() in obj.name.lower():
                to_remove.append(obj)
                break

    for obj in to_remove:
        print(f"  Removing object: {obj.name} ({obj.type})")
        bpy.data.objects.remove(obj, do_unlink=True)

    # Exclude collision / tmp collections from all view layers (hides contents)
    for scene in bpy.data.scenes:
        for view_layer in scene.view_layers:
            for layer_collection in view_layer.layer_collection.children:
                if layer_collection.name.lower() in EXCLUDE_COLLECTIONS:
                    layer_collection.exclude = True
                    print(f"  Excluded collection: {layer_collection.name}")

    # Clear world nodes (removes HDR / environment lighting)
    world = bpy.context.scene.world
    if world and world.use_nodes:
        world.node_tree.nodes.clear()
        world.use_nodes = False


def export_alembic(out_alembic_path, resolution):
    print("Exporting alembic...")

    hair = bpy.data.objects["hair_01"]
    bpy.context.view_layer.objects.active = hair
    hair.select_set(True)

    # Apply all geometry node modifiers
    t0 = time.time()
    for modif in hair.modifiers:
        print(f"  Applying modifier: {modif.name}")
        bpy.context.view_layer.objects.active = hair
        bpy.ops.object.modifier_apply(modifier=modif.name)
    print(f"  Geometry nodes applied in {time.time() - t0:.1f}s")

    # Convert hair curves to particle system (required for alembic hair export)
    bpy.ops.curves.convert_to_particle_system()
    bpy.context.object.show_instancer_for_render = False
    bpy.context.object.show_instancer_for_viewport = False
    bpy.data.particles["ParticleSettings"].display_step = resolution
    bpy.data.particles["ParticleSettings"].hair_step = resolution
    bpy.data.particles["ParticleSettings"].render_step = resolution

    # For alembic: show only scalp (hair is exported as particle system on it)
    for obj in bpy.data.objects:
        if obj.name != "smplx_scalp_blender":
            obj.hide_render = True
            obj.hide_viewport = True

    scalp = bpy.data.objects['smplx_scalp_blender']
    scalp.hide_render = False
    scalp.hide_viewport = False
    scalp.show_instancer_for_render = False
    scalp.show_instancer_for_viewport = False
    scalp.select_set(True)

    bpy.ops.wm.alembic_export(
        filepath=out_alembic_path,
        check_existing=False,
        start=1, end=1,
        selected=True,
        visible_objects_only=True,
        uvs=False, packuv=False, normals=False,
        use_instancing=False,
        global_scale=1.0,
        export_hair=True,
        export_particles=False,
        as_background_job=False,
        evaluation_mode='VIEWPORT',
        init_scene_frame_range=True,
    )
    print(f"  Alembic saved to {out_alembic_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_npz', required=True,
                        help='NPZ file with strand positions')
    parser.add_argument('--out_path', required=True,
                        help='Output directory for .blend and .abc files')
    parser.add_argument('--export_alembic', action='store_true',
                        help='Also export an alembic (.abc) hair file')
    parser.add_argument('-ss', '--strands_subsample', type=float, default=1.0,
                        help='Fraction of strands to keep (1.0 = all)')
    parser.add_argument('-vs', '--vertex_subsample', type=float, default=1.0,
                        help='Fraction of vertices per strand to keep (1.0 = all)')
    parser.add_argument('-ar', '--alembic_resolution', type=int, default=7,
                        help='Particle hair step resolution for alembic export')
    parser.add_argument('-sh', '--shrinkwrap', action='store_true',
                        help='Keep shrinkwrap modifier to prevent scalp penetration')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    # Load strand positions: (nr_strands, nr_pts_per_strand, 3)
    points = np.load(args.input_npz)["positions"]
    print(f"Loaded strands: {points.shape}")

    if args.strands_subsample != 1.0:
        n = int(points.shape[0] * args.strands_subsample)
        points = points[np.random.choice(points.shape[0], n, replace=False)].copy()
        print(f"  After strand subsample: {points.shape[0]} strands")

    if args.vertex_subsample != 1.0:
        skip = int(np.floor(1.0 / args.vertex_subsample))
        points = points[:, ::skip, :].copy()
        print(f"  After vertex subsample: {points.shape[1]} pts/strand")

    # Open base blend file (contains hair_01 curves + geometry nodes + scalp)
    path_in_blend = os.path.join(path_cur_script,
                                 "./assets/blender_vis_base_v26_with_shrinkwrap_full_base.blend")
    bpy.ops.wm.open_mainfile(filepath=path_in_blend)

    # Strip cameras, lights, SMPLX body, HDR world — keep hair + scalp only
    print("Purging non-hair objects...")
    purge_non_hair_objects()

    # Write hair geometry into hair_01 curves object
    print("Writing hair geometry...")
    obj = bpy.data.objects["hair_01"]
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    nr_strands, nr_pts = points.shape[0], points.shape[1]
    obj.data.add_curves([nr_pts] * nr_strands)

    flat = points.reshape(-1, 3).copy()
    flat[:, [1, 2]] = flat[:, [2, 1]]  # OpenGL → Blender coordinate swap
    flat[:, 1] *= -1
    obj.data.points.foreach_set("position", flat.flatten())

    if not args.shrinkwrap:
        bpy.ops.object.modifier_remove(modifier="Shrinkwrap Hair Curves")

    obj.data.update_tag()
    obj.modifiers.update()
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Save .blend
    out_blend = os.path.join(args.out_path, "blender_scene.blend")
    print(f"Saving {out_blend}")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)

    if args.export_alembic:
        export_alembic(os.path.join(args.out_path, "hair.abc"), args.alembic_resolution)

    print(f"Done → {args.out_path}")


if __name__ == '__main__':
    main()

