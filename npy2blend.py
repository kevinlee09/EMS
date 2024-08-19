import os
import sys
import bpy
import argparse
import mathutils
import numpy as np

def init_obj(obj_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_scene.obj(filepath=obj_path)
    obj = bpy.context.scene.objects[0]
    bpy.context.view_layer.objects.active = obj

if __name__ == "__main__":
    # Get arguments from Blender's command line
    args = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_item", default="revision_013", type=str)
    args = parser.parse_args(args)  # Parse arguments from Blender

    npy_data_dir = os.path.join("example_data", args.data_item, "recon_fiber_final")


    head_mesh = os.path.join("example_data", args.data_item, f"align_mesh/align_lmk_{args.data_item}.obj")
    print("head_mesh: ", head_mesh)
    head_path = os.path.abspath(head_mesh)
    init_obj(head_path)

    eyebrow_npy = os.path.join(npy_data_dir, f"{args.data_item}_fiber_final.npy")

    eyebrow_data = np.load(eyebrow_npy, allow_pickle=True)
    root = eyebrow_data[:, 0, :]

    obj = bpy.context.object

    psys_name = "hair_2"
    psys = obj.modifiers.new(psys_name, 'PARTICLE_SYSTEM').particle_system
    psys.settings.type = 'HAIR'
    psys.settings.emit_from = 'FACE'
    psys.settings.use_advanced_hair = True
    psys.settings.hair_step = eyebrow_data.shape[1] - 1
    psys.settings.count = eyebrow_data.shape[0]

    # 3. Set hair positions vertices
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ob = obj.evaluated_get(depsgraph)

    bpy.ops.particle.particle_edit_toggle()
    ob.particle_systems[psys_name].particles[0]. \
        hair_keys[0].co_object_set(ob, ob.modifiers[psys_name], ob.particle_systems[psys_name].particles[0],
                                  mathutils.Vector((0, 0, 0)))
    bpy.ops.particle.particle_edit_toggle()

    psys = ob.particle_systems[psys_name]

    for m in range(len(psys.particles)):
        root_xyz = root[m]
        psys.particles[m].location = (root_xyz[0], root_xyz[1], root_xyz[2])
        for n in range(len(psys.particles[m].hair_keys)):
            pt = eyebrow_data[m, n, :]
            psys.particles[m].hair_keys[n].co = (pt[0], pt[1], pt[2])

    # Toggle back and forth to update viewport
    bpy.ops.particle.particle_edit_toggle()
    bpy.ops.particle.particle_edit_toggle()

    bpy.ops.wm.save_mainfile(filepath=os.path.abspath(os.path.join(npy_data_dir, f"{args.data_item}.blend")))
    # bpy.ops.wm.alembic_export(filepath=os.path.abspath(os.path.join(item + "_eyebrow", f"{item}.abc")))