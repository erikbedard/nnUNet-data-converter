#!/usr/bin/python3

import argparse
from visualize import visualize
import numpy as np
import os
import sys
import pyvista as pv

from vtkmodules.vtkCommonDataModel import vtkImageData
from tqdm import tqdm


def main():

    # EXAMPLE command-line inputs:
    # To compare an initial mask to a modified version of itself
    # (set opacities to see through the object):
    # python compare_masks.py PATH_TO_NIFTI_REFERENCE PATH_TO_NIFTI_COMPARISON --opacities 0.2 1 1

    # To compare a ground truth segmentation label to a predicted segmentation where the label number is 2:
    # python compare_masks.py PATH_TO_NIFTI_GROUND_TRUTH PATH_TO_NIFTI_PREDICTION --plot-type prediction --ref_label_num 2 --comp_label_num 2

    # Super command with all options:
    # To visualize a segmentation prediction (eith array name "NIFTI") to its ground truth label (with array name "NIFTI")
    # and to save the visualization as a video in the background:
    # python compare_masks.py PATH_TO_NIFTI_GROUND_TRUTH PATH_TO_NIFTI_PREDICTION --ref_array NIFTI --ref_label_num 1 --comp_array NIFTI --comp_label_num 1 --plot_type prediction --opacities 0.2 1 1 --video_path PATH_TO_MP4 --background

    parser = argparse.ArgumentParser()

    # parse reference image
    parser.add_argument(dest="ref_image", type=str,
                        help="File path of NIFTI image with reference label data.")
    parser.add_argument("--ref_array", type=str, default=None,
                        help="Name of array to be read from 'ref_image' for visualization."
                             "If not specified, the default active array is used.")
    parser.add_argument("--ref_label_num", type=int, default=1,
                        help="Integer value of the reference label to be visualized. "
                             "If not specified, a default value of 1 is used.")

    # parse comparison image
    parser.add_argument(dest="comp_image", type=str,
                        help="File path of NIFTI image with comparison label data.")
    parser.add_argument("--comp_array", type=str, default=None,
                        help="Name of array to be read from 'comp_image' for visualization."
                             "If not specified, the default active array is used.")
    parser.add_argument("--comp_label_num", type=int, default=1,
                        help="Integer value of the comparison label to be visualized. "
                             "If not specified, a default value of 1 is used.")

    # parse visualization options
    parser.add_argument("--plot_type", type=str, choices=["generic", "prediction"], default="generic",
                        help="Select how to visualize the plot based on the inputs.")

    class float_between_zero_and_one:
        def __eq__(self, value):
            return 0.0 <= value <= 1.0

    parser.add_argument("--opacities", type=float, default=1, choices=[float_between_zero_and_one()], nargs=3,
                        help="Set opacity of intersecting, added, and removed voxels of the comparison image "
                             "relative to the reference image.")
    # optional video
    parser.add_argument("--video_path", type=str, default=None,
                        help="Optionally specify path (with '.mp4' extension) to save a 360-degree video of the visualization.")
    parser.add_argument("--background", default=False, action='store_true', required='--save_video_path' in sys.argv,
                        help="Save video in background only and do not show the plot.")

    args = parser.parse_args()

    # get reference mask
    reference = visualize.read_nifti(args.ref_image)
    if args.ref_array is not None:
        pv_ref = pv.wrap(reference)
        pv_ref.set_active_scalars(args.ref_array)

    reference = visualize.get_mask_from_labels(reference, args.ref_label_num)

    # get comparison mask
    comparison = visualize.read_nifti(args.comp_image)
    if args.ref_array is not None:
        pv_comp = pv.wrap(reference)
        pv_comp.set_active_scalars(args.comp_array)

    comparison = visualize.get_mask_from_labels(comparison, args.comp_label_num)

    # validate
    base = os.path.basename(args.video_path)
    ext = base.split('.')[-1]
    if ext != 'mp4':
        raise argparse.ArgumentError(message="Video path must have '.mp4' extension.")
    show_plot = not args.background

    compare_masks(reference, comparison,
                  comparison_type=args.plot_type,
                  opacities=args.opacities,
                  video_path=args.video_path,
                  show_plot=show_plot)


def compare_masks(reference_mask: vtkImageData, comparison_mask: vtkImageData, comparison_type="generic",
                           opacities=(1, 1, 1), video_path=None, show_plot=True):

    def plot_meshes(*meshes, opacities=(1, 1, 1), legend_names=("Intersection", "Added", "Removed")):
        pv.set_plot_theme('document')
        colors = ['grey', 'orange', 'blue']

        def make_plotter(off_screen=False):
            plotter = pv.Plotter(off_screen=off_screen)
            plotter.hide_axes()

            for i, mesh in enumerate(meshes):
                if mesh.number_of_points > 0:
                    plotter.add_mesh(mesh, label=legend_names[i], opacity=opacities[i], color=colors[i])
            plotter.meshes = meshes
            plotter.add_legend(bcolor=(0.95, 0.95, 0.95))
            return plotter

        if show_plot:
            plotter = make_plotter()
            plotter.show()

        if video_path is not None:
            plotter = make_plotter(off_screen=True)
            render_video(plotter, video_path)

    def extract_foreground(mesh: pv.UniformGrid, array_name: str, foreground: float = 1):
        arr = mesh.cell_data[array_name]
        cell_ids = np.argwhere(arr == foreground)
        return mesh.extract_cells(cell_ids)

    def create_meshes(reference_mask, comparison_mask, data_labels):
        reference = pv.wrap(reference_mask)
        comparison = pv.wrap(comparison_mask)
        reference = visualize.image_points_to_voxels(reference)
        comparison = visualize.image_points_to_voxels(comparison)

        reference_array = reference.active_scalars.astype('uint8')
        comparison_array = comparison.active_scalars

        # create meshes to visualize
        intersection = reference_array * comparison_array
        inter_mesh = pv.UniformGrid()
        inter_mesh.deep_copy(reference)
        inter_mesh.cell_data[data_labels[0]] = intersection

        comparison_not_reference = comparison_array * np.where(reference_array == 0, 1, 0)
        comparison_not_reference_mesh = pv.UniformGrid()
        comparison_not_reference_mesh.deep_copy(reference)
        comparison_not_reference_mesh.cell_data[data_labels[1]] = comparison_not_reference

        reference_not_comparison = reference_array * np.where(comparison_array == 0, 1, 0)
        reference_not_comparison_mesh = pv.UniformGrid()
        reference_not_comparison_mesh.deep_copy(reference)
        reference_not_comparison_mesh.cell_data[data_labels[2]] = reference_not_comparison

        inter_mesh = extract_foreground(inter_mesh, array_name=data_labels[0])
        comparison_not_reference_mesh = extract_foreground(comparison_not_reference_mesh, array_name=data_labels[1])
        reference_not_comparison_mesh = extract_foreground(reference_not_comparison_mesh, array_name=data_labels[2])

        return inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh

    if comparison_type == "generic":
        data_labels = ("Intersection", "Added", "Removed")
    elif comparison_type == "prediction":
        data_labels = ("Correct Prediction", "False Positive", "False Negative")
    else:
        raise RuntimeError

    inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh \
        = create_meshes(reference_mask, comparison_mask, data_labels)

    plot_meshes(inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh,
                opacities=opacities,
                legend_names=data_labels)


def render_video(plotter, file_name: str,
                 frame_rate: int = 30,
                 aspect_ratio=np.array([16, 9]),
                 ppi=160,  # leave this as a multiple of 16 or codec fails
                 revolution=360,  # degrees in a full revolution
                 n_frames=360,  # frames per revolution
                 fade_in=0,  # number of frames to hold still for before rotating
                 fade_out=0,  # number of frames to hold still for after rotating
                 ):

    resolution = aspect_ratio * ppi

    plotter.open_movie(file_name, framerate=frame_rate)
    plotter.show(auto_close=False, window_size=resolution)

    for _ in range(fade_in):
        plotter.write_frame()

    combined_mesh = plotter.meshes[0]
    for i in range(1, len(plotter.meshes)):
        combined_mesh.merge(plotter.meshes[i])

    angle = revolution / n_frames

    mesh_center = visualize.get_mesh_center(combined_mesh)
    plotter.write_frame()
    for _ in tqdm(range(n_frames), desc="rendering '" + file_name + "'"):
        for mesh in plotter.meshes:
            mesh.rotate_z(angle, point=mesh_center, inplace=True)
        plotter.write_frame()

    for _ in range(fade_out):
        plotter.write_frame()

    plotter.close()


if __name__ == "__main__":
    main()