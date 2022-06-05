import numpy as np
import pyvista as pv

from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkIOImage import vtkNIFTIImageReader


def visualize_numpy_mask(mask: np.ndarray, spacing=(1, 1, 1), origin=(0, 0, 0)):
    assert(np.all(np.logical_or(mask == 0, mask == 1)))
    grid = pv.UniformGrid()
    grid.dimensions = np.array(mask.shape) + 1
    grid.spacing = spacing
    grid.origin = origin
    grid.cell_data["values"] = mask.flatten(order="F")

    arr = grid.cell_data["values"]
    cell_ids = np.argwhere(arr == 1)
    grid = grid.extract_cells(cell_ids)
    grid.plot()


def visualize_two_masks(reference_mask_path, reference_mask_label_num,
                        comparison_mask_path, comparison_mask_label_num,
                        comparison_type="generic",
                        screenshot_path=None,
                        opacities=(1, 1, 1)):

    reference = read_nifti(reference_mask_path)
    reference_mask = get_mask_from_labels(reference, reference_mask_label_num)

    comparison = read_nifti(comparison_mask_path)
    comparison_mask = get_mask_from_labels(comparison, comparison_mask_label_num)

    _visualize_two_masks(reference_mask, comparison_mask,
                         comparison_type=comparison_type,
                         screenshot_path=screenshot_path,
                         opacities=opacities)


def _visualize_two_masks(reference_mask: vtkImageData, comparison_mask: vtkImageData,
                         comparison_type="generic",
                         screenshot_path=None,
                         opacities=(1, 1, 1)):

    def plot_meshes(*meshes,
                    opacities=(1, 1, 1),
                    legend_names=("Intersection", "Added", "Removed"),
                    screenshot_path=None):

        pv.set_plot_theme('document')
        plotter = pv.Plotter()
        colors = ['grey', 'blue', 'red']

        for i, mesh in enumerate(meshes):
            if mesh.number_of_points > 0:
                plotter.add_mesh(mesh, label=legend_names[i], opacity=opacities[i], color=colors[i])
            plotter.add_legend(bcolor=(0.95, 0.95, 0.95))

        if screenshot_path is None:
            plotter.show()
        else:
            plotter.off_screen = True
            plotter.show(screenshot=screenshot_path, window_size=(1600, 1600))

    def extract_foreground(mesh: pv.UniformGrid, array_name: str, foreground: float = 1):
        arr = mesh.cell_data[array_name]
        cell_ids = np.argwhere(arr == foreground)
        return mesh.extract_cells(cell_ids)

    def image_points_to_voxels(image: pv.UniformGrid):
        origin = image.origin
        spacing = np.array(image.spacing)
        dims = np.array(image.dimensions) + 1

        m_dims = (3, 3)
        d_matrix = image.GetDirectionMatrix()
        nd_matrix = np.empty(m_dims)
        for i in range(m_dims[0]):
            for j in range(m_dims[1]):
                nd_matrix[i, j] = d_matrix.GetElement(i, j)

        origin = origin - np.matmul(nd_matrix, spacing) / 2

        voxel_img = pv.UniformGrid(dims=dims, spacing=spacing, origin=origin)

        num_arrays = image.GetPointData().GetNumberOfArrays()
        for i in range(num_arrays):
            arr = image.GetPointData().GetAbstractArray(i)
            voxel_img.GetCellData().AddArray(arr)

        return voxel_img

    def create_meshes(reference_mask, comparison_mask, data_labels):
        reference = pv.wrap(reference_mask)
        comparison = pv.wrap(comparison_mask)
        reference = image_points_to_voxels(reference)
        comparison = image_points_to_voxels(comparison)

        reference.cell_data["NIFTI"] = reference.cell_data["NIFTI"].astype('uint8')
        comparison.cell_data["NIFTI"] = comparison.cell_data["NIFTI"].astype('uint8')

        reference_array = reference.cell_data["NIFTI"]
        comparison_array = comparison.cell_data["NIFTI"]

        # create meshes to visualize
        intersection = reference_array * comparison_array
        inter_mesh = pv.UniformGrid()
        inter_mesh.deep_copy(reference)
        del inter_mesh.cell_data['NIFTI']
        inter_mesh.cell_data[data_labels[0]] = intersection

        comparison_not_reference = comparison_array * (reference_array == 0).astype('uint8')
        comparison_not_reference_mesh = pv.UniformGrid()
        comparison_not_reference_mesh.deep_copy(reference)
        del comparison_not_reference_mesh.cell_data['NIFTI']
        comparison_not_reference_mesh.cell_data[data_labels[1]] = comparison_not_reference

        reference_not_comparison = reference_array * (comparison_array == 0).astype('uint8')
        reference_not_comparison_mesh = pv.UniformGrid()
        reference_not_comparison_mesh.deep_copy(reference)
        del reference_not_comparison_mesh.cell_data['NIFTI']
        reference_not_comparison_mesh.cell_data[data_labels[2]] = reference_not_comparison

        inter_mesh = extract_foreground(inter_mesh, array_name=data_labels[0])
        comparison_not_reference_mesh = extract_foreground(comparison_not_reference_mesh, array_name=data_labels[1])
        reference_not_comparison_mesh = extract_foreground(reference_not_comparison_mesh, array_name=data_labels[2])

        return inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh

    if comparison_type == "generic":
        data_labels = ("Intersection", "Added", "Removed")
    elif comparison_type == "truth_prediction":
        data_labels = ("Correct Prediction", "False Positive", "False Negative")
    else:
        return RuntimeError

    inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh \
        = create_meshes(reference_mask, comparison_mask, data_labels)

    plot_meshes(inter_mesh, comparison_not_reference_mesh, reference_not_comparison_mesh,
                opacities=opacities, legend_names=data_labels, screenshot_path=screenshot_path)


def get_mask_from_labels(image: vtkImageData, label_num: int):
    """
    Extracts a binary mask from an image with labels.
    Args:
        image: image with integer label data (e.g. 0, 1, 2...)
        label_num: number of the label to be extracted
    Returns:
        vtkImageData object with binary mask of selected label
    """
    threshold = vtkImageThreshold()
    threshold.SetInputData(image)
    threshold.ThresholdBetween(label_num, label_num)
    threshold.SetInValue(1)
    threshold.SetOutValue(0)
    threshold.Update()
    return threshold.GetOutput()


def read_nifti(filepath):
    """
    Read a nifti image (.nii or .nii.gz).
    Args:
        filepath: path of NIFTI image to be read
    Returns:
        vtkImageData object
    """
    reader = vtkNIFTIImageReader()
    reader.SetFileName(filepath)
    reader.Update()
    vtk_image = reader.GetOutput()
    return vtk_image
