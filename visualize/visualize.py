import numpy as np
import pyvista as pv

from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.util.numpy_support import numpy_to_vtk


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

    voxel_img.set_active_scalars(image.active_scalars_name)

    return voxel_img


def get_mesh_center(mesh: pv.UnstructuredGrid):
    bnds = np.array(mesh.bounds)  # (xmin, xmax, ymin, ymax, zmin, zmax)
    bnd_lengths = np.array([bnds[1] - bnds[0], bnds[3] - bnds[2], bnds[5] - bnds[4]])
    mesh_center = bnds[[0, 2, 4]] + bnd_lengths / 2
    return mesh_center


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


def apply_mask_to_image(mask: vtkImageData, image: vtkImageData):
    name = image.GetPointData().GetScalars().GetName()
    image_scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    mask_scalars = vtk_to_numpy(mask.GetPointData().GetScalars())
    assert(mask.GetScalarRange() == (0,1))
    applied = image_scalars * mask_scalars

    vtk_scalars = numpy_to_vtk(applied)
    vtk_scalars.SetName(name)
    image.GetPointData().SetScalars(vtk_scalars)
    return image


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


def exec_cli(method, args: list):
    # execute a method using CLI arguments

    import sys
    original_args = sys.argv

    args.insert(0, '')
    sys.argv = args
    method()

    # subprocess.run([sys.executable, '-m', file, *args])

    sys.argv = original_args
