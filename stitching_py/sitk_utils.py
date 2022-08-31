import SimpleITK as sitk


def to_str_sitk(img: sitk.Image):
    return f"Size={img.GetSize()}, Spacing={img.GetSpacing()}, Origin={img.GetOrigin()}, Direction={img.GetDirection()}"


def resample_mask(mask: sitk.Image, ref_img: sitk.Image):
    return sitk.Resample(mask, ref_img, sitk.Transform(), sitk.NearestNeighbor, 0, mask.GetPixelID())


def resample_img(img: sitk.Image, ref_img: sitk.Image, verbose=True):
    if (
        img.GetSize() == ref_img.GetSize()
        and img.GetSpacing() == ref_img.GetSpacing()
        and img.GetOrigin() == ref_img.GetOrigin()
        and img.GetDirection() == ref_img.GetDirection()
    ):
        print("[*] Image needs no resampling") if verbose else None
        return img
    print(f"[*] Resample Image to {to_str_sitk(ref_img)}") if verbose else None
    # Resample(image1, referenceImage, transform, interpolator, defaultPixelValue, outputPixelType, useNearestNeighborExtrapolator)
    return sitk.Resample(img, ref_img, sitk.Transform(), sitk.sitkLinear, 0, img.GetPixelID())


def get_3D_corners(img: sitk.Image):
    shape = img.GetSize()
    out = []
    for x in [0, 1]:
        for y in [0, 1]:
            for z in [0, 1]:
                point = img.TransformIndexToPhysicalPoint((shape[0] * x, shape[1] * y, shape[2] * z))
                out.append(point)
    return out


# def resample_shared_space(img: sitk.Image, ref_img: sitk.Image, transform=sitk.Transform(), verbose=True):
#    #  Resample(image1,                 transform, interpolator, defaultPixelValue, outputPixelType ,useNearestNeighborExtrapolator,)
#    #  Resample(image1, referenceImage, transform, interpolator, defaultPixelValue, outputPixelType, useNearestNeighborExtrapolator)
#    # x Resample(image1, size,           transform, interpolator, outputOrigin, outputSpacing, outputDirection, defaultPixelValue , outputPixelType, useNearestNeighborExtrapolator)
#    extreme_points = get_3D_corners(img) + get_3D_corners(ref_img)
#    # print(extreme_points)
#    # Use the original spacing (arbitrary decision).
#    output_spacing = ref_img.GetSpacing()
#    # Identity cosine matrix (arbitrary decision).
#    output_direction = ref_img.GetDirection()
#    # Minimal x,y coordinates are the new origin.
#    output_origin = [min(extreme_points, key=lambda p: p[i])[i] for i in range(3)]
#    output_max = [max(extreme_points, key=lambda p: p[i])[i] for i in range(3)]
#    # Compute grid size based on the physical size and spacing.
#    output_size = [abs(int((output_max[i] - output_origin[i]) / output_spacing[i])) for i in range(3)]
#    print(extreme_points)
#    print(to_str_sitk(img), to_str_sitk(ref_img))
#    print(output_origin, output_size)
#    print(f"[*] Resample and Join Image") if verbose else None
#    a = sitk.Resample(img, output_size, transform, sitk.sitkLinear, output_origin, output_spacing, output_direction)
#    b = sitk.Resample(ref_img, output_size, transform, sitk.sitkLinear, output_origin, output_spacing, output_direction)
#    # sitk.Compose()
#    return sitk.Add(a, b)


def resize_image_itk(ori_img, target_img, resample_method=sitk.sitkLinear):
    """
    https://programmer.ink/think/resample-method-and-code-notes-of-python-simpleitk-library.html
    use itk Method to convert the original image resample To be consistent with the target image
    :param ori_img: Original alignment required itk image
    :param target_img: Target to align itk image
    :param resample_method: itk interpolation method : sitk.sitkLinear-linear  sitk.sitkNearestNeighbor-Nearest neighbor
    :return:img_res_itk: Resampling okay itk image
    """
    target_Size = target_img.GetSize()  # Target image size [x,y,z]
    target_Spacing = target_img.GetSpacing()  # Voxel block size of the target [x,y,z]
    target_origin = target_img.GetOrigin()  # Starting point of target [x,y,z]
    target_direction = target_img.GetDirection()  # Target direction [crown, sagittal, transverse] = [z,y,x]

    # The method of itk is resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # Target image to resample
    # Set the information of the target image
    resampler.SetSize(target_Size)  # Target image size
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # Set different type according to the need to resample the image
    if resample_method == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)  # Nearest neighbor interpolation is used for mask, and uint16 is saved
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # Linear interpolation is used for PET/CT/MRI and the like, and float32 is saved
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    itk_img_resampled = resampler.Execute(ori_img)  # Get the resampled image
    return itk_img_resampled


def pad_same(image, ref_img, default_value=-1):
    upper = []
    lower = []
    for index in range(3):
        image0_min_extent = image.GetOrigin()[index]
        image0_max_extent = image.GetOrigin()[index] + image.GetSize()[index] * image.GetSpacing()[index]
        min_extent = min(image0_min_extent, ref_img.GetOrigin()[index])
        max_extent = max(image0_max_extent, ref_img.GetOrigin()[index] + ref_img.GetSize()[index] * ref_img.GetSpacing()[index])
        lower.append(int((image0_min_extent - min_extent) / image.GetSpacing()[index] + 1))
        upper.append(int((max_extent - image0_max_extent) / image.GetSpacing()[index] + 1))

    filter = sitk.ConstantPadImageFilter()
    #  filter->SetInput(input);
    print(lower, upper)
    filter.SetPadLowerBound(lower)
    filter.SetPadUpperBound(upper)
    filter.SetConstant(default_value)
    return filter.Execute(image)


def padZ(image: sitk.Image, pad_min_z, pad_max_z, unique_value) -> sitk.Image:

    filter = sitk.ConstantPadImageFilter()
    #  filter->SetInput(input);
    filter.SetPadLowerBound([0, 0, pad_min_z])
    filter.SetPadUpperBound([0, 0, pad_max_z])
    filter.SetConstant(unique_value)
    return filter.Execute(image)


def cropZ(image: sitk.Image, pad_min_z, pad_max_z, verbose=True, z_index=2) -> sitk.Image:
    if verbose:
        print("[*] crop ", pad_min_z, abs(pad_max_z), "pixels")
    filter = sitk.CropImageFilter()
    filter.SetLowerBoundaryCropSize([abs(pad_min_z) if i == z_index else 0 for i in range(3)])
    filter.SetUpperBoundaryCropSize([abs(pad_max_z) if i == z_index else 0 for i in range(3)])
    return filter.Execute(image)


def divide_by_max(img: sitk.Image) -> sitk.Image:
    filter = sitk.MinimumMaximumImageFilter()
    filter.Execute(img)
    maximum = filter.GetMaximum()
    if maximum == 0:
        print("[!] Warning the max of this image is 0. It is probably empty ")
        return img
    return sitk.Divide(img, maximum)


def affine_registration_transform(fixed_image: sitk.Image, moving_image) -> sitk.Transform:
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.1,
        numberOfIterations=100,
        convergenceMinimumValue=1e-20,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    tran = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
    # Always check the reason optimization terminated.
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print("Optimizer's stopping condition, {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    return tran


def apply_transform(moving_image, fixed_image, transform, is_segmentation=False) -> sitk.Image:
    return sitk.Resample(
        moving_image,
        fixed_image,
        transform,
        sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )


def register_on_sub_image(fixed_sub_image, moving_sub_image, fixed_image, moving_image):
    transform = affine_registration_transform(fixed_sub_image, moving_sub_image)
    sitk.CompositeTransform()
    return apply_transform(moving_image, fixed_image, transform)
