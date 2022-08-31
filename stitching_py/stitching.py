# stitching - command line tool for whole-body image stitching.#
# Copyright 2016 Ben Glocker <b.glocker@imperial.ac.uk> and 2022 Robert Graf (Python translation)#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at#
#    http://www.apache.org/licenses/LICENSE-2.0#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import SimpleITK as sitk
import numpy as np
from sitk_utils import padZ, resample_img, to_str_sitk
import time


def main(files: list[str], filename_out: str = "", margin: int = 0, average_overlap: bool = False, verbose=True):
    stime = time.time()
    if len(files) > 1:
        print("stitching image...")
        unique_value: float = -1234567
        loaded = sitk.Cast(sitk.ReadImage(files[0]), sitk.sitkFloat32)  # copy

        if margin != 0:
            image0: sitk.Image = loaded[:, :, margin:-margin]
        else:
            image0: sitk.Image = loaded
        # determine physical extent of stitched volume
        image0_min_extent = image0.GetOrigin()[2]
        image0_max_extent = image0.GetOrigin()[2] + image0.GetSize()[2] * image0.GetSpacing()[2]
        min_extent = image0_min_extent
        max_extent = image0_max_extent
        images: list[sitk.Image] = []
        for i, f in enumerate(files):
            if i == 0:
                continue
            temp = sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32)
            if margin != 0:
                image: sitk.Image = temp[:, :, margin:-margin]
            else:
                image: sitk.Image = temp
            images.append(image)
            min_extent = min(min_extent, image.GetOrigin()[2])
            max_extent = max(max_extent, image.GetOrigin()[2] + image.GetSize()[2] * image.GetSpacing()[2])
        if verbose:
            print("[*] take x,y size from", files[0])

        # generate stitched volume and fill in first image
        pad_min_z = int((image0_min_extent - min_extent) / image0.GetSpacing()[2] + 1)
        pad_max_z = int((max_extent - image0_max_extent) / image0.GetSpacing()[2] + 1)

        target = padZ(image0, pad_min_z, pad_max_z, unique_value)

        def get_threshold_as_np(input):
            # find valid image values
            arr = sitk.GetArrayFromImage(input)
            arr[arr < unique_value + 100] = unique_value
            arr[arr > unique_value + 100] = 1
            arr[arr < unique_value + 100] = 0
            return arr

        def np_to_skit(arr, ref):
            sitk_img: sitk.Image = sitk.GetImageFromArray(arr)
            sitk_img.SetOrigin(ref.GetOrigin())
            sitk_img.SetSpacing(ref.GetSpacing())
            return sitk_img

        target_arr = sitk.GetArrayFromImage(target)
        counts_arr = get_threshold_as_np(target)
        target_arr *= counts_arr
        target_arr = target_arr / target_arr.max()

        # iterate over remaining images and add to stitched volume

        for cur_file, cur_img in zip(files[1:], images):

            empty_arr = 1 - counts_arr

            cur_img_min_extent = cur_img.GetOrigin()[2]
            cur_img_max_extent = cur_img.GetOrigin()[2] + cur_img.GetSize()[2] * cur_img.GetSpacing()[2]
            if verbose:
                print("[*] stich next file", cur_file)
            pad_min_z = int((cur_img_min_extent - min_extent) / cur_img.GetSpacing()[2] + 1)
            pad_max_z = int((max_extent - cur_img_max_extent) / cur_img.GetSpacing()[2] + 1)
            cur_img = padZ(cur_img, pad_min_z, pad_max_z, unique_value)
            # Resample to same space
            cur_img = resample_img(cur_img, target, verbose)
            # print(min)
            cur_arr = sitk.GetArrayFromImage(cur_img).copy()

            binary_arr = get_threshold_as_np(cur_img)
            # take only value for empty voxels, otherwise average values in overlap areas
            if not average_overlap:
                binary_arr = empty_arr * binary_arr

            cur_arr = cur_arr * binary_arr
            target_arr = cur_arr / cur_arr.max() + target_arr  #
            counts_arr = binary_arr + counts_arr
        counts_arr[counts_arr == 0] = 1
        target_arr /= counts_arr
        out_skit = np_to_skit(target_arr, target)

        off_z_min = 0

        while target_arr[:, :, off_z_min].sum() == 0:
            off_z_min += 1

        off_z_max = -1
        while target_arr[:, :, off_z_max].sum() == 0:
            off_z_max -= 1
        out_skit: sitk.Image = out_skit[:, :, off_z_min:off_z_max]  #
        if verbose:
            print(to_str_sitk(out_skit))
            print("[#] Write Image ", filename_out)
        sitk.WriteImage(out_skit, filename_out)
        print("done. took ", time.time() - stime)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-i", "--images", nargs="+", default=[], help="filenames of images")
    parser.add_argument("-o", "--output", type=str, default="out.nii.gz", help="filename of output image")
    parser.add_argument("-m", "--margin", type=int, default=0, help="image margin that is ignored when stitching")
    parser.add_argument("-a", "--averaging", default=False, action=argparse.BooleanOptionalAction, help="enable averaging in overlap areas")
    parser.add_argument("-v", "--verbose", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.verbose:
        print(args)
    main(args.images, args.output, args.margin, args.averaging, args.verbose)
