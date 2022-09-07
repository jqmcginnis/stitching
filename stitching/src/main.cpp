/*
* stitching - command line tool for whole-body image stitching.
*
* Copyright 2016 Ben Glocker <b.glocker@imperial.ac.uk>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "itkio.h"
#include "miaImage.h"
#include "miaImageProcessing.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace mia;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

template <typename T>
void strings_to_values(const std::vector<std::string>& string_seq, std::vector<T>& values)
{
  for(std::vector<std::string>::const_iterator it = string_seq.begin(); it != string_seq.end(); ++it)
  {
    std::stringstream ss(*it);
    std::copy(std::istream_iterator<T>(ss), std::istream_iterator<T>(), back_inserter(values));
  }
}

int main(int argc, char* argv[])
{
  std::vector<std::string> sfiles;
  std::string filename_out;
  int margin = 0;
  bool average_overlap = false;

  try
  {
    // Declare the supported options.
    po::options_description options("options");
    options.add_options()
    ("help,h", "produce help message")
    ("images,i", po::value<std::vector<std::string>>(&sfiles)->multitoken(), "filenames of images")
    ("output,o", po::value<std::string>(&filename_out), "filename of output image")
    ("margin,m", po::value<int>(&margin), "image margin that is ignored when stitching")
    ("averaging,a", po::bool_switch(&average_overlap)->default_value(false), "enable averaging in overlap areas")
    ;

    po::variables_map vm;

    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    if (vm.count("help") || vm.size() == 0)
    {
      std::cout << options << std::endl;
      return 0;
    }
  }
  catch(std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return 1;
  }

  // convert string-based parameters to values
  std::vector<std::string> files;
  strings_to_values(sfiles, files);

  if (files.size() > 1)
  {
    namespace ch = std::chrono;
    auto start = ch::high_resolution_clock::now();
    std::cout << "stitching image...";

    float unique_value = -123456789.0;
    auto loaded = itkio::load(files[0]);

    // determine physical extent of stitched volume
    auto image0 = subimage(loaded, 0, 0, margin, loaded.sizeX(), loaded.sizeY(), loaded.sizeZ()-2*margin).clone();
    auto image0_min_extent = image0.origin()[2];
    auto image0_max_extent = image0.origin()[2] + image0.sizeZ() * image0.spacing()[2];
    auto min_extent = image0_min_extent;
    auto max_extent = image0_max_extent;
    std::vector<Image> images;
    for (int i = 1; i < files.size(); i++)
    {
      auto temp = itkio::load(files[i]);
      auto image = subimage(temp, 0, 0, margin, temp.sizeX(), temp.sizeY(), temp.sizeZ()-2*margin).clone();
      images.push_back(image);
      if (image.origin()[2] < min_extent)
      {
        min_extent = image.origin()[2];
      }
      if (image.origin()[2] + image.sizeZ() * image.spacing()[2] > max_extent)
      {
        max_extent = image.origin()[2] + image.sizeZ() * image.spacing()[2];
      }
    }
    std::cout << std::endl;

    std::cout << "Min:" << min_extent << std::endl;
    std::cout << "Max:" << max_extent << std::endl;

    // generate stitched volume and fill in first image
    auto pad_min_z = static_cast<int>((image0_min_extent - min_extent) / image0.spacing()[2] + 1);
    auto pad_max_z = static_cast<int>((max_extent - image0_max_extent) / image0.spacing()[2] + 1);
    auto target = pad(image0, 0, 0, 0, 0, pad_min_z, pad_max_z, unique_value); // stitched extent

    target.dataType(mia::FLOAT);
    itkio::save(target, "target_step1.nii.gz");

    std::cout<< target.origin() <<std::endl;
    std::cout<< target.sizeX() <<std::endl;
    std::cout<< target.sizeY() <<std::endl;
    std::cout<< target.sizeZ() <<std::endl;

    auto counts = target.clone(); // has the same dimensions as stitched images

    //output first intermediate result
    target.dataType(mia::FLOAT);
    itkio::save(target, "c++_intermediate_result1.nii.gz"); // these are the same

    //find valid image values
    // set all elements of counts to 0, except 
    threshold(target, counts, unique_value, unique_value); // counts is the output
    // set all elements of counts to 1

    invert_binary(counts, counts);
    itkio::save(counts, "counts.nii.gz");
    mul(target, counts, target); // output is in target

    itkio::save(target, "target_step2.nii.gz");

    //output first intermediate result
    target.dataType(mia::FLOAT);
    itkio::save(target, "c++_intermediate_result2.nii.gz"); // these are already the same

    // iterate over remaining images and add to stitched volume
    auto binary = target.clone(); // clone to have the target dimensions
    auto empty = target.clone();
    for (int i = 0; i < images.size(); i++)
    {
      threshold(counts, empty, 0, 0); // empty is output
      itkio::save(target, "empty0.nii.gz");
      auto trg = target.clone();
      std::cout<< "Trg Z Size" << trg.sizeZ() <<std::endl;

      // this is misleading - we do resample here, but we do not average here!
      // resample(images[i], trg, mia::LINEAR, unique_value); // trg is output
      resample(images[i], trg, mia::LINEAR, 0); //unique_value); // trg is output

      std::stringstream ss;
      ss << "c++_trg_" << i << ".nii.gz"; 
      std::string s = ss.str();
      target.dataType(mia::FLOAT);
      itkio::save(trg, s);

      std::cout<< "Trg Z Size" << trg.sizeZ() <<std::endl;
      threshold(trg, binary, unique_value, unique_value); // binary is output
      std::cout<< "Trg Z Size" << trg.sizeZ() <<std::endl;
      invert_binary(binary, binary); // binary is output

      //take only value for empty voxels, otherwise average values in overlap areas
      if (!average_overlap) mul(empty, binary, binary); // output is saved in binary, is executed if average overlap is not activated

      mul(trg, binary, trg);

      if (i==0){
        itkio::save(trg, "mul_trg0.nii.gz");
      }

      add(trg, target, target);
      if (i==0){
        itkio::save(target, "add_target0.nii.gz");
      }

      add(binary, counts, counts);
      if (i==0){
        itkio::save(counts, "cnts_trg0.nii.gz");
      }

      /*
      std::stringstream ss;
      ss << "c++_trg_" << i << ".nii.gz"; 
      std::string s = ss.str();
      target.dataType(mia::FLOAT);
      itkio::save(trg, s);
      */
    }

    target.dataType(mia::FLOAT);
    itkio::save(target, "c++_intermediate_result3.nii.gz");

    std::cout << "Counts X" << counts.sizeX()<< std::endl;
    std::cout << "Counts Y" << counts.sizeY()<< std::endl;
    std::cout << "Counts Z" << counts.sizeZ()<< std::endl;

    //remove extra empty slices introduced to rounding of pad values
    int central_x = counts.sizeX() / 2;
    int central_y = counts.sizeY() / 2;

    std::cout << "central x and y" << std::endl;
    std::cout<<central_x<<std::endl;
    std::cout<<central_y<<std::endl;

    std::cout << "Counts X" << counts.sizeX()<< std::endl;
    std::cout << "Counts Y" << counts.sizeY()<< std::endl;
    std::cout << "Counts Z" << counts.sizeZ()<< std::endl;

    // crop z axis and delete empty slices (from top till bottom and vice versa)
    int off_z_min = 0;
    while (counts(central_x, central_y, off_z_min) == 0 && off_z_min < counts.sizeZ() - 1)
    {
      off_z_min++;
    }
    int off_z_max = counts.sizeZ();
    std::cout << "init z max" << off_z_max << std::endl;
    std::cout << off_z_max <<std::endl;
    while (counts(central_x, central_y, off_z_max - 1) == 0 && off_z_max > 0)
    {
      off_z_max--;
    }

    std::cout << off_z_min << std::endl;
    std::cout << off_z_max << std::endl;

    threshold(counts, binary, 0, 0);
    add(binary, counts, counts);
    div(target, counts, target);

    target = subimage(target, 0, 0, off_z_min, target.sizeX(), target.sizeY(), off_z_max - off_z_min).clone();

    std::cout << "Target X" << target.sizeX()<< std::endl;
    std::cout << "Target Y" << target.sizeY()<< std::endl;
    std::cout << "Target Z" << target.sizeZ()<< std::endl;

    target.dataType(mia::FLOAT);
    itkio::save(target, filename_out);

    auto stop = ch::high_resolution_clock::now();
    std::cout << "done. took " << ch::duration_cast< ch::milliseconds >(stop-start).count() << " ms" << std::endl;
  }
}
