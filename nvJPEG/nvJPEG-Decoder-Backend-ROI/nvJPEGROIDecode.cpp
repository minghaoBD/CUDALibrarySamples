/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
  
#include "nvJPEGROIDecode.h"
#include "threadpool.h"

int parseDecodeCoordinates(const char* argv, decode_params_t& params)
{
    std::istringstream decode_area(argv);
    std::string temp;
    int idx = 0;
    while(getline(decode_area, temp,','))
    {
        if(idx == 0)
        {
            params.offset_x = std::stoi(temp);
        }
        else if (idx == 1)
        {
            params.offset_y = std::stoi(temp);
        }
        else if( idx == 2)
        {
            params.roi_width = std::stoi(temp);
        }
        else if (idx == 3)
        {
            params.roi_height = std::stoi(temp);
        }
        else
        {
            std::cout<<"Invalid ROI"<<std::endl;
            return EXIT_FAILURE;
        }
        idx++;
    }
    if (params.offset_x < 0 || params.offset_y < 0)
    {
        std::cout<<"Invalid ROI"<<std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

float get_scale_factor(nvjpegChromaSubsampling_t chroma_subsampling)
{
  float scale_factor = 3.0; // it should be 3.0 for 444
  if(chroma_subsampling == NVJPEG_CSS_420 || chroma_subsampling == NVJPEG_CSS_411) {
    scale_factor = 1.5;
  }
  else if(chroma_subsampling == NVJPEG_CSS_422 || chroma_subsampling == NVJPEG_CSS_440) {
    scale_factor = 2.0;
  }
  else if(chroma_subsampling == NVJPEG_CSS_410) {
    scale_factor = 1.25;
  }
  else if(chroma_subsampling = NVJPEG_CSS_GRAY){
    scale_factor = 1.0;
  }

  return scale_factor;
}

bool pick_gpu_backend(nvjpegJpegStream_t&  jpeg_stream)
{
  unsigned int frame_width,frame_height;
  nvjpegChromaSubsampling_t chroma_subsampling;
      
  CHECK_NVJPEG(nvjpegJpegStreamGetFrameDimensions(jpeg_stream,
        &frame_width, &frame_height));
  CHECK_NVJPEG(nvjpegJpegStreamGetChromaSubsampling(jpeg_stream,&chroma_subsampling));
  auto scale_factor = get_scale_factor(chroma_subsampling);

  bool use_gpu_backend = false;
  // use NVJPEG_BACKEND_GPU_HYBRID when dimensions are greater than 512x512
  if( frame_height*frame_width * scale_factor > 512*512 * 3)
  {
    use_gpu_backend = true;
  }
  return use_gpu_backend;
}

bool check_roi(const FileData &img_data, const std::vector<size_t> &img_len, 
               decode_params_t &params, int index, 
               std::vector<int> &widths, std::vector<int> &heights, 
               std::vector<bool> &valid_images){

  int img_widths[NVJPEG_MAX_COMPONENT];
  int img_heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;

  CHECK_NVJPEG(nvjpegGetImageInfo(
    params.nvjpeg_handle, (unsigned char *)img_data[index].data(), img_len[index],
    &channels, &subsampling, img_widths, img_heights))
  int img_width = img_widths[0];
  int img_height = img_heights[0];

  bool valid_roi = true;
  if (img_width < params.offset_x + params.roi_width || img_height < params.offset_y + params.roi_height){
      valid_roi = false;
      valid_images[index] = false;
      widths[index] = img_width;
      heights[index] = img_height;
  }
  else{
      widths[index] = params.roi_width;
      heights[index] = params.roi_height;
  }

  return valid_roi;
}

void select_backend(nvjpegJpegDecoder_t& decoder, nvjpegJpegState_t& decoder_state, 
                    decode_per_thread_params& per_thread_params, decode_params_t &params,
                    int &buffer_index){
  bool use_gpu_backend =  pick_gpu_backend(per_thread_params.jpeg_streams[buffer_index]);
  decoder = use_gpu_backend ?  per_thread_params.nvjpeg_dec_gpu: per_thread_params.nvjpeg_dec_cpu;
  decoder_state = use_gpu_backend ? per_thread_params.dec_state_gpu:per_thread_params.dec_state_cpu;

  switch(params.backend_enum){
    case 0:
      break;
    case 1:
      decoder =   per_thread_params.nvjpeg_dec_cpu;
      decoder_state = per_thread_params.dec_state_cpu;
      break;
    case 2:
      decoder =   per_thread_params.nvjpeg_dec_gpu;
      decoder_state = per_thread_params.dec_state_gpu;
      break;
  }
  return;
}

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<nvjpegImage_t> &out, decode_params_t &params, ThreadPool &workers,
                  double &time, std::vector<int> &widths,
                  std::vector<int> &heights) {
  cudaEvent_t startEvent = NULL, stopEvent = NULL;
  float loopTime = 0; 
  std::vector<bool> valid_images(params.batch_size, true);
  
  CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
  CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));

  CHECK_CUDA(cudaEventRecord(startEvent, params.global_stream));
  
  
  if(params.num_threads < 2)
  {
    std::vector<nvjpegDecodeParams_t> decode_params(params.batch_size);
    auto& per_thread_params = params.nvjpeg_per_thread_data[0];
    int buffer_index = 0;
    

    for (int i = 0; i < params.batch_size; i++) {
        CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &decode_params[i])); 
        if (params.roi_on) {
          if (!check_roi(img_data, img_len, params, i, widths, heights, valid_images)) {
            std::cout << "ROI exceeds the boundaries of an image. This image will be fully decoded." << std::endl;
          }
          else{
            CHECK_NVJPEG(nvjpegDecodeParamsSetROI(decode_params[i], params.offset_x, params.offset_y, params.roi_width, params.roi_height))
          }
        }

        CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decode_params[i], params.fmt));

        CHECK_NVJPEG(
            nvjpegJpegStreamParse(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i],
            0, 0, per_thread_params.jpeg_streams[buffer_index]));

        nvjpegJpegDecoder_t decoder;
        nvjpegJpegState_t decoder_state;
        select_backend(decoder, decoder_state, per_thread_params, params, buffer_index);

        CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoder_state,
            per_thread_params.pinned_buffers[buffer_index]));

        CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, decoder, decoder_state,
            decode_params[i], per_thread_params.jpeg_streams[buffer_index]));

        CHECK_CUDA(cudaEventSynchronize(per_thread_params.decode_events[buffer_index]));

        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, decoder, decoder_state,
            per_thread_params.jpeg_streams[buffer_index], per_thread_params.stream));

        CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, decoder, decoder_state,
            &out[i], per_thread_params.stream));

        CHECK_CUDA(cudaEventRecord(per_thread_params.decode_events[buffer_index],  per_thread_params.stream))

        CHECK_NVJPEG(nvjpegDecodeParamsDestroy(decode_params[i]));

        buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

    }
    for(int i = 0; i < pipeline_stages; i++ ) {
      CHECK_CUDA(cudaEventSynchronize(per_thread_params.decode_events[i]));
    }
  }
  else
  {
    std::vector<int> buffer_indices(params.num_threads, 0);
    
    for (int i = 0; i < params.batch_size; i++) {
        if (params.roi_on){
          check_roi(img_data, img_len, params, i, widths, heights, valid_images);
        }
        
        workers.enqueue(std::bind(
            [&params, &buffer_indices, &out, &img_data, &img_len, &valid_images](int iidx, int thread_idx)
                {
                  nvjpegDecodeParams_t decode_params;
                  CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &decode_params)); 

                  auto& per_thread_params = params.nvjpeg_per_thread_data[thread_idx];

                  if (params.roi_on)
                  {
                    if (valid_images[iidx]){
                      CHECK_NVJPEG(nvjpegDecodeParamsSetROI(decode_params, params.offset_x, params.offset_y, params.roi_width, params.roi_height))
                    }
                    else{
                      std::cout << "ROI exceeds the boundaries of an image. This image will be fully decoded." << std::endl;
                    }
                  }

                  CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decode_params, params.fmt));

                  CHECK_NVJPEG(nvjpegJpegStreamParse(params.nvjpeg_handle, (const unsigned char *)img_data[iidx].data(), img_len[iidx],
                    0, 0, per_thread_params.jpeg_streams[buffer_indices[thread_idx]]));
      
                  nvjpegJpegDecoder_t decoder;
                  nvjpegJpegState_t decoder_state;
                  select_backend(decoder, decoder_state, per_thread_params, params, buffer_indices[thread_idx]);

                  CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoder_state,
                    per_thread_params.pinned_buffers[buffer_indices[thread_idx]]));

                  CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, decoder, decoder_state,
                    decode_params, per_thread_params.jpeg_streams[buffer_indices[thread_idx]]));

                  CHECK_CUDA(cudaEventSynchronize(per_thread_params.decode_events[buffer_indices[thread_idx]]));

                  CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, decoder, decoder_state,
                    per_thread_params.jpeg_streams[buffer_indices[thread_idx]], per_thread_params.stream));

                  CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, decoder, decoder_state,
                    &out[iidx], per_thread_params.stream));

                  CHECK_CUDA(cudaEventRecord(per_thread_params.decode_events[buffer_indices[thread_idx]],  per_thread_params.stream))

                  CHECK_NVJPEG(nvjpegDecodeParamsDestroy(decode_params));
                  // switch pinned buffer in pipeline mode to avoid an extra sync
                  buffer_indices[thread_idx] = 1 - buffer_indices[thread_idx]; 
                }, i, std::placeholders::_1
                )
            );
    }
    workers.wait();
    for ( auto& per_thread_params : params.nvjpeg_per_thread_data) {
        for(int i = 0; i < pipeline_stages; i++) {
          CHECK_CUDA(cudaEventSynchronize(per_thread_params.decode_events[i]));
        }
    }
  }
  
  CHECK_CUDA(cudaEventRecord(stopEvent, params.global_stream));
  CHECK_CUDA(cudaEventSynchronize(stopEvent));
  CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
  time = 0.001 * static_cast<double>(loopTime);

  return EXIT_SUCCESS;
}

int write_images(std::vector<nvjpegImage_t> &iout, std::vector<int> &widths,
                 std::vector<int> &heights, decode_params_t &params,
                 FileNames &filenames) {
  for (int i = 0; i < params.batch_size; i++) {
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = filenames[i].rfind("/");
    std::string sFileName =
        (std::string::npos == position)
            ? filenames[i]
            : filenames[i].substr(position + 1, filenames[i].size());

    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName
                                                : sFileName.substr(0, position);
    std::string fname(params.output_dir + "/" + sFileName + ".bmp");

    int err;
    if (params.fmt == NVJPEG_OUTPUT_RGB || params.fmt == NVJPEG_OUTPUT_BGR) {
      err = writeBMP(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                     iout[i].channel[1], iout[i].pitch[1], iout[i].channel[2],
                     iout[i].pitch[2], widths[i], heights[i]);
    } else if (params.fmt == NVJPEG_OUTPUT_RGBI ||
               params.fmt == NVJPEG_OUTPUT_BGRI) {
      // Write BMP from interleaved data
      err = writeBMPi(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                      widths[i], heights[i]);
    }
    if (err) {
      std::cout << "Cannot write output file: " << fname << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Done writing decoded image to file: " << fname << std::endl;
  }
  return EXIT_SUCCESS;
}

double process_images(FileNames &image_names, decode_params_t &params,
                      double &total) {
  // vector for storing raw files and file lengths
  FileData file_data(params.batch_size);
  std::vector<size_t> file_len(params.batch_size);
  FileNames current_names(params.batch_size);
  std::vector<int> widths(params.batch_size);
  std::vector<int> heights(params.batch_size);
  // we wrap over image files to process total_images of files
  FileNames::iterator file_iter = image_names.begin();

  int total_processed = 0;

  // output buffers
  std::vector<nvjpegImage_t> iout(params.batch_size);
  // output buffer sizes, for convenience
  std::vector<nvjpegImage_t> isz(params.batch_size);

  CHECK_CUDA(cudaStreamCreateWithFlags(&params.global_stream, cudaStreamNonBlocking));

  for (int i = 0; i < iout.size(); i++) {
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }
  ThreadPool workers(params.num_threads);
  
  double test_time = 0;
  int warmup = 0;
  while (total_processed < params.total_images) {
    if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                        file_len, current_names))
      return EXIT_FAILURE;

    if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                        current_names, params))
      return EXIT_FAILURE;

    double time;
    
    if (decode_images(file_data, file_len, iout, params, workers, time, widths, heights))
      return EXIT_FAILURE;
    if (warmup < params.warmup) {
      warmup++;
    } else {
      total_processed += params.batch_size;
      test_time += time;
    }

    if (params.write_decoded)
      write_images(iout, widths, heights, params, current_names);
  }
  total = test_time;

  release_buffers(iout);

  CHECK_CUDA(cudaStreamDestroy(params.global_stream));

  return EXIT_SUCCESS;
}


int main(int argc, const char *argv[]) {
  int pidx;

  if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
      (pidx = findParamIndex(argv, argc, "--help")) != -1) {
    std::cout << "Usage: " << argv[0]
              << " -i images_dir [-roi roi_regions] [-backend backend_enum] [-b batch_size] [-t total_images] "
                 "[-w warmup_iterations] [-o output_dir] "
                 "[-pipelined] [-batched] [-fmt output_format]\n";
    std::cout << "Parameters: " << std::endl;
    std::cout << "\timages_dir\t:\tPath to single image or directory of images"
              << std::endl;
    std::cout << "\troi_regions\t:\tSpecify the ROI in the following format [x_offset, y_offset, roi_width, roi_height]"
              << std::endl;
    std::cout << "\tbackend_eum\t:\tType of backend for the nvJPEG (0 - NVJPEG_BACKEND_DEFAULT"
              << ", 1 - NVJPEG_BACKEND_HYBRID,\n \t\t\t\t2 - NVJPEG_BACKEND_GPU_HYBRID)"
              << std::endl;
    std::cout << "\tbatch_size\t:\tDecode images from input by batches of "
                 "specified size"
              << std::endl;
    std::cout << "\ttotal_images\t:\tDecode this much images, if there are "
                 "less images \n"
              << "\t\t\t\t\tin the input than total images, decoder will loop "
                 "over the input"
              << std::endl;
    std::cout << "\twarmup_iterations\t:\tRun this amount of batches first "
                 "without measuring performance"
              << std::endl;
    std::cout << "\toutput_dir\t:\tWrite decoded images as BMPs to this directory"
              << std::endl;
    std::cout << "\tpipelined\t:\tUse decoding in phases" << std::endl;
    std::cout << "\tbatched\t\t:\tUse batched interface" << std::endl;
    std::cout << "\toutput_format\t:\tnvJPEG output format for decoding. One "
                 "of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]"
              << std::endl;
    return EXIT_SUCCESS;
  }

  decode_params_t params;

  params.input_dir = "./";
  if ((pidx = findParamIndex(argv, argc, "-i")) != -1) {
    params.input_dir = argv[pidx + 1];
  } else {
    // Search in default paths for input images.
     int found = getInputDir(params.input_dir, argv[0]);
    if (!found)
    {
      std::cout << "Please specify input directory with encoded images"<< std::endl;
      return EXIT_FAILURE;
    }
  }
  
  params.offset_x = 0;
  params.offset_y = 0;
  params.roi_width = 0;
  params.roi_height = 0;
  params.roi_on = false;
  if ((pidx = findParamIndex(argv, argc, "-roi")) != -1)
  {
    params.roi_on = true;
    if(parseDecodeCoordinates(argv[pidx + 1], params))
    {
      return EXIT_SUCCESS;
    }
  }

  params.batch_size = 1;
  if ((pidx = findParamIndex(argv, argc, "-b")) != -1) {
    params.batch_size = std::atoi(argv[pidx + 1]);
  }

  params.total_images = -1;
  if ((pidx = findParamIndex(argv, argc, "-t")) != -1) {
    params.total_images = std::atoi(argv[pidx + 1]);
  }

  params.warmup = 0;
  if ((pidx = findParamIndex(argv, argc, "-w")) != -1) {
    params.warmup = std::atoi(argv[pidx + 1]);
  }

  params.num_threads = 1;
  if ((pidx = findParamIndex(argv, argc, "-j")) != -1) {
    params.num_threads = std::atoi(argv[pidx + 1]);
  }

  params.fmt = NVJPEG_OUTPUT_RGB;
  if ((pidx = findParamIndex(argv, argc, "-fmt")) != -1) {
    std::string sfmt = argv[pidx + 1];
    if (sfmt == "rgb")
      params.fmt = NVJPEG_OUTPUT_RGB;
    else if (sfmt == "bgr")
      params.fmt = NVJPEG_OUTPUT_BGR;
    else if (sfmt == "rgbi")
      params.fmt = NVJPEG_OUTPUT_RGBI;
    else if (sfmt == "bgri")
      params.fmt = NVJPEG_OUTPUT_BGRI;
    else if (sfmt == "yuv")
      params.fmt = NVJPEG_OUTPUT_YUV;
    else if (sfmt == "y")
      params.fmt = NVJPEG_OUTPUT_Y;
    else if (sfmt == "unchanged")
      params.fmt = NVJPEG_OUTPUT_UNCHANGED;
    else {
      std::cout << "Unknown format: " << sfmt << std::endl;
      return EXIT_FAILURE;
    }
  }

  params.write_decoded = false;
  if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
    params.output_dir = argv[pidx + 1];
    if (params.fmt != NVJPEG_OUTPUT_RGB && params.fmt != NVJPEG_OUTPUT_BGR &&
        params.fmt != NVJPEG_OUTPUT_RGBI && params.fmt != NVJPEG_OUTPUT_BGRI) {
      std::cout << "We can write ony BMPs, which require output format be "
                   "either RGB/BGR or RGBi/BGRi"
                << std::endl;
      return EXIT_FAILURE;
    }
    params.write_decoded = true;
  }

  params.backend_enum = 0;
  if ((pidx = findParamIndex(argv, argc, "-backend")) != -1) {
    params.backend_enum = std::atoi(argv[pidx + 1]);
    if (params.backend_enum < 0 || params.backend_enum > 2) {
      std::cout << "backend_enum must be a valid backnend type (i.e. 0 to 2). Note that ROI function is not supported in hardware backend."
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  params.nvjpeg_per_thread_data.resize(params.num_threads);
  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  nvjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};

  nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
                                &pinned_allocator,NVJPEG_FLAGS_DEFAULT,  &params.nvjpeg_handle);

  for(auto& nvjpeg_data : params.nvjpeg_per_thread_data)
    create_nvjpeg_data(params.nvjpeg_handle, nvjpeg_data);

  // read source images
  FileNames image_names;
  readInput(params.input_dir, image_names);

  if (params.total_images == -1) {
    params.total_images = image_names.size();
  } else if (params.total_images % params.batch_size) {
    params.total_images =
        ((params.total_images) / params.batch_size) * params.batch_size;
    std::cout << "Changing total_images number to " << params.total_images
              << " to be multiple of batch_size - " << params.batch_size
              << std::endl;
  }

  std::cout << "Decoding images in directory: " << params.input_dir
            << ", total " << params.total_images << ", batchsize "
            << params.batch_size << std::endl;

  double total;
  if (process_images(image_names, params, total)) return EXIT_FAILURE;
  std::cout << "Total decoding time: " << total << " (s)" << std::endl;
  std::cout << "Avg decoding time per image: " << total / params.total_images
            << " (s)" << std::endl;
  std::cout << "Avg images per sec: " << params.total_images / total
            << std::endl;
  std::cout << "Avg decoding time per batch: "
            << total / ((params.total_images + params.batch_size - 1) /
                        params.batch_size)
            << " (s)" << std::endl;

  for(auto& nvjpeg_data : params.nvjpeg_per_thread_data)
    destroy_nvjpeg_data(nvjpeg_data);

  CHECK_NVJPEG(nvjpegDestroy(params.nvjpeg_handle));

  return EXIT_SUCCESS;
}
