/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief CloverCL static class.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Contains common functionality required by all OCL kernels 
*/

//#include "mpi.h"

#include "CloverCL.h"

#include <string>
#include <utility>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <math.h>

bool CloverCL::initialised;

cl_platform_id CloverCL::platform_c;
cl_context CloverCL::context_c;
cl_device_id CloverCL::device_c;
cl_device_id* CloverCL::devices_list; 
cl_command_queue CloverCL::queue_c;
cl_command_queue CloverCL::outoforder_queue_c;

//cl_program CloverCL::program_c;
//cl_program CloverCL::ideal_gas_prog;
cl_program CloverCL::ideal_vis_uh_prog;
cl_program CloverCL::calcdt_minred_prog;
cl_program CloverCL::accel_revert_prog;
cl_program CloverCL::pdv_fluxcalc_prog;
cl_program CloverCL::field_sumred_reset_prog;
cl_program CloverCL::initialise_generate_chunk_prog;

cl_program CloverCL::advec_cell_knl_xdir_sweep1_prog;
cl_program CloverCL::advec_cell_knl_xdir_sweep2_prog;
cl_program CloverCL::advec_cell_knl_ydir_sweep1_prog;
cl_program CloverCL::advec_cell_knl_ydir_sweep2_prog;


cl_program CloverCL::accelerate_prog;                  
//cl_program CloverCL::field_summary_prog;      
//cl_program CloverCL::flux_calc_prog;          
//cl_program CloverCL::reset_field_prog;
//cl_program CloverCL::revert_prog;
//cl_program CloverCL::viscosity_prog;
//cl_program CloverCL::calc_dt_prog;                   
//cl_program CloverCL::pdv_prog;                
//cl_program CloverCL::initialise_chunk_prog;   
//cl_program CloverCL::min_reduction_prog;      
//cl_program CloverCL::sum_reduction_prog;
//cl_program CloverCL::update_halo_prog;
//cl_program CloverCL::generate_chunk_prog;     

//cl_program CloverCL::advec_cell_knl_xdir_sec1_sweep1_prog; 
//cl_program CloverCL::advec_cell_knl_xdir_sec1_sweep2_prog; 
//cl_program CloverCL::advec_cell_knl_xdir_sec2_prog;        
//cl_program CloverCL::advec_cell_knl_xdir_sec3_prog;        
//cl_program CloverCL::advec_cell_knl_y_sec1_sweep1_prog;    
//cl_program CloverCL::advec_cell_knl_y_sec1_sweep2_prog;     
//cl_program CloverCL::advec_cell_knl_y_sec2_prog;            
//cl_program CloverCL::advec_cell_knl_y_sec3_prog;            
cl_program CloverCL::advec_mom_knl_vol_prog;             
cl_program CloverCL::advec_mom_knl_node_x_prog;          
cl_program CloverCL::advec_mom_knl_node_y_prog;          
cl_program CloverCL::advec_mom_knl_node_mass_pre_x_prog; 
cl_program CloverCL::advec_mom_knl_node_mass_pre_y_prog; 
cl_program CloverCL::advec_mom_knl_mom_flux_x_vec1_prog;    
cl_program CloverCL::advec_mom_knl_mom_flux_x_notvec1_prog; 
cl_program CloverCL::advec_mom_knl_mom_flux_y_vec1_prog;    
cl_program CloverCL::advec_mom_knl_mom_flux_y_notvec1_prog; 
cl_program CloverCL::advec_mom_knl_vel_x_prog;           
cl_program CloverCL::advec_mom_knl_vel_y_prog;           

//cl_program CloverCL::pack_comms_buffers_prog; 
//cl_program CloverCL::unpack_comms_buffers_prog;
//cl_program CloverCL::read_comm_buffers_prog;
//cl_program CloverCL::write_comm_buffers_prog;

cl_uint CloverCL::native_wg_multiple;
size_t CloverCL::prefer_wg_multiple;
size_t CloverCL::max_reduction_wg_size;
cl_uint CloverCL::device_procs;
size_t CloverCL::device_max_wg_size;
cl_ulong CloverCL::device_local_mem_size;
cl_device_type CloverCL::device_type; 
size_t CloverCL::device_prefer_wg_multiple;

int CloverCL::number_of_red_levels;
int CloverCL::xmax_plusfour_rounded;
int CloverCL::xmax_plusfive_rounded;
int CloverCL::ymax_plusfour_rounded;
int CloverCL::ymax_plusfive_rounded;

int CloverCL::mpi_rank; 
int CloverCL::xmax_c;
int CloverCL::ymax_c;

cl_mem CloverCL::density0_buffer_c;
cl_mem CloverCL::density1_buffer_c;
cl_mem CloverCL::energy0_buffer_c;
cl_mem CloverCL::energy1_buffer_c;
cl_mem CloverCL::pressure_buffer_c;
cl_mem CloverCL::soundspeed_buffer_c;
cl_mem CloverCL::celldx_buffer_c;
cl_mem CloverCL::celldy_buffer_c;
cl_mem CloverCL::viscosity_buffer_c;
cl_mem CloverCL::xvel0_buffer_c;
cl_mem CloverCL::yvel0_buffer_c;
cl_mem CloverCL::xvel1_buffer_c;
cl_mem CloverCL::yvel1_buffer_c;
cl_mem CloverCL::xarea_buffer_c;
cl_mem CloverCL::yarea_buffer_c;
cl_mem CloverCL::vol_flux_x_buffer_c;
cl_mem CloverCL::vol_flux_y_buffer_c;
cl_mem CloverCL::mass_flux_x_buffer_c;
cl_mem CloverCL::mass_flux_y_buffer_c;
cl_mem CloverCL::stepbymass_buffer_c;
cl_mem CloverCL::volume_buffer_c;
cl_mem CloverCL::node_flux_buffer_c;
cl_mem CloverCL::node_mass_post_buffer_c;
cl_mem CloverCL::node_mass_pre_buffer_c;
cl_mem CloverCL::advec_vel_buffer_c;
cl_mem CloverCL::mom_flux_buffer_c;
cl_mem CloverCL::pre_vol_buffer_c;
cl_mem CloverCL::post_vol_buffer_c;
cl_mem CloverCL::vertexdx_buffer_c;
cl_mem CloverCL::vertexx_buffer_c;
cl_mem CloverCL::vertexdy_buffer_c;
cl_mem CloverCL::vertexy_buffer_c;
cl_mem CloverCL::pre_mass_buffer_c;
cl_mem CloverCL::post_mass_buffer_c;
cl_mem CloverCL::advec_vol_buffer_c;
cl_mem CloverCL::post_ener_buffer_c;
cl_mem CloverCL::ener_flux_buffer_c;
cl_mem CloverCL::cellx_buffer_c;
cl_mem CloverCL::celly_buffer_c;
cl_mem CloverCL::dt_min_val_array_buffer_c;
cl_mem CloverCL::dt_min_val_buffer_c;
cl_mem CloverCL::vol_tmp_buffer_c;
cl_mem CloverCL::mass_tmp_buffer_c;
cl_mem CloverCL::ie_tmp_buffer_c;
cl_mem CloverCL::ke_tmp_buffer_c;
cl_mem CloverCL::press_tmp_buffer_c;
cl_mem CloverCL::vol_sum_val_buffer_c;
cl_mem CloverCL::mass_sum_val_buffer_c;
cl_mem CloverCL::ie_sum_val_buffer_c;
cl_mem CloverCL::ke_sum_val_buffer_c;
cl_mem CloverCL::press_sum_val_buffer_c;
cl_mem CloverCL::state_density_buffer_c;
cl_mem CloverCL::state_energy_buffer_c;
cl_mem CloverCL::state_xvel_buffer_c;
cl_mem CloverCL::state_yvel_buffer_c;
cl_mem CloverCL::state_xmin_buffer_c;
cl_mem CloverCL::state_xmax_buffer_c;
cl_mem CloverCL::state_ymin_buffer_c;
cl_mem CloverCL::state_ymax_buffer_c;
cl_mem CloverCL::state_radius_buffer_c;
cl_mem CloverCL::state_geometry_buffer_c;

cl_mem CloverCL::cpu_min_red_buffer_c;
cl_mem CloverCL::cpu_vol_red_buffer_c;
cl_mem CloverCL::cpu_mass_red_buffer_c;
cl_mem CloverCL::cpu_ie_red_buffer_c;
cl_mem CloverCL::cpu_ke_red_buffer_c;
cl_mem CloverCL::cpu_press_red_buffer_c;

cl_mem CloverCL::top_send_buffer_c;
cl_mem CloverCL::top_recv_buffer_c;
cl_mem CloverCL::bottom_send_buffer_c;
cl_mem CloverCL::bottom_recv_buffer_c;
cl_mem CloverCL::left_send_buffer_c;
cl_mem CloverCL::left_recv_buffer_c;
cl_mem CloverCL::right_send_buffer_c;
cl_mem CloverCL::right_recv_buffer_c;

cl_kernel CloverCL::ideal_gas_predict_knl_c;
cl_kernel CloverCL::ideal_gas_NO_predict_knl_c;
cl_kernel CloverCL::viscosity_knl_c;
cl_kernel CloverCL::flux_calc_knl_c;
cl_kernel CloverCL::accelerate_knl_c;
cl_kernel CloverCL::advec_mom_vol_knl_c;
cl_kernel CloverCL::advec_mom_node_x_knl_c;
cl_kernel CloverCL::advec_mom_node_mass_pre_x_knl_c;
cl_kernel CloverCL::advec_mom_flux_x_vec1_knl_c;
cl_kernel CloverCL::advec_mom_flux_x_vecnot1_knl_c;
cl_kernel CloverCL::advec_mom_vel_x_knl_c;
cl_kernel CloverCL::advec_mom_node_y_knl_c;
cl_kernel CloverCL::advec_mom_node_mass_pre_y_knl_c;
cl_kernel CloverCL::advec_mom_flux_y_vec1_knl_c;
cl_kernel CloverCL::advec_mom_flux_y_vecnot1_knl_c;
cl_kernel CloverCL::advec_mom_vel_y_knl_c;
cl_kernel CloverCL::dt_calc_knl_c;

cl_kernel CloverCL::advec_cell_xdir_sweep1_sec1_knl_c; 
cl_kernel CloverCL::advec_cell_xdir_sweep1_sec2_knl_c; 
cl_kernel CloverCL::advec_cell_xdir_sweep1_sec3_knl_c; 
cl_kernel CloverCL::advec_cell_xdir_sweep2_sec1_knl_c; 
cl_kernel CloverCL::advec_cell_xdir_sweep2_sec2_knl_c; 
cl_kernel CloverCL::advec_cell_xdir_sweep2_sec3_knl_c; 
cl_kernel CloverCL::advec_cell_ydir_sweep1_sec1_knl_c; 
cl_kernel CloverCL::advec_cell_ydir_sweep1_sec2_knl_c; 
cl_kernel CloverCL::advec_cell_ydir_sweep1_sec3_knl_c; 
cl_kernel CloverCL::advec_cell_ydir_sweep2_sec1_knl_c; 
cl_kernel CloverCL::advec_cell_ydir_sweep2_sec2_knl_c; 
cl_kernel CloverCL::advec_cell_ydir_sweep2_sec3_knl_c; 

cl_kernel CloverCL::pdv_correct_knl_c;
cl_kernel CloverCL::pdv_predict_knl_c;
cl_kernel CloverCL::reset_field_knl_c;
cl_kernel CloverCL::revert_knl_c;
cl_kernel CloverCL::generate_chunk_knl_c;
cl_kernel CloverCL::initialise_chunk_cell_x_knl_c;
cl_kernel CloverCL::initialise_chunk_cell_y_knl_c;
cl_kernel CloverCL::initialise_chunk_vertex_x_knl_c;
cl_kernel CloverCL::initialise_chunk_vertex_y_knl_c;
cl_kernel CloverCL::initialise_chunk_volume_area_knl_c;
cl_kernel CloverCL::field_summary_knl_c;

cl_kernel CloverCL::update_halo_left_cell_knl_c;
cl_kernel CloverCL::update_halo_right_cell_knl_c;
cl_kernel CloverCL::update_halo_top_cell_knl_c;
cl_kernel CloverCL::update_halo_bottom_cell_knl_c;

cl_kernel CloverCL::update_halo_left_vel_knl_c;
cl_kernel CloverCL::update_halo_right_vel_knl_c;
cl_kernel CloverCL::update_halo_top_vel_knl_c;
cl_kernel CloverCL::update_halo_bottom_vel_knl_c;

cl_kernel CloverCL::update_halo_left_flux_x_knl_c;
cl_kernel CloverCL::update_halo_right_flux_x_knl_c;
cl_kernel CloverCL::update_halo_top_flux_x_knl_c;
cl_kernel CloverCL::update_halo_bottom_flux_x_knl_c;

cl_kernel CloverCL::update_halo_left_flux_y_knl_c;
cl_kernel CloverCL::update_halo_right_flux_y_knl_c;
cl_kernel CloverCL::update_halo_top_flux_y_knl_c;
cl_kernel CloverCL::update_halo_bottom_flux_y_knl_c;

cl_kernel CloverCL::read_top_buffer_knl_c;
cl_kernel CloverCL::read_right_buffer_knl_c;
cl_kernel CloverCL::read_bottom_buffer_knl_c;
cl_kernel CloverCL::read_left_buffer_knl_c;
cl_kernel CloverCL::write_top_buffer_knl_c;
cl_kernel CloverCL::write_right_buffer_knl_c;
cl_kernel CloverCL::write_bottom_buffer_knl_c;
cl_kernel CloverCL::write_left_buffer_knl_c;

std::vector<cl_kernel> CloverCL::min_reduction_kernels;
std::vector<cl_kernel> CloverCL::vol_sum_reduction_kernels;
std::vector<cl_kernel> CloverCL::mass_sum_reduction_kernels;
std::vector<cl_kernel> CloverCL::ie_sum_reduction_kernels;
std::vector<cl_kernel> CloverCL::ke_sum_reduction_kernels;
std::vector<cl_kernel> CloverCL::press_sum_reduction_kernels;

std::vector<size_t> CloverCL::num_workitems_tolaunch;
std::vector<size_t> CloverCL::num_workitems_per_wg;
std::vector<int> CloverCL::local_mem_size;
std::vector<int> CloverCL::size_limits;
std::vector<int> CloverCL::buffer_sizes;
std::vector<bool> CloverCL::input_even;
std::vector<int> CloverCL::num_elements_per_wi;

std::vector<cl_mem> CloverCL::min_interBuffers;
std::vector<cl_mem> CloverCL::vol_interBuffers;
std::vector<cl_mem> CloverCL::mass_interBuffers;
std::vector<cl_mem> CloverCL::ie_interBuffers;
std::vector<cl_mem> CloverCL::ke_interBuffers;
std::vector<cl_mem> CloverCL::press_interBuffers;

std::vector<int> CloverCL::min_local_memory_objects;
std::vector<int> CloverCL::vol_local_memory_objects;
std::vector<int> CloverCL::mass_local_memory_objects;
std::vector<int> CloverCL::ie_local_memory_objects;
std::vector<int> CloverCL::ke_local_memory_objects;
std::vector<int> CloverCL::press_local_memory_objects;

//std::vector<cl::Event> CloverCL::global_events;
cl_event CloverCL::last_event;


void CloverCL::init(
        std::string platform_name,
        std::string platform_type,
        int x_min,
        int x_max,
        int y_min,
        int y_max,
        int num_states,
        double g_small,
        double g_big,
        double dtmin,
        double dtc_safe,
        double dtu_safe,
        double dtv_safe,
        double dtdiv_safe) 
{
#ifdef OCL_VERBOSE
    std::cout << "num states = " << num_states << std::endl;
    std::cout << "x_max = " << x_max << std::endl;
    std::cout << "y_max = " << y_max << std::endl;

    printDeviceInformation();
#endif

    initPlatform(platform_name);
    initContext(platform_type);
    initDevice(0);
    initCommandQueue();
    loadProgram(x_min, x_max, y_min, y_max);
    createKernelObjects();

    //determineWorkGroupSizeInfo();

    calculateKernelLaunchParams(x_max, y_max);

    //calculateReductionStructure(x_max, y_max);

    createBuffers(x_max, y_max, num_states);
    //allocateReductionInterBuffers();
    //allocateLocalMemoryObjects();
    //build_reduction_kernel_objects(); 
    std::cout << "after build reduction kernel objects " << std::endl;

#ifdef DUMP_BINARY
    dumpBinary();
#endif

    initialiseKernelArgs(x_min, x_max, y_min, y_max,
                         g_small, g_big, dtmin, dtc_safe,
                         dtu_safe, dtv_safe, dtdiv_safe);
    initialised = true;

    std::cout << "after initialise kernel args" << std::endl;

    //MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    xmax_c = x_max;
    ymax_c = y_max; 
    std::cout << "after mpi comm rank" << std::endl;

    std::cout << "at end of init section" << std::endl; 
}


void CloverCL::calculateKernelLaunchParams(int xmax, int ymax) {

    int x_rnd, y_rnd; 

    int x_divisor = fixed_wg_min_size_large_dim;
    int y_divisor = fixed_wg_min_size_large_dim;

    x_rnd = ( (xmax+4) / x_divisor ) * x_divisor;

    if (x_rnd != xmax+4) {
        x_rnd = x_rnd + x_divisor; 
    }
    
    xmax_plusfour_rounded = x_rnd; 

    x_rnd = ( (xmax+5) / x_divisor ) * x_divisor;

    if (x_rnd != xmax+5) {
        x_rnd = x_rnd + x_divisor; 
    }

    xmax_plusfive_rounded = x_rnd; 


    y_rnd = ( (ymax+4) / y_divisor ) * y_divisor;

    if (y_rnd != ymax+4) {
        y_rnd = y_rnd + y_divisor; 
    }

    ymax_plusfour_rounded = y_rnd; 

    y_rnd = ( (ymax+5) / y_divisor ) * y_divisor;

    if (y_rnd != ymax+5) {
        y_rnd = y_rnd + y_divisor; 
    }

    ymax_plusfive_rounded = y_rnd; 

#ifdef OCL_VERBOSE
    std::cout << "Kernel launch xmaxplusfour rounded: " << xmax_plusfour_rounded << std::endl;
    std::cout << "Kernel launch xmaxplusfive rounded: " << xmax_plusfive_rounded << std::endl;
    std::cout << "Kernel launch ymaxplusfour rounded: " << ymax_plusfour_rounded << std::endl;
    std::cout << "Kernel launch ymaxplusfive rounded: " << ymax_plusfive_rounded << std::endl;
#endif
}

void CloverCL::determineWorkGroupSizeInfo() {

    cl_int err; 

    err = clGetKernelWorkGroupInfo(ideal_gas_predict_knl_c, device_c, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &prefer_wg_multiple, NULL); 
    err = clGetKernelWorkGroupInfo(ideal_gas_predict_knl_c, device_c, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_reduction_wg_size, NULL); 

    err = clGetDeviceInfo(device_c, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &device_prefer_wg_multiple, NULL); 
    err = clGetDeviceInfo(device_c, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &native_wg_multiple, NULL); 
    
    err = clGetDeviceInfo(device_c, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &device_procs, NULL); 
    err = clGetDeviceInfo(device_c, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_uint), &device_max_wg_size, NULL); 
    err = clGetDeviceInfo(device_c, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_local_mem_size, NULL); 

    //ideal_gas_predict_knl.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &prefer_wg_multiple);
    //ideal_gas_predict_knl.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &max_reduction_wg_size);

    //device.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_procs);
    //device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &device_max_wg_size);
    //device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_local_mem_size);

#ifdef OCL_VERBOSE
    std::cout << "Kernel prefer work group multiple: " << prefer_wg_multiple << std::endl;
    std::cout << "Kernel max work group size: " << max_reduction_wg_size << std::endl;

    std::cout << "Device preferred vector width double multiple: " << device_prefer_wg_multiple << std::endl;
    std::cout << "Device native vector width double: " << native_wg_multiple << std::endl;
    
    std::cout << "Device Num of compute units: " << device_procs << std::endl;
    std::cout << "Device Max WG Size: " << device_max_wg_size << std::endl;
    std::cout << "Device Local memmory size: " << device_local_mem_size << std::endl;

    if (device_type == CL_DEVICE_TYPE_CPU) { 
        std::cout << "Device Type selected: CPU" << std::endl;
    }
    else if (device_type == CL_DEVICE_TYPE_GPU) {
        std::cout << "Device Type selected: GPU" << std::endl;
    }
    else if (device_type == CL_DEVICE_TYPE_ACCELERATOR) {
        std::cout << "Device Type selected: ACCELERATOR" << std::endl;
    }
    else if (device_type == CL_DEVICE_TYPE_DEFAULT) {
        std::cout << "Device Type selected: DEFAULT" << std::endl;
    }
    else {
        std::cout << "ERROR Device Type selected: NOT SUPPORTED" << std::endl;
    }
#endif

}

void CloverCL::build_reduction_kernel_objects() {

    cl_int err; 

    min_reduction_kernels.clear();
    vol_sum_reduction_kernels.clear();
    mass_sum_reduction_kernels.clear();
    ie_sum_reduction_kernels.clear();
    ke_sum_reduction_kernels.clear();
    press_sum_reduction_kernels.clear();

    if (device_type == CL_DEVICE_TYPE_CPU) {
        //build the CPU and Phi reduction objects 

        if ( number_of_red_levels == 1 ) { 

            //build level 1 of CPU reduction 
            min_reduction_kernels.push_back( clCreateKernel(calcdt_minred_prog, "reduction_minimum_cpu_ocl_kernel", &err) );
            vol_sum_reduction_kernels.push_back( clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            mass_sum_reduction_kernels.push_back( clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            ie_sum_reduction_kernels.push_back( clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            ke_sum_reduction_kernels.push_back( clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            press_sum_reduction_kernels.push_back( clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));

            err = clSetKernelArg(min_reduction_kernels[0],        0, sizeof(cl_mem), &CloverCL::dt_min_val_array_buffer_c);
            err = clSetKernelArg(vol_sum_reduction_kernels[0],    0, sizeof(cl_mem), &CloverCL::vol_tmp_buffer_c);
            err = clSetKernelArg(mass_sum_reduction_kernels[0],   0, sizeof(cl_mem), &CloverCL::mass_tmp_buffer_c);
            err = clSetKernelArg(ie_sum_reduction_kernels[0],     0, sizeof(cl_mem), &CloverCL::ie_tmp_buffer_c);
            err = clSetKernelArg(ke_sum_reduction_kernels[0],     0, sizeof(cl_mem), &CloverCL::ke_tmp_buffer_c);
            err = clSetKernelArg(press_sum_reduction_kernels[0],  0, sizeof(cl_mem), &CloverCL::press_tmp_buffer_c);

            //min_reduction_kernels[0].setArg(      0, CloverCL::dt_min_val_array_buffer);
            //vol_sum_reduction_kernels[0].setArg(  0, CloverCL::vol_tmp_buffer);
            //mass_sum_reduction_kernels[0].setArg( 0, CloverCL::mass_tmp_buffer);
            //ie_sum_reduction_kernels[0].setArg(   0, CloverCL::ie_tmp_buffer);
            //ke_sum_reduction_kernels[0].setArg(   0, CloverCL::ke_tmp_buffer);
            //press_sum_reduction_kernels[0].setArg(0, CloverCL::press_tmp_buffer);

            err = clSetKernelArg(min_reduction_kernels[1],        1, sizeof(cl_mem), &CloverCL::dt_min_val_buffer_c); 
            err = clSetKernelArg(vol_sum_reduction_kernels[1],    1, sizeof(cl_mem), &CloverCL::vol_sum_val_buffer_c); 
            err = clSetKernelArg(mass_sum_reduction_kernels[1],   1, sizeof(cl_mem), &CloverCL::mass_sum_val_buffer_c); 
            err = clSetKernelArg(ie_sum_reduction_kernels[1],     1, sizeof(cl_mem), &CloverCL::ie_sum_val_buffer_c); 
            err = clSetKernelArg(ke_sum_reduction_kernels[1],     1, sizeof(cl_mem), &CloverCL::ke_sum_val_buffer_c); 
            err = clSetKernelArg(press_sum_reduction_kernels[1],  1, sizeof(cl_mem), &CloverCL::press_sum_val_buffer_c); 

            //min_reduction_kernels[1].setArg(      1, CloverCL::dt_min_val_buffer);
            //vol_sum_reduction_kernels[1].setArg(1, CloverCL::vol_sum_val_buffer); 
            //mass_sum_reduction_kernels[1].setArg(1, CloverCL::mass_sum_val_buffer); 
            //ie_sum_reduction_kernels[1].setArg(1, CloverCL::ie_sum_val_buffer);
            //ke_sum_reduction_kernels[1].setArg(1, CloverCL::ke_sum_val_buffer); 
            //press_sum_reduction_kernels[1].setArg(1, CloverCL::press_sum_val_buffer);

            err = clSetKernelArg(min_reduction_kernels[0],        2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(vol_sum_reduction_kernels[0],    2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(mass_sum_reduction_kernels[0],   2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(ie_sum_reduction_kernels[0],     2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(ke_sum_reduction_kernels[0],     2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(press_sum_reduction_kernels[0],  2, sizeof(int), &CloverCL::num_elements_per_wi[0]);

            //min_reduction_kernels[0].setArg(      2, CloverCL::num_elements_per_wi[0]);
            //vol_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            //mass_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]);
            //ie_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            //ke_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            //press_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 

        }
        else {

            //build level 1 of CPU reduction 
            min_reduction_kernels.push_back(        clCreateKernel(calcdt_minred_prog, "reduction_minimum_cpu_ocl_kernel", &err) );
            vol_sum_reduction_kernels.push_back(    clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            mass_sum_reduction_kernels.push_back(   clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            ie_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            ke_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            press_sum_reduction_kernels.push_back(  clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));

            err = clSetKernelArg(min_reduction_kernels[0],       0, sizeof(cl_mem), &CloverCL::dt_min_val_array_buffer_c);
            err = clSetKernelArg(vol_sum_reduction_kernels[0],   0, sizeof(cl_mem), &CloverCL::vol_tmp_buffer_c);
            err = clSetKernelArg(mass_sum_reduction_kernels[0],  0, sizeof(cl_mem), &CloverCL::mass_tmp_buffer_c);
            err = clSetKernelArg(ie_sum_reduction_kernels[0],    0, sizeof(cl_mem), &CloverCL::ie_tmp_buffer_c);
            err = clSetKernelArg(ke_sum_reduction_kernels[0],    0, sizeof(cl_mem), &CloverCL::ke_tmp_buffer_c);
            err = clSetKernelArg(press_sum_reduction_kernels[0], 0, sizeof(cl_mem), &CloverCL::press_tmp_buffer_c);

            //min_reduction_kernels[0].setArg(      0, CloverCL::dt_min_val_array_buffer);
            //vol_sum_reduction_kernels[0].setArg(  0, CloverCL::vol_tmp_buffer);
            //mass_sum_reduction_kernels[0].setArg( 0, CloverCL::mass_tmp_buffer);
            //ie_sum_reduction_kernels[0].setArg(   0, CloverCL::ie_tmp_buffer);
            //ke_sum_reduction_kernels[0].setArg(   0, CloverCL::ke_tmp_buffer);
            //press_sum_reduction_kernels[0].setArg(0, CloverCL::press_tmp_buffer);
            
            err = clSetKernelArg(min_reduction_kernels[0],       1, sizeof(cl_mem), &CloverCL::cpu_min_red_buffer_c); 
            err = clSetKernelArg(vol_sum_reduction_kernels[0],   1, sizeof(cl_mem), &CloverCL::cpu_vol_red_buffer_c); 
            err = clSetKernelArg(mass_sum_reduction_kernels[0],  1, sizeof(cl_mem), &CloverCL::cpu_mass_red_buffer_c); 
            err = clSetKernelArg(ie_sum_reduction_kernels[0],    1, sizeof(cl_mem), &CloverCL::cpu_ie_red_buffer_c); 
            err = clSetKernelArg(ke_sum_reduction_kernels[0],    1, sizeof(cl_mem), &CloverCL::cpu_ke_red_buffer_c); 
            err = clSetKernelArg(press_sum_reduction_kernels[0], 1, sizeof(cl_mem), &CloverCL::cpu_press_red_buffer_c); 

            //min_reduction_kernels[0].setArg(      1, CloverCL::cpu_min_red_buffer);
            //vol_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_vol_red_buffer); 
            //mass_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_mass_red_buffer); 
            //ie_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_ie_red_buffer); 
            //ke_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_ke_red_buffer); 
            //press_sum_reduction_kernels[0].setArg(1, CloverCL::cpu_press_red_buffer); 

            err = clSetKernelArg(min_reduction_kernels[0],       2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(vol_sum_reduction_kernels[0],   2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(mass_sum_reduction_kernels[0],  2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(ie_sum_reduction_kernels[0],    2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(ke_sum_reduction_kernels[0],    2, sizeof(int), &CloverCL::num_elements_per_wi[0]);
            err = clSetKernelArg(press_sum_reduction_kernels[0], 2, sizeof(int), &CloverCL::num_elements_per_wi[0]);

            //min_reduction_kernels[0].setArg(      2, CloverCL::num_elements_per_wi[0]);
            //vol_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            //mass_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]);
            //ie_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            //ke_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 
            //press_sum_reduction_kernels[0].setArg(2, CloverCL::num_elements_per_wi[0]); 


            //build level 2 of CPU reduction 
            min_reduction_kernels.push_back(        clCreateKernel(calcdt_minred_prog, "reduction_minimum_cpu_ocl_kernel", &err) );
            vol_sum_reduction_kernels.push_back(    clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            mass_sum_reduction_kernels.push_back(   clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            ie_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            ke_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));
            press_sum_reduction_kernels.push_back(  clCreateKernel(field_sumred_reset_prog, "reduction_sum_cpu_ocl_kernel", &err));

            err = clSetKernelArg(min_reduction_kernels[1],       0, sizeof(cl_mem), &CloverCL::cpu_min_red_buffer_c); 
            err = clSetKernelArg(vol_sum_reduction_kernels[1],   0, sizeof(cl_mem), &CloverCL::cpu_vol_red_buffer_c); 
            err = clSetKernelArg(mass_sum_reduction_kernels[1],  0, sizeof(cl_mem), &CloverCL::cpu_mass_red_buffer_c); 
            err = clSetKernelArg(ie_sum_reduction_kernels[1],    0, sizeof(cl_mem), &CloverCL::cpu_ie_red_buffer_c); 
            err = clSetKernelArg(ke_sum_reduction_kernels[1],    0, sizeof(cl_mem), &CloverCL::cpu_ke_red_buffer_c); 
            err = clSetKernelArg(press_sum_reduction_kernels[1], 0, sizeof(cl_mem), &CloverCL::cpu_press_red_buffer_c); 
        

            //min_reduction_kernels[1].setArg(      0, CloverCL::cpu_min_red_buffer);
            //vol_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_vol_red_buffer); 
            //mass_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_mass_red_buffer); 
            //ie_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_ie_red_buffer); 
            //ke_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_ke_red_buffer);
            //press_sum_reduction_kernels[1].setArg(0, CloverCL::cpu_press_red_buffer); 

            err = clSetKernelArg(min_reduction_kernels[1],       1, sizeof(cl_mem), &CloverCL::dt_min_val_buffer_c); 
            err = clSetKernelArg(vol_sum_reduction_kernels[1],   1, sizeof(cl_mem), &CloverCL::vol_sum_val_buffer_c); 
            err = clSetKernelArg(mass_sum_reduction_kernels[1],  1, sizeof(cl_mem), &CloverCL::mass_sum_val_buffer_c); 
            err = clSetKernelArg(ie_sum_reduction_kernels[1],    1, sizeof(cl_mem), &CloverCL::ie_sum_val_buffer_c); 
            err = clSetKernelArg(ke_sum_reduction_kernels[1],    1, sizeof(cl_mem), &CloverCL::ke_sum_val_buffer_c); 
            err = clSetKernelArg(press_sum_reduction_kernels[1], 1, sizeof(cl_mem), &CloverCL::press_sum_val_buffer_c); 
            
            //min_reduction_kernels[1].setArg(      1, CloverCL::dt_min_val_buffer);
            //vol_sum_reduction_kernels[1].setArg(1, CloverCL::vol_sum_val_buffer); 
            //mass_sum_reduction_kernels[1].setArg(1, CloverCL::mass_sum_val_buffer); 
            //ie_sum_reduction_kernels[1].setArg(1, CloverCL::ie_sum_val_buffer);
            //ke_sum_reduction_kernels[1].setArg(1, CloverCL::ke_sum_val_buffer); 
            //press_sum_reduction_kernels[1].setArg(1, CloverCL::press_sum_val_buffer);

            err = clSetKernelArg(min_reduction_kernels[1],       2, sizeof(int), &CloverCL::num_elements_per_wi[1]); 
            err = clSetKernelArg(vol_sum_reduction_kernels[1],   2, sizeof(int), &CloverCL::num_elements_per_wi[1]);
            err = clSetKernelArg(mass_sum_reduction_kernels[1],  2, sizeof(int), &CloverCL::num_elements_per_wi[1]);
            err = clSetKernelArg(ie_sum_reduction_kernels[1],    2, sizeof(int), &CloverCL::num_elements_per_wi[1]);
            err = clSetKernelArg(ke_sum_reduction_kernels[1],    2, sizeof(int), &CloverCL::num_elements_per_wi[1]);
            err = clSetKernelArg(press_sum_reduction_kernels[1], 2, sizeof(int), &CloverCL::num_elements_per_wi[1]);

            //min_reduction_kernels[1].setArg(      2, CloverCL::num_elements_per_wi[1]);
            //vol_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            //mass_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            //ie_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            //ke_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 
            //press_sum_reduction_kernels[1].setArg(2, CloverCL::num_elements_per_wi[1]); 

        }
        
    }
    else if ( (CloverCL::device_type == CL_DEVICE_TYPE_GPU) || (device_type == CL_DEVICE_TYPE_ACCELERATOR) ){
        //build the GPU reduction objects

        for (int i=1; i<=CloverCL::number_of_red_levels; i++) {

            if (CloverCL::size_limits[i-1] == -1) { 

                //build a normal GPU reduction kernel
                min_reduction_kernels.push_back(        clCreateKernel(calcdt_minred_prog, "reduction_minimum_ocl_kernel", &err) );
                vol_sum_reduction_kernels.push_back(    clCreateKernel(field_sumred_reset_prog, "reduction_sum_ocl_kernel", &err));
                mass_sum_reduction_kernels.push_back(   clCreateKernel(field_sumred_reset_prog, "reduction_sum_ocl_kernel", &err));
                ie_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_ocl_kernel", &err));
                ke_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_ocl_kernel", &err));
                press_sum_reduction_kernels.push_back(  clCreateKernel(field_sumred_reset_prog, "reduction_sum_ocl_kernel", &err));

                if (i==1) {
                    err = clSetKernelArg(min_reduction_kernels[i-1],        0, sizeof(cl_mem), &CloverCL::dt_min_val_array_buffer_c);
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    0, sizeof(cl_mem), &CloverCL::vol_tmp_buffer_c);
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   0, sizeof(cl_mem), &CloverCL::mass_tmp_buffer_c);
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ie_tmp_buffer_c);
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ke_tmp_buffer_c);
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  0, sizeof(cl_mem), &CloverCL::press_tmp_buffer_c);

                    //min_reduction_kernels[i-1].setArg(      0, CloverCL::dt_min_val_array_buffer);
                    //vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::vol_tmp_buffer);
                    //mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::mass_tmp_buffer);
                    //ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ie_tmp_buffer);
                    //ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ke_tmp_buffer);
                    //press_sum_reduction_kernels[i-1].setArg(0, CloverCL::press_tmp_buffer);
                }
                else {
                    err = clSetKernelArg(min_reduction_kernels[i-1],        0, sizeof(cl_mem), &CloverCL::min_interBuffers[i-2]);
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    0, sizeof(cl_mem), &CloverCL::vol_interBuffers[i-2]);
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   0, sizeof(cl_mem), &CloverCL::mass_interBuffers[i-2]);
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ie_interBuffers[i-2]);
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ke_interBuffers[i-2]);
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  0, sizeof(cl_mem), &CloverCL::press_interBuffers[i-2]);

                    //min_reduction_kernels[i-1].setArg(0, CloverCL::min_interBuffers[i-2]);
                    //vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::vol_interBuffers[i-2]);
                    //mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::mass_interBuffers[i-2]);
                    //ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ie_interBuffers[i-2]);
                    //ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ke_interBuffers[i-2]);
                    //press_sum_reduction_kernels[i-1].setArg(0, CloverCL::press_interBuffers[i-2]);
                }

                err = clSetKernelArg(min_reduction_kernels[i-1],       1, CloverCL::min_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    1, CloverCL::vol_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   1, CloverCL::mass_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     1, CloverCL::ie_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     1, CloverCL::ke_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(press_sum_reduction_kernels[i-1],  1, CloverCL::press_local_memory_objects[i-1], NULL); 

                //min_reduction_kernels[i-1].setArg(1, CloverCL::min_local_memory_objects[i-1]);
                //vol_sum_reduction_kernels[i-1].setArg(1,   CloverCL::vol_local_memory_objects[i-1]);
                //mass_sum_reduction_kernels[i-1].setArg(1,  CloverCL::mass_local_memory_objects[i-1]);
                //ie_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ie_local_memory_objects[i-1]);
                //ke_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ke_local_memory_objects[i-1]);
                //press_sum_reduction_kernels[i-1].setArg(1, CloverCL::press_local_memory_objects[i-1]);

                if (i==CloverCL::number_of_red_levels) {
                    err = clSetKernelArg(min_reduction_kernels[i-1],       2, sizeof(cl_mem), &CloverCL::dt_min_val_buffer_c); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    2, sizeof(cl_mem), &CloverCL::vol_sum_val_buffer_c); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   2, sizeof(cl_mem), &CloverCL::mass_sum_val_buffer_c); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ie_sum_val_buffer_c); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ke_sum_val_buffer_c); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  2, sizeof(cl_mem), &CloverCL::press_sum_val_buffer_c); 

                    //min_reduction_kernels[i-1].setArg(2, CloverCL::dt_min_val_buffer);
                    //vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_sum_val_buffer);
                    //mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_sum_val_buffer);
                    //ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_sum_val_buffer);
                    //ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_sum_val_buffer);
                    //press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_sum_val_buffer);
                }
                else {
                    err = clSetKernelArg(min_reduction_kernels[i-1],       2, sizeof(cl_mem), &CloverCL::min_interBuffers[i-1]); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    2, sizeof(cl_mem), &CloverCL::vol_interBuffers[i-1]); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   2, sizeof(cl_mem), &CloverCL::mass_interBuffers[i-1]); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ie_interBuffers[i-1]); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ke_interBuffers[i-1]); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  2, sizeof(cl_mem), &CloverCL::press_interBuffers[i-1]); 

                    //min_reduction_kernels[i-1].setArg(2, CloverCL::min_interBuffers[i-1]);
                    //vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_interBuffers[i-1]);
                    //mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_interBuffers[i-1]);
                    //ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_interBuffers[i-1]);
                    //ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_interBuffers[i-1]);
                    //press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_interBuffers[i-1]);
                }
            }
            else {

                //build a last level GPU reduction kernel
                min_reduction_kernels.push_back(        clCreateKernel(calcdt_minred_prog, "reduction_minimum_last_ocl_kernel", &err)  );
                vol_sum_reduction_kernels.push_back(    clCreateKernel(field_sumred_reset_prog, "reduction_sum_last_ocl_kernel", &err));
                mass_sum_reduction_kernels.push_back(   clCreateKernel(field_sumred_reset_prog, "reduction_sum_last_ocl_kernel", &err));
                ie_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_last_ocl_kernel", &err));
                ke_sum_reduction_kernels.push_back(     clCreateKernel(field_sumred_reset_prog, "reduction_sum_last_ocl_kernel", &err));
                press_sum_reduction_kernels.push_back(  clCreateKernel(field_sumred_reset_prog, "reduction_sum_last_ocl_kernel", &err));

                if (i==1) {
                    //if on first level then set input to equal source buffer
                    err = clSetKernelArg(min_reduction_kernels[i-1],       0, sizeof(cl_mem), &CloverCL::dt_min_val_array_buffer_c); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    0, sizeof(cl_mem), &CloverCL::vol_tmp_buffer_c); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   0, sizeof(cl_mem), &CloverCL::mass_tmp_buffer_c); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ie_tmp_buffer_c); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ke_tmp_buffer_c); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  0, sizeof(cl_mem), &CloverCL::press_tmp_buffer_c); 

                    //min_reduction_kernels[i-1].setArg(0, CloverCL::dt_min_val_array_buffer);
                    //vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::vol_tmp_buffer);
                    //mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::mass_tmp_buffer);
                    //ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ie_tmp_buffer);
                    //ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ke_tmp_buffer);
                    //press_sum_reduction_kernels[i-1].setArg(0, CloverCL::press_tmp_buffer);
                }
                else {
                    err = clSetKernelArg(min_reduction_kernels[i-1],       0, sizeof(cl_mem), &CloverCL::min_interBuffers[i-2]); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    0, sizeof(cl_mem), &CloverCL::vol_interBuffers[i-2]); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   0, sizeof(cl_mem), &CloverCL::mass_interBuffers[i-2]); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ie_interBuffers[i-2]); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     0, sizeof(cl_mem), &CloverCL::ke_interBuffers[i-2]); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  0, sizeof(cl_mem), &CloverCL::press_interBuffers[i-2]); 

                    //min_reduction_kernels[i-1].setArg(0, CloverCL::min_interBuffers[i-2]);
                    //vol_sum_reduction_kernels[i-1].setArg(  0, CloverCL::vol_interBuffers[i-2]);
                    //mass_sum_reduction_kernels[i-1].setArg( 0, CloverCL::mass_interBuffers[i-2]);
                    //ie_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ie_interBuffers[i-2]);
                    //ke_sum_reduction_kernels[i-1].setArg(   0, CloverCL::ke_interBuffers[i-2]);
                    //press_sum_reduction_kernels[i-1].setArg(0, CloverCL::press_interBuffers[i-2]);
                }

                err = clSetKernelArg(min_reduction_kernels[i-1],       1, CloverCL::min_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    1, CloverCL::vol_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   1, CloverCL::mass_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     1, CloverCL::ie_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     1, CloverCL::ke_local_memory_objects[i-1], NULL); 
                err = clSetKernelArg(press_sum_reduction_kernels[i-1],  1, CloverCL::press_local_memory_objects[i-1], NULL); 

                //min_reduction_kernels[i-1].setArg(1, CloverCL::min_local_memory_objects[i-1]);
                //vol_sum_reduction_kernels[i-1].setArg(1,   CloverCL::vol_local_memory_objects[i-1]);
                //mass_sum_reduction_kernels[i-1].setArg(1,  CloverCL::mass_local_memory_objects[i-1]);
                //ie_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ie_local_memory_objects[i-1]);
                //ke_sum_reduction_kernels[i-1].setArg(1,    CloverCL::ke_local_memory_objects[i-1]);
                //press_sum_reduction_kernels[i-1].setArg(1, CloverCL::press_local_memory_objects[i-1]);

                if (i==CloverCL::number_of_red_levels) {
                    //if last level of reduction set output to be output buffer
                    err = clSetKernelArg(min_reduction_kernels[i-1],       2, sizeof(cl_mem), &CloverCL::dt_min_val_buffer_c); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    2, sizeof(cl_mem), &CloverCL::vol_sum_val_buffer_c); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   2, sizeof(cl_mem), &CloverCL::mass_sum_val_buffer_c); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ie_sum_val_buffer_c); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ke_sum_val_buffer_c); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  2, sizeof(cl_mem), &CloverCL::press_sum_val_buffer_c); 

                    //min_reduction_kernels[i-1].setArg(2, CloverCL::dt_min_val_buffer);
                    //vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_sum_val_buffer);
                    //mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_sum_val_buffer);
                    //ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_sum_val_buffer);
                    //ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_sum_val_buffer);
                    //press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_sum_val_buffer);
                }
                else {
                    err = clSetKernelArg(min_reduction_kernels[i-1],       2, sizeof(cl_mem), &CloverCL::min_interBuffers[i-1]); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    2, sizeof(cl_mem), &CloverCL::vol_interBuffers[i-1]); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   2, sizeof(cl_mem), &CloverCL::mass_interBuffers[i-1]); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ie_interBuffers[i-1]); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     2, sizeof(cl_mem), &CloverCL::ke_interBuffers[i-1]); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  2, sizeof(cl_mem), &CloverCL::press_interBuffers[i-1]); 

                    //min_reduction_kernels[i-1].setArg(2, CloverCL::min_interBuffers[i-1]);
                    //vol_sum_reduction_kernels[i-1].setArg(2,   CloverCL::vol_interBuffers[i-1]);
                    //mass_sum_reduction_kernels[i-1].setArg(2,  CloverCL::mass_interBuffers[i-1]);
                    //ie_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ie_interBuffers[i-1]);
                    //ke_sum_reduction_kernels[i-1].setArg(2,    CloverCL::ke_interBuffers[i-1]);
                    //press_sum_reduction_kernels[i-1].setArg(2, CloverCL::press_interBuffers[i-1]);
                }

                err = clSetKernelArg(min_reduction_kernels[i-1],       3, sizeof(int), &CloverCL::size_limits[i-1]); 
                err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    3, sizeof(int), &CloverCL::size_limits[i-1]); 
                err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   3, sizeof(int), &CloverCL::size_limits[i-1]); 
                err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     3, sizeof(int), &CloverCL::size_limits[i-1]); 
                err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     3, sizeof(int), &CloverCL::size_limits[i-1]); 
                err = clSetKernelArg(press_sum_reduction_kernels[i-1],  3, sizeof(int), &CloverCL::size_limits[i-1]); 

                //min_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                //vol_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                //mass_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                //ie_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                //ke_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);
                //press_sum_reduction_kernels[i-1].setArg(3, CloverCL::size_limits[i-1]);

                if (CloverCL::input_even[i-1]==true) {
                    int even = 1;
                    err = clSetKernelArg(min_reduction_kernels[i-1],       4, sizeof(int), &even); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    4, sizeof(int), &even); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   4, sizeof(int), &even); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     4, sizeof(int), &even); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     4, sizeof(int), &even); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  4, sizeof(int), &even); 

                    //min_reduction_kernels[i-1].setArg(4, 1);
                    //vol_sum_reduction_kernels[i-1].setArg(4, 1);
                    //mass_sum_reduction_kernels[i-1].setArg(4, 1);
                    //ie_sum_reduction_kernels[i-1].setArg(4, 1);
                    //ke_sum_reduction_kernels[i-1].setArg(4, 1);
                    //press_sum_reduction_kernels[i-1].setArg(4, 1);
                }
                else {
                    int even = 0;
                    err = clSetKernelArg(min_reduction_kernels[i-1],       4, sizeof(int), &even); 
                    err = clSetKernelArg(vol_sum_reduction_kernels[i-1],    4, sizeof(int), &even); 
                    err = clSetKernelArg(mass_sum_reduction_kernels[i-1],   4, sizeof(int), &even); 
                    err = clSetKernelArg(ie_sum_reduction_kernels[i-1],     4, sizeof(int), &even); 
                    err = clSetKernelArg(ke_sum_reduction_kernels[i-1],     4, sizeof(int), &even); 
                    err = clSetKernelArg(press_sum_reduction_kernels[i-1],  4, sizeof(int), &even); 

                    //min_reduction_kernels[i-1].setArg(4, 0);
                    //vol_sum_reduction_kernels[i-1].setArg(4, 0);
                    //mass_sum_reduction_kernels[i-1].setArg(4, 0);
                    //ie_sum_reduction_kernels[i-1].setArg(4, 0);
                    //ke_sum_reduction_kernels[i-1].setArg(4, 0);
                    //press_sum_reduction_kernels[i-1].setArg(4, 0);
                }
            }
        }

    }
    else {
        std::cout << "ERROR in CloverCL build reduction structure objectes method: device type is unsupported" 
                  << std::endl;
    }
}

void CloverCL::calculateReductionStructure(int xmax, int ymax) {

    int x_rnd = ((xmax+2) / fixed_wg_min_size_large_dim ) * fixed_wg_min_size_large_dim;

    if ((x_rnd != xmax+2))
        x_rnd = x_rnd + fixed_wg_min_size_large_dim;

    int y_rnd = ((ymax+2) / fixed_wg_min_size_small_dim ) * fixed_wg_min_size_small_dim;

    if ((y_rnd != ymax+2))
        y_rnd = y_rnd + fixed_wg_min_size_small_dim;

    int num_elements = (x_rnd / fixed_wg_min_size_large_dim) * (y_rnd / fixed_wg_min_size_small_dim);

    num_workitems_tolaunch.clear();
    num_workitems_per_wg.clear();
    local_mem_size.clear();
    size_limits.clear();
    buffer_sizes.clear();
    input_even.clear();
    num_elements_per_wi.clear();

    if (device_type == CL_DEVICE_TYPE_CPU) {

        if ( num_elements < device_procs*2) { 
            //just launch one level of reduction as not enough elements

            number_of_red_levels = 1;

            num_workitems_tolaunch.push_back(1);
            num_workitems_per_wg.push_back(1);
            num_elements_per_wi.push_back(num_elements);
        }
        else {

            number_of_red_levels = 2;
            
            num_workitems_tolaunch.push_back(device_procs);
            num_workitems_per_wg.push_back(1);
            num_elements_per_wi.push_back(num_elements);

            num_workitems_tolaunch.push_back(1);
            num_workitems_per_wg.push_back(1);
            num_elements_per_wi.push_back(device_procs);
        }

#ifdef OCL_VERBOSE
        std::cout << "number_of_red_levels after loop: " << number_of_red_levels << std::endl;
        std::cout << "number of workitems to launch vector size: " << num_workitems_tolaunch.size() << std::endl;
        std::cout << "number of workitems per wg vector size: " << num_workitems_per_wg.size() << std::endl;
        std::cout << "number of local_mem_size vector size: " << local_mem_size.size() << std::endl;
        std::cout << "number of size_limits vector size: " << size_limits.size() << std::endl;
        std::cout << "number of buffer_sizes vector size: " << buffer_sizes.size() << std::endl;
        std::cout << "number of input_even vector size: " << input_even.size() << std::endl;
        std::cout << "number of num_elements_per_wi vector size: " << num_elements_per_wi.size() << std::endl;

        for (int i=0; i<number_of_red_levels; i++) {
            std::cout << "Red level:            " << i+1 << std::endl;
            std::cout << "Work items to launch: " << num_workitems_tolaunch[i] << std::endl;
            std::cout << "Work items per wg:    " << num_workitems_per_wg[i] << std::endl;
            //std::cout << "Size limit:           " << size_limits[i] << std::endl;
            std::cout << "Num Element per wi:   " << num_elements_per_wi[i] << std::endl;
            std::cout << std::endl;
        }
#endif

    }
    else if ( (device_type == CL_DEVICE_TYPE_GPU) || (device_type == CL_DEVICE_TYPE_ACCELERATOR) ){

        int wg_ingest_value, temp_wg_ingest_size, remaining_wis;
        int normal_wg_size;
        number_of_red_levels = 0;

        normal_wg_size = REDUCTION_WG_SIZE;
        wg_ingest_value = 2*normal_wg_size;

        if ( fmod(log2(normal_wg_size),1)!=0  ) {
            //reduction workgroup size selected is not a power of 2
            std::cerr << "Error: reduction local workgroup size is NOT a power of 2" << std::endl; 
            exit(EXIT_FAILURE);
        }
        if ( normal_wg_size > max_reduction_wg_size ) {
            std::cerr << "Error: reduction local workgroup size is greater than device maximum. Normal WGsize: " 
                      << normal_wg_size << " Device maxWG: " << device_max_wg_size << std::endl; 
            exit(EXIT_FAILURE);
        }

        //add initial starting buffer to buffers vector
        buffer_sizes.push_back(num_elements);

        do {
            number_of_red_levels++;

            if (buffer_sizes.back() <= wg_ingest_value) {
            //only one workgroup required 

                if (buffer_sizes.back() == wg_ingest_value) {
                    num_workitems_tolaunch.push_back(normal_wg_size);
            	    num_workitems_per_wg.push_back(normal_wg_size);
            	    local_mem_size.push_back(normal_wg_size);
            	    size_limits.push_back(-1);
            	    input_even.push_back(true);
                }
                else {
            	    temp_wg_ingest_size = wg_ingest_value / 2;

            	    while( (temp_wg_ingest_size > buffer_sizes.back()) && (temp_wg_ingest_size >= prefer_wg_multiple*2 )  ) {
                        wg_ingest_value = temp_wg_ingest_size; 
            	        temp_wg_ingest_size = temp_wg_ingest_size / 2; 
            	    }
            	    normal_wg_size = wg_ingest_value / 2;

            	    num_workitems_tolaunch.push_back(normal_wg_size);
            	    num_workitems_per_wg.push_back(normal_wg_size);
            	    local_mem_size.push_back(normal_wg_size);

                    if (buffer_sizes.back() == wg_ingest_value) {
            	    //last level is a multiple of 2 there don't need a limit 
            	    size_limits.push_back(-1);
            	    input_even.push_back(true);
            	    }
            	    else if (buffer_sizes.back() % 2 == 0) {
            	        //last level input is even amount
            	        size_limits.push_back(buffer_sizes.back() / 2);
            	        input_even.push_back(true);
            	    }
            	    else {
            	        //last level input is odd amount
            	        size_limits.push_back(buffer_sizes.back() / 2);
            	        input_even.push_back(false);
            	    }
                }

                buffer_sizes.push_back(1);

            }
            else if ( buffer_sizes.back() % wg_ingest_value==0 ) {
                num_workitems_tolaunch.push_back(buffer_sizes.back() / wg_ingest_value * normal_wg_size);
                num_workitems_per_wg.push_back(normal_wg_size);
                local_mem_size.push_back(normal_wg_size);
                size_limits.push_back(-1);
                buffer_sizes.push_back(buffer_sizes.back() / wg_ingest_value);
                input_even.push_back(true);
            }
            else {
                //basic strategy is currently to use the maximum possible size of workgroup and then 
                //limit the number reduced in the final workgroup to allow for arbitrary sizes
                //this approach may well need changing but will do for now 
                //eg may be better to balance things across the GPU eg 1WG / MP

                num_workitems_tolaunch.push_back( (buffer_sizes.back() / wg_ingest_value + 1) * normal_wg_size);
                num_workitems_per_wg.push_back(normal_wg_size);
                local_mem_size.push_back(normal_wg_size);

                remaining_wis = buffer_sizes.back() % wg_ingest_value;
                size_limits.push_back(remaining_wis / 2);
                buffer_sizes.push_back(buffer_sizes.back() / wg_ingest_value + 1);
                if (remaining_wis % 2 == 0) {
                    input_even.push_back(true);
                }
                else {
                    input_even.push_back(false);
                }
            }

        } while(buffer_sizes.back() != 1);

#ifdef OCL_VERBOSE
        std::cout << "number_of_red_levels after loop: " << number_of_red_levels << std::endl;
        std::cout << "number of workitems to launch vector size: " << num_workitems_tolaunch.size() << std::endl;
        std::cout << "number of workitems per wg vector size: " << num_workitems_per_wg.size() << std::endl;
        std::cout << "number of local_mem_size vector size: " << local_mem_size.size() << std::endl;
        std::cout << "number of size_limits vector size: " << size_limits.size() << std::endl;
        std::cout << "number of buffer_sizes vector size: " << buffer_sizes.size() << std::endl;
        std::cout << "number of input_even vector size: " << input_even.size() << std::endl;

        for (int i=0; i<number_of_red_levels; i++) {
            std::cout << "Red level:            " << i+1 << std::endl;
            std::cout << "buffer input size:    " << buffer_sizes[i] << std::endl;
            std::cout << "Work items to launch: " << num_workitems_tolaunch[i] << std::endl;
            std::cout << "Work items per wg:    " << num_workitems_per_wg[i] << std::endl;
            std::cout << "Local memory size:    " << local_mem_size[i] << std::endl;
            std::cout << "Size limit:           " << size_limits[i] << std::endl;
            std::cout << "Input Even:           " << std::boolalpha << input_even[i] << std::endl;
            std::cout << "Buffer output size:   " << buffer_sizes[i+1] << std::endl;
            std::cout << std::endl;
        }
#endif
    }
    else {
        std::cout << "ERROR in CloverCL reduction structure: device type is unsupported" << std::endl;
    }

}


void CloverCL::allocateLocalMemoryObjects() {

    if (device_type == CL_DEVICE_TYPE_CPU) {
#ifdef OCL_VERBOSE
        std::cout << "No local memory objects to create as device type is CPU" << std::endl;
#endif
    }
    //else if (device_type == CL_DEVICE_TYPE_ACCELERATOR) {
    //#ifdef OCL_VERBOSE
    //    std::cout << "No local memory objects to create as device type is ACCELERATOR" << std::endl;
    //#endif
    //}
    else if ( (device_type == CL_DEVICE_TYPE_GPU) || (device_type == CL_DEVICE_TYPE_ACCELERATOR) ) {
        for (int i=0; i<number_of_red_levels; i++) {
            
            min_local_memory_objects.push_back(   local_mem_size[i]*sizeof(cl_double)  );
            vol_local_memory_objects.push_back(   local_mem_size[i]*sizeof(cl_double)  );
            mass_local_memory_objects.push_back(  local_mem_size[i]*sizeof(cl_double)  );
            ie_local_memory_objects.push_back(    local_mem_size[i]*sizeof(cl_double)  );
            ke_local_memory_objects.push_back(    local_mem_size[i]*sizeof(cl_double)  );
            press_local_memory_objects.push_back( local_mem_size[i]*sizeof(cl_double)  );

            //min_local_memory_objects.push_back(   cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            //vol_local_memory_objects.push_back(   cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            //mass_local_memory_objects.push_back(  cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            //ie_local_memory_objects.push_back(    cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            //ke_local_memory_objects.push_back(    cl::Local(local_mem_size[i]*sizeof(cl_double))  );
            //press_local_memory_objects.push_back( cl::Local(local_mem_size[i]*sizeof(cl_double))  );
        }

#ifdef OCL_VERBOSE
        std::cout << "min local memory objects vector size: "     << min_local_memory_objects.size() << std::endl;
        std::cout << "vol local memory objects vector size: "     << vol_local_memory_objects.size() << std::endl;
        std::cout << "mass local memory objects vector size: "    << mass_local_memory_objects.size() << std::endl;
        std::cout << "ie local memory objects vector size: "      << ie_local_memory_objects.size() << std::endl;
        std::cout << "ke local memory objects vector size: "      << ke_local_memory_objects.size() << std::endl;
        std::cout << "press local memory objects vector size: "   << press_local_memory_objects.size() << std::endl;

        for (int i=0; i<number_of_red_levels; i++) {
           std::cout << "reduction level " << i+1 << "min local object size: "   << min_local_memory_objects[i]/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "vol local object size: "   << vol_local_memory_objects[i]/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "mass local object size: "  << mass_local_memory_objects[i]/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "ie local object size: "    << ie_local_memory_objects[i]/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "ke local object size: "    << ke_local_memory_objects[i]/sizeof(double) << std::endl;
           std::cout << "reduction level " << i+1 << "press local object size: " << press_local_memory_objects[i]/sizeof(double) << std::endl;
        }
#endif
    }
    else {
        std::cout << "ERROR in CloverCL.C local memory object creation: device type not supported " << std::endl;
    }

}

void CloverCL::allocateReductionInterBuffers() {

    cl_int err;

    if (device_type == CL_DEVICE_TYPE_CPU) {

        if ( number_of_red_levels == 1 ) { 
#ifdef OCL_VERBOSE
            std::cout << "No intermediate reduction buffers required for CPU reduction as num_elements too small"  << std::endl;
#endif
        }
        else {
            cpu_min_red_buffer_c = clCreateBuffer(context_c, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_vol_red_buffer_c = clCreateBuffer(context_c, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_mass_red_buffer_c = clCreateBuffer(context_c, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_ie_red_buffer_c = clCreateBuffer(context_c, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_ke_red_buffer_c = clCreateBuffer(context_c, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);
            cpu_press_red_buffer_c = clCreateBuffer(context_c, CL_MEM_READ_WRITE, device_procs*sizeof(double), NULL, &err);

#ifdef OCL_VERBOSE
            std::cout << "Intermediate reduction buffers on CPU created with size: " << device_procs << std::endl;
#endif
        }

    }
    else if ( (device_type == CL_DEVICE_TYPE_GPU) || (device_type==CL_DEVICE_TYPE_ACCELERATOR) ){

        for (int i=1; i<=number_of_red_levels-1; i++) {

            min_interBuffers.push_back(  clCreateBuffer( context_c, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            vol_interBuffers.push_back(  clCreateBuffer( context_c, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            mass_interBuffers.push_back( clCreateBuffer( context_c, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            ie_interBuffers.push_back(   clCreateBuffer( context_c, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            ke_interBuffers.push_back(   clCreateBuffer( context_c, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));
            press_interBuffers.push_back(clCreateBuffer( context_c, CL_MEM_READ_WRITE, buffer_sizes[i]*sizeof(double), NULL, &err));

        }

#ifdef OCL_VERBOSE
        size_t size;
        std::cout << "min inter buffers vector size: "   << min_interBuffers.size() << std::endl;
        std::cout << "vol inter buffers vector size: "   << vol_interBuffers.size() << std::endl;
        std::cout << "mass inter buffers vector size: "  << mass_interBuffers.size() << std::endl;
        std::cout << "ie inter buffers vector size: "    << ie_interBuffers.size() << std::endl;
        std::cout << "ke inter buffers vector size: "    << ke_interBuffers.size() << std::endl;
        std::cout << "press inter buffers vector size: " << press_interBuffers.size() << std::endl;

        for (int i=0; i<=number_of_red_levels-2; i++) {
            err = clGetMemObjectInfo(min_interBuffers[i], CL_MEM_SIZE, sizeof(size_t), &size, NULL); 
            std::cout << "min inter buffers level: "   << i << " buffer elements: " << size/sizeof(double) << std::endl;

            err = clGetMemObjectInfo(vol_interBuffers[i], CL_MEM_SIZE, sizeof(size_t), &size, NULL); 
            std::cout << "vol inter buffers level: "   << i << " buffer elements: " << size/sizeof(double) << std::endl;

            err = clGetMemObjectInfo(mass_interBuffers[i], CL_MEM_SIZE, sizeof(size_t), &size, NULL); 
            std::cout << "mass inter buffers level: "  << i << " buffer elements: " << size/sizeof(double) << std::endl;

            err = clGetMemObjectInfo(ie_interBuffers[i], CL_MEM_SIZE, sizeof(size_t), &size, NULL); 
            std::cout << "ie inter buffers level: "    << i << " buffer elements: " << size/sizeof(double) << std::endl;

            err = clGetMemObjectInfo(ke_interBuffers[i], CL_MEM_SIZE, sizeof(size_t), &size, NULL); 
            std::cout << "ke inter buffers level: "    << i << " buffer elements: " << size/sizeof(double) << std::endl;

            err = clGetMemObjectInfo(press_interBuffers[i], CL_MEM_SIZE, sizeof(size_t), &size, NULL); 
            std::cout << "press inter buffers level: " << i << " buffer elements: " << size/sizeof(double) << std::endl;
        }
#endif
    } else {
        std::cout << "ERROR in CloverCL.C allocate inter buffers: device type not supported" << std::endl;
    }

}

void CloverCL::printDeviceInformation() {
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
  
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
  
    for (i = 0; i < platformCount; i++) {
  
        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
  
        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {
  
            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);
  
            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);
  
            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);
  
            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);
  
            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);
    
            //print max number of work items 
	        int mwgs;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwgs), &mwgs, NULL);
            printf(" %d.%d Max num work items: %d\n", j+1, 5, mwgs);

            //print global memory size 
            cl_ulong global_mem;
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &global_mem, NULL);
            printf(" %d.%d Global memory size: %lu\n", j+1, 6, global_mem);

            //print clock speed
            cl_uint clock_speed;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    sizeof(cl_uint), &clock_speed, NULL);
            printf(" %d.%d Clock speed: %u\n", j+1, 7, clock_speed);

            //print device type
            cl_device_type typeofdevice;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &typeofdevice, NULL);
            switch (typeofdevice) {
                case CL_DEVICE_TYPE_CPU         : std::cout << "Device is a CPU" << std::endl; break;
                case CL_DEVICE_TYPE_GPU         : std::cout << "Device is a GPU" << std::endl; break;
                case CL_DEVICE_TYPE_ACCELERATOR : std::cout << "Device is a ACCEL" << std::endl; break;
                case CL_DEVICE_TYPE_DEFAULT     : std::cout << "Device is a Default" << std::endl; break;
                default: std::cout << "Device is something else" << std::endl; 
            }
        }
    }
}
      
void CloverCL::initPlatform(std::string name) 
{
    cl_int err;
    cl_uint numPlatforms;
    int platforms_limit = 5;
    size_t platformVendor_length = 60 * sizeof(char);
    size_t platformVendor_retsize;
    char* platformVendor = new char[platformVendor_length];
    std::string platformVendor_str; 

    /*
     * Lowercase the name provided
     */
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    cl_platform_id* platforms = new cl_platform_id[ platforms_limit * sizeof(cl_platform_id)];
    
    err = clGetPlatformIDs(platforms_limit, platforms, &numPlatforms);

    if ( err != CL_SUCCESS || numPlatforms <= 0) {
        reportError(err, "Querying Platforms");
    }

    
    for (int i=0; i<numPlatforms; i++) {
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, platformVendor_length, platformVendor, &platformVendor_retsize);

        if (err != CL_SUCCESS) {
            reportError(err, "Getting Platform Info");
        }

        if (platformVendor_retsize > platformVendor_length) {
            reportError(err, "Returned platform vendors is longer than buffer");
        }

        platformVendor_str = std::string(platformVendor);

#ifdef OCL_VERBOSE
        std::cout << "Platform vendor " << i << " found: " << platformVendor_str << std::endl; 
#endif
        std::transform(platformVendor_str.begin(), platformVendor_str.end(), platformVendor_str.begin(), ::tolower);
        if(platformVendor_str.find(name) != std::string::npos) {
            platform_c = platforms[i];
            break;
        }
        
    }

#ifdef OCL_VERBOSE
    clGetPlatformInfo(platform_c, CL_PLATFORM_VENDOR, platformVendor_length, platformVendor, &platformVendor_retsize);
    platformVendor_str = std::string(platformVendor);
    std::cout << "Set platform to " << platformVendor_str << std::endl;
#endif
    
    delete [] platforms;
    delete [] platformVendor;
}

void CloverCL::initContext(std::string preferred_type)
{
    cl_int err;

    /*
     * Lowercase the type provided
     */
    std::transform(preferred_type.begin(), preferred_type.end(), preferred_type.begin(), ::tolower);

    cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_c, 0 };

    device_type = CL_DEVICE_TYPE_DEFAULT;

    if (preferred_type == "gpu") {
        device_type = CL_DEVICE_TYPE_GPU;
    } else if(preferred_type == "cpu") {
        device_type = CL_DEVICE_TYPE_CPU;
    } else if(preferred_type == "phi") {
        device_type = CL_DEVICE_TYPE_ACCELERATOR;
    } else if (preferred_type == "fpga") {
        device_type = CL_DEVICE_TYPE_ACCELERATOR;
    }

#ifdef OCL_VERBOSE
    if (preferred_type == "gpu") {
        std::cout << "Device Type selected: GPU" << std::endl;
    } else if(preferred_type == "cpu") {
        std::cout << "Device Type selected: CPU" << std::endl;
    } else if(preferred_type == "phi") {
        std::cout << "Device Type selected: PHI Accelerator" << std::endl;
    } else if(preferred_type == "fpga") {
        std::cout << "Device Type selected: Altera FPGA" << std::endl;
    } else {
        std::cout << "Device Type selected: something else" << std::endl; 
    }
#endif

    /*
     * Get the device context.
     */
    context_c = clCreateContextFromType(cprops, device_type, NULL, NULL, &err);

    if (err != CL_SUCCESS) {
        reportError(err, "Error creating context");
    }
}

void CloverCL::initDevice(int id)
{
    cl_int err;
    size_t devices_list_size;

    err = clGetContextInfo(context_c, CL_CONTEXT_DEVICES, 0, NULL, &devices_list_size);

    if (err != CL_SUCCESS) {
        reportError(err, "Failed to get devices list size from context");
    }
    if (devices_list_size <= 0) {
        reportError(err, "Devices list size is <= 0");
    }

    devices_list = new cl_device_id[devices_list_size / sizeof(cl_device_id)]; 

    err = clGetContextInfo(context_c, CL_CONTEXT_DEVICES, devices_list_size, devices_list, NULL);

    if (err != CL_SUCCESS) {
        reportError(err, "Failed to create device list");
    }

    device_c = devices_list[id]; 

#ifdef OCL_VERBOSE
    size_t device_name_size;
    char *device_name; 
    clGetDeviceInfo(device_c, CL_DEVICE_NAME, 0, NULL, &device_name_size); 

    device_name = new char[device_name_size];

    clGetDeviceInfo(device_c, CL_DEVICE_NAME, device_name_size, device_name, NULL);

    std::string device_name_str = std::string(device_name);

    std::cout << "Name of the selected device: " << device_name_str << std::endl; 

    delete [] device_name;
#endif
}

void CloverCL::initCommandQueue()
{
    cl_int err;

    queue_c = clCreateCommandQueue(context_c, device_c, CL_QUEUE_PROFILING_ENABLE, &err);

    if (err != CL_SUCCESS) {
        reportError(err, "Creating in order queue");
    }

    outoforder_queue_c = clCreateCommandQueue(context_c, device_c, CL_QUEUE_PROFILING_ENABLE, &err);

    if (err != CL_SUCCESS) {
        reportError(err, "Creating out of order queue");
    }

    //queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    //outoforder_queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|CL_QUEUE_PROFILING_ENABLE, &err);
}

void CloverCL::createBuffers(int x_max, int y_max, int num_states)
{
    cl_int err;

    density0_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    density1_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    energy0_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    energy1_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    pressure_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    soundspeed_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    celldx_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_ONLY, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    celldy_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_ONLY, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    cellx_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*sizeof(double), NULL, &err);

    celly_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (y_max+4)*sizeof(double), NULL, &err);

    viscosity_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);


    //xvel0_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    xvel0_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+8)*(y_max+5)*sizeof(double), NULL, &err);

    //yvel0_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    yvel0_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+8)*(y_max+5)*sizeof(double), NULL, &err);

    //xvel1_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    xvel1_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+8)*(y_max+5)*sizeof(double), NULL, &err);

    //yvel1_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);
    yvel1_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+8)*(y_max+5)*sizeof(double), NULL, &err);

    //xarea_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+4)*sizeof(double), NULL, &err);
    xarea_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+8)*(y_max+4)*sizeof(double), NULL, &err);


    yarea_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_ONLY, (x_max+4)*(y_max+5)*sizeof(double), NULL, &err);



    vol_flux_x_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+4)*sizeof(double), NULL, &err);

    vol_flux_y_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+5)*sizeof(double), NULL, &err);

    mass_flux_x_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+4)*sizeof(double), NULL, &err);

    mass_flux_y_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+5)*sizeof(double), NULL, &err);

    stepbymass_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    volume_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+4)*sizeof(double), NULL, &err);

    vertexdx_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*sizeof(double), NULL, &err);

    vertexx_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*sizeof(double), NULL, &err);

    vertexdy_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (y_max+5)*sizeof(double), NULL, &err);

    vertexy_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (y_max+5)*sizeof(double), NULL, &err);

    mass_flux_x_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+4)*sizeof(double), NULL, &err);

    mass_flux_y_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+4)*(y_max+5)*sizeof(double), NULL, &err);

    node_flux_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    node_mass_post_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    node_mass_pre_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    post_vol_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    pre_vol_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    pre_mass_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    post_mass_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    advec_vel_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    mom_flux_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    advec_vol_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    post_ener_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    ener_flux_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*(y_max+5)*sizeof(double), NULL, &err);

    dt_min_val_array_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max)*(y_max)*sizeof(double), NULL, &err);

    dt_min_val_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);

    vol_sum_val_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);

    mass_sum_val_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);

    ie_sum_val_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);

    ke_sum_val_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);

    press_sum_val_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, sizeof(double), NULL, &err);

    state_density_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_energy_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_xvel_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_yvel_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_xmin_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_xmax_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_ymin_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_ymax_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_radius_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(double), NULL, &err); 

    state_geometry_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, num_states*sizeof(int), NULL, &err); 

    vol_tmp_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max)*(y_max)*sizeof(double), NULL, &err);

    mass_tmp_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max)*(y_max)*sizeof(double), NULL, &err);

    ie_tmp_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max)*(y_max)*sizeof(double), NULL, &err);

    ke_tmp_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max)*(y_max)*sizeof(double), NULL, &err);

    press_tmp_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max)*(y_max)*sizeof(double), NULL, &err);

    top_send_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);

    top_recv_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);

    bottom_send_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);

    bottom_recv_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (x_max+5)*2*sizeof(double), NULL, &err);

    left_send_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);

    left_recv_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);

    right_send_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);

    right_recv_buffer_c = clCreateBuffer( context_c, CL_MEM_READ_WRITE, (y_max+5)*2*sizeof(double), NULL, &err);

}


void CloverCL::initialiseKernelArgs(int x_min, int x_max, int y_min, int y_max,
                                    double g_small, double g_big, double dtmin,
                                    double dtc_safe, double dtu_safe, 
                                    double dtv_safe, double dtdiv_safe)
{
    cl_int err; 

    //err = clSetKernelArg(viscosity_knl_c, 0, sizeof(cl_mem), &celldx_buffer_c);
    //err = clSetKernelArg(viscosity_knl_c, 1, sizeof(cl_mem), &celldy_buffer_c);
    //err = clSetKernelArg(viscosity_knl_c, 2, sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(viscosity_knl_c, 3, sizeof(cl_mem), &pressure_buffer_c);
    //err = clSetKernelArg(viscosity_knl_c, 4, sizeof(cl_mem), &viscosity_buffer_c);
    //err = clSetKernelArg(viscosity_knl_c, 5, sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(viscosity_knl_c, 6, sizeof(cl_mem), &yvel0_buffer_c);

    err = clSetKernelArg(accelerate_knl_c, 1,  sizeof(cl_mem), &xarea_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 2,  sizeof(cl_mem), &yarea_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 3,  sizeof(cl_mem), &volume_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 4,  sizeof(cl_mem), &density0_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 5,  sizeof(cl_mem), &pressure_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 6,  sizeof(cl_mem), &viscosity_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 7,  sizeof(cl_mem), &xvel0_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 8,  sizeof(cl_mem), &yvel0_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 9,  sizeof(cl_mem), &xvel1_buffer_c);
    err = clSetKernelArg(accelerate_knl_c, 10, sizeof(cl_mem), &yvel1_buffer_c);
    //err = clSetKernelArg(accelerate_knl_c, 11, sizeof(cl_mem), &stepbymass_buffer_c);

    //err = clSetKernelArg(field_summary_knl_c, 0,  sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 1,  sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 2,  sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 3,  sizeof(cl_mem), &pressure_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 4,  sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 5,  sizeof(cl_mem), &yvel0_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 6,  sizeof(cl_mem), &vol_tmp_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 7,  sizeof(cl_mem), &mass_tmp_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 8,  sizeof(cl_mem), &ie_tmp_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 9,  sizeof(cl_mem), &ke_tmp_buffer_c);
    //err = clSetKernelArg(field_summary_knl_c, 10, sizeof(cl_mem), &press_tmp_buffer_c);

    //err = clSetKernelArg(reset_field_knl_c, 0, sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(reset_field_knl_c, 1, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(reset_field_knl_c, 2, sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(reset_field_knl_c, 3, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(reset_field_knl_c, 4, sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(reset_field_knl_c, 5, sizeof(cl_mem), &xvel1_buffer_c);
    //err = clSetKernelArg(reset_field_knl_c, 6, sizeof(cl_mem), &yvel0_buffer_c);
    //err = clSetKernelArg(reset_field_knl_c, 7, sizeof(cl_mem), &yvel1_buffer_c);

    //err = clSetKernelArg(revert_knl_c, 0, sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(revert_knl_c, 1, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(revert_knl_c, 2, sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(revert_knl_c, 3, sizeof(cl_mem), &energy1_buffer_c);

    //err = clSetKernelArg(flux_calc_knl_c, 1, sizeof(cl_mem), &xarea_buffer_c);
    //err = clSetKernelArg(flux_calc_knl_c, 2, sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(flux_calc_knl_c, 3, sizeof(cl_mem), &xvel1_buffer_c);
    //err = clSetKernelArg(flux_calc_knl_c, 4, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(flux_calc_knl_c, 5, sizeof(cl_mem), &yarea_buffer_c);
    //err = clSetKernelArg(flux_calc_knl_c, 6, sizeof(cl_mem), &yvel0_buffer_c);
    //err = clSetKernelArg(flux_calc_knl_c, 7, sizeof(cl_mem), &yvel1_buffer_c);
    //err = clSetKernelArg(flux_calc_knl_c, 8, sizeof(cl_mem), &vol_flux_y_buffer_c);

    //err = clSetKernelArg(initialise_chunk_cell_x_knl_c, 1, sizeof(cl_mem), &vertexx_buffer_c);
    //err = clSetKernelArg(initialise_chunk_cell_x_knl_c, 2, sizeof(cl_mem), &cellx_buffer_c);
    //err = clSetKernelArg(initialise_chunk_cell_x_knl_c, 3, sizeof(cl_mem), &celldx_buffer_c);

    //err = clSetKernelArg(initialise_chunk_cell_y_knl_c,      1, sizeof(cl_mem), &vertexy_buffer_c);
    //err = clSetKernelArg(initialise_chunk_cell_y_knl_c,      2, sizeof(cl_mem), &celly_buffer_c);
    //err = clSetKernelArg(initialise_chunk_cell_y_knl_c,      3, sizeof(cl_mem), &celldy_buffer_c);

    //err = clSetKernelArg(initialise_chunk_vertex_x_knl_c,    2, sizeof(cl_mem), &vertexx_buffer_c);
    //err = clSetKernelArg(initialise_chunk_vertex_x_knl_c,    3, sizeof(cl_mem), &vertexdx_buffer_c);

    //err = clSetKernelArg(initialise_chunk_vertex_y_knl_c,    2, sizeof(cl_mem), &vertexy_buffer_c);
    //err = clSetKernelArg(initialise_chunk_vertex_y_knl_c,    3, sizeof(cl_mem), &vertexdy_buffer_c);

    //err = clSetKernelArg(initialise_chunk_volume_area_knl_c, 2, sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(initialise_chunk_volume_area_knl_c, 3, sizeof(cl_mem), &celldx_buffer_c);
    //err = clSetKernelArg(initialise_chunk_volume_area_knl_c, 4, sizeof(cl_mem), &celldy_buffer_c);
    //err = clSetKernelArg(initialise_chunk_volume_area_knl_c, 5, sizeof(cl_mem), &xarea_buffer_c);
    //err = clSetKernelArg(initialise_chunk_volume_area_knl_c, 6, sizeof(cl_mem), &yarea_buffer_c);

    //err = clSetKernelArg(generate_chunk_knl_c, 0,  sizeof(cl_mem), &vertexx_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 1,  sizeof(cl_mem), &vertexy_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 2,  sizeof(cl_mem), &cellx_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 3,  sizeof(cl_mem), &celly_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 4,  sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 5,  sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 6,  sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 7,  sizeof(cl_mem), &yvel0_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 9,  sizeof(cl_mem), &state_density_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 10, sizeof(cl_mem), &state_energy_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 11, sizeof(cl_mem), &state_xvel_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 12, sizeof(cl_mem), &state_yvel_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 13, sizeof(cl_mem), &state_xmin_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 14, sizeof(cl_mem), &state_xmax_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 15, sizeof(cl_mem), &state_ymin_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 16, sizeof(cl_mem), &state_ymax_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 17, sizeof(cl_mem), &state_radius_buffer_c);
    //err = clSetKernelArg(generate_chunk_knl_c, 18, sizeof(cl_mem), &state_geometry_buffer_c);

    //err = clSetKernelArg(pdv_correct_knl_c, 1,  sizeof(cl_mem), &xarea_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 2,  sizeof(cl_mem), &yarea_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 3,  sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 4,  sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 5,  sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 6,  sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 7,  sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 8,  sizeof(cl_mem), &pressure_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 9,  sizeof(cl_mem), &viscosity_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 10, sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 11, sizeof(cl_mem), &xvel1_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 12, sizeof(cl_mem), &yvel0_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 13, sizeof(cl_mem), &yvel1_buffer_c);
    //err = clSetKernelArg(pdv_correct_knl_c, 14, sizeof(cl_mem), &vol_tmp_buffer_c);

    //err = clSetKernelArg(pdv_predict_knl_c, 1,  sizeof(cl_mem), &xarea_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 2,  sizeof(cl_mem), &yarea_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 3,  sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 4,  sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 5,  sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 6,  sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 7,  sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 8,  sizeof(cl_mem), &pressure_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 9,  sizeof(cl_mem), &viscosity_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 10, sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 11, sizeof(cl_mem), &xvel1_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 12, sizeof(cl_mem), &yvel0_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 13, sizeof(cl_mem), &yvel1_buffer_c);
    //err = clSetKernelArg(pdv_predict_knl_c, 14, sizeof(cl_mem), &vol_tmp_buffer_c);

    //err = clSetKernelArg(dt_calc_knl_c, 0,  sizeof(double), &g_small);
    //err = clSetKernelArg(dt_calc_knl_c, 1,  sizeof(double), &g_big);
    //err = clSetKernelArg(dt_calc_knl_c, 2,  sizeof(double), &dtmin);
    //err = clSetKernelArg(dt_calc_knl_c, 3,  sizeof(double), &dtc_safe);
    //err = clSetKernelArg(dt_calc_knl_c, 4,  sizeof(double), &dtu_safe);
    //err = clSetKernelArg(dt_calc_knl_c, 5,  sizeof(double), &dtv_safe);
    //err = clSetKernelArg(dt_calc_knl_c, 6,  sizeof(double), &dtdiv_safe);
    //err = clSetKernelArg(dt_calc_knl_c, 7,  sizeof(cl_mem), &xarea_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 8,  sizeof(cl_mem), &yarea_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 9,  sizeof(cl_mem), &cellx_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 10, sizeof(cl_mem), &celly_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 11, sizeof(cl_mem), &celldx_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 12, sizeof(cl_mem), &celldy_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 13, sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 14, sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 15, sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 16, sizeof(cl_mem), &pressure_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 17, sizeof(cl_mem), &viscosity_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 18, sizeof(cl_mem), &soundspeed_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 19, sizeof(cl_mem), &xvel0_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 20, sizeof(cl_mem), &yvel0_buffer_c);
    //err = clSetKernelArg(dt_calc_knl_c, 21, sizeof(cl_mem), &dt_min_val_array_buffer_c);

    //err = clSetKernelArg(ideal_gas_predict_knl_c,    0, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(ideal_gas_predict_knl_c,    1, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(ideal_gas_predict_knl_c,    2, sizeof(cl_mem), &pressure_buffer_c);
    //err = clSetKernelArg(ideal_gas_predict_knl_c,    3, sizeof(cl_mem), &soundspeed_buffer_c);

    //err = clSetKernelArg(ideal_gas_NO_predict_knl_c, 0, sizeof(cl_mem), &density0_buffer_c);
    //err = clSetKernelArg(ideal_gas_NO_predict_knl_c, 1, sizeof(cl_mem), &energy0_buffer_c);
    //err = clSetKernelArg(ideal_gas_NO_predict_knl_c, 2, sizeof(cl_mem), &pressure_buffer_c);
    //err = clSetKernelArg(ideal_gas_NO_predict_knl_c, 3, sizeof(cl_mem), &soundspeed_buffer_c);


    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec1_knl_c, 0, sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec1_knl_c, 1, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec1_knl_c, 2, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec1_knl_c, 3, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec1_knl_c, 4, sizeof(cl_mem), &post_vol_buffer_c);

    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec2_knl_c,    0, sizeof(cl_mem), &vertexdx_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec2_knl_c,    1, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec2_knl_c,    2, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec2_knl_c,    3, sizeof(cl_mem), &mass_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec2_knl_c,    4, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec2_knl_c,    5, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec2_knl_c,    6, sizeof(cl_mem), &ener_flux_buffer_c);

    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    0, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    1, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    2, sizeof(cl_mem), &mass_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    3, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    4, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    5, sizeof(cl_mem), &pre_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    6, sizeof(cl_mem), &post_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    7, sizeof(cl_mem), &advec_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    8, sizeof(cl_mem), &post_ener_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep1_sec3_knl_c,    9, sizeof(cl_mem), &ener_flux_buffer_c);


    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec1_knl_c, 0, sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec1_knl_c, 1, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec1_knl_c, 2, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec1_knl_c, 3, sizeof(cl_mem), &post_vol_buffer_c);

    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec2_knl_c,    0, sizeof(cl_mem), &vertexdx_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec2_knl_c,    1, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec2_knl_c,    2, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec2_knl_c,    3, sizeof(cl_mem), &mass_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec2_knl_c,    4, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec2_knl_c,    5, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec2_knl_c,    6, sizeof(cl_mem), &ener_flux_buffer_c);

    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    0, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    1, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    2, sizeof(cl_mem), &mass_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    3, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    4, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    5, sizeof(cl_mem), &pre_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    6, sizeof(cl_mem), &post_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    7, sizeof(cl_mem), &advec_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    8, sizeof(cl_mem), &post_ener_buffer_c);
    //err = clSetKernelArg(advec_cell_xdir_sweep2_sec3_knl_c,    9, sizeof(cl_mem), &ener_flux_buffer_c);


    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec1_knl_c, 0, sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec1_knl_c, 1, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec1_knl_c, 2, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec1_knl_c, 3, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec1_knl_c, 4, sizeof(cl_mem), &post_vol_buffer_c);

    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec2_knl_c,    0, sizeof(cl_mem), &vertexdy_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec2_knl_c,    1, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec2_knl_c,    2, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec2_knl_c,    3, sizeof(cl_mem), &mass_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec2_knl_c,    4, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec2_knl_c,    5, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec2_knl_c,    6, sizeof(cl_mem), &ener_flux_buffer_c);

    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    0, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    1, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    2, sizeof(cl_mem), &mass_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    3, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    4, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    5, sizeof(cl_mem), &pre_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    6, sizeof(cl_mem), &post_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    7, sizeof(cl_mem), &advec_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    8, sizeof(cl_mem), &post_ener_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep1_sec3_knl_c,    9, sizeof(cl_mem), &ener_flux_buffer_c);


    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec1_knl_c, 0, sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec1_knl_c, 1, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec1_knl_c, 2, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec1_knl_c, 3, sizeof(cl_mem), &post_vol_buffer_c);

    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec2_knl_c,    0, sizeof(cl_mem), &vertexdy_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec2_knl_c,    1, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec2_knl_c,    2, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec2_knl_c,    3, sizeof(cl_mem), &mass_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec2_knl_c,    4, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec2_knl_c,    5, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec2_knl_c,    6, sizeof(cl_mem), &ener_flux_buffer_c);

    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    0, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    1, sizeof(cl_mem), &energy1_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    2, sizeof(cl_mem), &mass_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    3, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    4, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    5, sizeof(cl_mem), &pre_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    6, sizeof(cl_mem), &post_mass_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    7, sizeof(cl_mem), &advec_vol_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    8, sizeof(cl_mem), &post_ener_buffer_c);
    //err = clSetKernelArg(advec_cell_ydir_sweep2_sec3_knl_c,    9, sizeof(cl_mem), &ener_flux_buffer_c);


    //err = clSetKernelArg(advec_mom_vol_knl_c,             0, sizeof(cl_mem), &volume_buffer_c);
    //err = clSetKernelArg(advec_mom_vol_knl_c,             1, sizeof(cl_mem), &vol_flux_x_buffer_c);
    //err = clSetKernelArg(advec_mom_vol_knl_c,             2, sizeof(cl_mem), &vol_flux_y_buffer_c);
    //err = clSetKernelArg(advec_mom_vol_knl_c,             3, sizeof(cl_mem), &pre_vol_buffer_c);
    //err = clSetKernelArg(advec_mom_vol_knl_c,             4, sizeof(cl_mem), &post_vol_buffer_c);

    //err = clSetKernelArg(advec_mom_node_x_knl_c,          0, sizeof(cl_mem), &CloverCL::mass_flux_x_buffer_c);
    //err = clSetKernelArg(advec_mom_node_x_knl_c,          1, sizeof(cl_mem), &CloverCL::node_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_node_x_knl_c,          2, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_mom_node_x_knl_c,          3, sizeof(cl_mem), &post_vol_buffer_c);
    //err = clSetKernelArg(advec_mom_node_x_knl_c,          4, sizeof(cl_mem), &node_mass_post_buffer_c);

    //err = clSetKernelArg(advec_mom_node_mass_pre_x_knl_c, 0, sizeof(cl_mem), &node_mass_pre_buffer_c);
    //err = clSetKernelArg(advec_mom_node_mass_pre_x_knl_c, 1, sizeof(cl_mem), &node_mass_post_buffer_c);
    //err = clSetKernelArg(advec_mom_node_mass_pre_x_knl_c, 2, sizeof(cl_mem), &CloverCL::node_flux_buffer_c);

    //err = clSetKernelArg(advec_mom_node_y_knl_c,          0, sizeof(cl_mem), &mass_flux_y_buffer_c);
    //err = clSetKernelArg(advec_mom_node_y_knl_c,          1, sizeof(cl_mem), &node_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_node_y_knl_c,          2, sizeof(cl_mem), &node_mass_post_buffer_c);
    //err = clSetKernelArg(advec_mom_node_y_knl_c,          3, sizeof(cl_mem), &density1_buffer_c);
    //err = clSetKernelArg(advec_mom_node_y_knl_c,          4, sizeof(cl_mem), &post_vol_buffer_c);

    //err = clSetKernelArg(advec_mom_node_mass_pre_y_knl_c, 0, sizeof(cl_mem), &node_mass_pre_buffer_c);
    //err = clSetKernelArg(advec_mom_node_mass_pre_y_knl_c, 1, sizeof(cl_mem), &node_mass_post_buffer_c);
    //err = clSetKernelArg(advec_mom_node_mass_pre_y_knl_c, 2, sizeof(cl_mem), &node_flux_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_x_vec1_knl_c,     0, sizeof(cl_mem), &node_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_x_vec1_knl_c,     1, sizeof(cl_mem), &node_mass_pre_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_x_vec1_knl_c,     3, sizeof(cl_mem), &advec_vel_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_x_vec1_knl_c,     4, sizeof(cl_mem), &mom_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_x_vec1_knl_c,     5, sizeof(cl_mem), &celldx_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_x_vecnot1_knl_c,  0, sizeof(cl_mem), &node_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_x_vecnot1_knl_c,  1, sizeof(cl_mem), &node_mass_pre_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_x_vecnot1_knl_c,  3, sizeof(cl_mem), &advec_vel_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_x_vecnot1_knl_c,  4, sizeof(cl_mem), &mom_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_x_vecnot1_knl_c,  5, sizeof(cl_mem), &celldx_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_y_vec1_knl_c,     0, sizeof(cl_mem), &node_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_y_vec1_knl_c,     1, sizeof(cl_mem), &node_mass_pre_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_y_vec1_knl_c,     3, sizeof(cl_mem), &advec_vel_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_y_vec1_knl_c,     4, sizeof(cl_mem), &mom_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_y_vec1_knl_c,     5, sizeof(cl_mem), &celldy_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_y_vecnot1_knl_c,  0, sizeof(cl_mem), &node_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_y_vecnot1_knl_c,  1, sizeof(cl_mem), &node_mass_pre_buffer_c);

    //err = clSetKernelArg(advec_mom_flux_y_vecnot1_knl_c,  3, sizeof(cl_mem), &advec_vel_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_y_vecnot1_knl_c,  4, sizeof(cl_mem), &mom_flux_buffer_c);
    //err = clSetKernelArg(advec_mom_flux_y_vecnot1_knl_c,  5, sizeof(cl_mem), &celldy_buffer_c);

    //err = clSetKernelArg(advec_mom_vel_x_knl_c,           0, sizeof(cl_mem), &node_mass_post_buffer_c);
    //err = clSetKernelArg(advec_mom_vel_x_knl_c,           1, sizeof(cl_mem), &node_mass_pre_buffer_c);
    //err = clSetKernelArg(advec_mom_vel_x_knl_c,           2, sizeof(cl_mem), &mom_flux_buffer_c);

    //err = clSetKernelArg(advec_mom_vel_y_knl_c,           0, sizeof(cl_mem), &node_mass_post_buffer_c);
    //err = clSetKernelArg(advec_mom_vel_y_knl_c,           1, sizeof(cl_mem), &node_mass_pre_buffer_c);
    //err = clSetKernelArg(advec_mom_vel_y_knl_c,           2, sizeof(cl_mem), &mom_flux_buffer_c);

}


void CloverCL::loadProgram(int xmin, int xmax, int ymin, int ymax)
{

    //build_one_program(xmin, xmax, ymin, ymax, "ideal_gas_knl.aocx", &ideal_gas_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "viscosity_knl.aocx", &viscosity_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "update_halo_knl.aocx", &update_halo_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "ideal_gas_viscosity_updatehalo_knl.aocx", &ideal_vis_uh_prog);

    //build_one_program(xmin, xmax, ymin, ymax, "calc_dt_min_reduction_knl.aocx", &calcdt_minred_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "calc_dt_knl.aocx", &calc_dt_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "min_reduction_knl.aocx", &min_reduction_prog);

    build_one_program(xmin, xmax, ymin, ymax, "accelerate_removeMemOp_double2_1921_3842_1922_1923.aocx", &accelerate_prog);


    //build_one_program(xmin, xmax, ymin, ymax, "accelerate_revert_knl.aocx", &calcdt_minred_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "revert_knl.aocx", &revert_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "accelerate_knl.aocx", &accelerate_prog);

    //build_one_program(xmin, xmax, ymin, ymax, "flux_calc_knl.aocx", &flux_calc_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "pdv_knl.aocx", &pdv_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "pdv_fluxcalc_knl.aocx", &pdv_fluxcalc_prog);

    //build_one_program(xmin, xmax, ymin, ymax, "field_summary_knl.aocx", &field_summary_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "sum_reduction_knl.aocx", &sum_reduction_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "reset_field_knl.aocx", &reset_field_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "fieldsummary_sumreduction_reset_knl.aocx", &field_sumred_reset_prog);


    //build_one_program(xmin, xmax, ymin, ymax, "initialise_chunk_knl.aocx", &initialise_chunk_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "generate_chunk_knl.aocx", &generate_chunk_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "initialise_generate_chunk_knl.aocx", &initialise_generate_chunk_prog);

    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_xdir_sec1.aocx",          &advec_cell_knl_xdir_sec1_sweep1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_xdir_sec1_sweep2.aocx",   &advec_cell_knl_xdir_sec1_sweep2_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_xdir_sec2.aocx",          &advec_cell_knl_xdir_sec2_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_xdir_sec3.aocx",          &advec_cell_knl_xdir_sec3_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_y_sec1_sweep1.aocx",      &advec_cell_knl_y_sec1_sweep1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_y_sec1_sweep2.aocx",      &advec_cell_knl_y_sec1_sweep2_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_y_sec2.aocx",             &advec_cell_knl_y_sec2_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_knl_y_sec3.aocx",             &advec_cell_knl_y_sec3_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_xdir_sweep1_knls.aocx",          &advec_cell_knl_xdir_sweep1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_xdir_sweep2_knls.aocx",          &advec_cell_knl_xdir_sweep2_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_ydir_sweep1_knls.aocx",          &advec_cell_knl_ydir_sweep1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_cell_ydir_sweep2_knls.aocx",          &advec_cell_knl_ydir_sweep2_prog);

    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_vol.aocx",                 &advec_mom_knl_vol_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_node_x.aocx",              &advec_mom_knl_node_x_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_node_y.aocx",              &advec_mom_knl_node_y_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_node_mass_pre_x.aocx",     &advec_mom_knl_node_mass_pre_x_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_node_mass_pre_y.aocx",     &advec_mom_knl_node_mass_pre_y_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_mom_flux_x_notvec1.aocx",  &advec_mom_knl_mom_flux_x_notvec1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_mom_flux_y_notvec1.aocx",  &advec_mom_knl_mom_flux_y_notvec1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_vel_x.aocx",               &advec_mom_knl_vel_x_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_vel_y.aocx",               &advec_mom_knl_vel_y_prog);
    
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_mom_flux_x_vec1.aocx", &advec_mom_knl_mom_flux_x_vec1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "advec_mom_knl_mom_flux_y_vec1.aocx", &advec_mom_knl_mom_flux_y_vec1_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "", &read_comm_buffers_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "", &write_comm_buffers_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "pack_comms_buffers_knl.aocx", &pack_comms_buffers_prog);
    //build_one_program(xmin, xmax, ymin, ymax, "unpack_comms_buffers_knl.aocx", &unpack_comms_buffers_prog);

    std::cout <<"at end of loadprogram" << std::endl;
}

void CloverCL::build_one_program(int xmin, int xmax, int ymin, int ymax, std::string filename, cl_program* prog)
{
    cl_int err;
    size_t lengths[1];
    unsigned char* binaries[1] = {NULL}; 
    cl_int status[1]; 
    cl_int prog_err;
    char buildOptions [350];


#ifdef OCL_VERBOSE
    std::cout << "Executing build one program for kernel: " << filename << std::endl; 
#endif

#define BUILD_LOG() \
    size_t build_log_size; \
    clGetProgramBuildInfo(*prog, device_c, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size); \
    char *buildlog = new char[build_log_size]; \
    clGetProgramBuildInfo(*prog, device_c, CL_PROGRAM_BUILD_LOG, build_log_size, buildlog, NULL); \
    std::string buildlog_str = std::string(buildlog); \
    std::cout << "Build log: " << buildlog_str << std::endl; 
    
    const char* filename_c = filename.c_str(); 

    FILE* fp = fopen(filename_c, "rb"); 
    fseek(fp, 0, SEEK_END);
    lengths[0] = ftell(fp); 
    binaries[0] = (unsigned char*)malloc(sizeof(unsigned char)*lengths[0]);
    rewind(fp);
    fread(binaries[0], lengths[0], 1, fp);
    fclose(fp);

    *prog = clCreateProgramWithBinary(context_c, 1, devices_list, lengths, 
                                              (const unsigned char **)binaries, status, &prog_err);

    if (prog_err != CL_SUCCESS) {
        reportError(prog_err, "Creating program with Binary");
    }

    if (device_type == CL_DEVICE_TYPE_GPU) {

#ifdef OCL_VERBOSE
        std::cout << "Executing GPU specific kernels " << std::endl;
#endif

        sprintf(buildOptions, 
                "-DXMIN=%u -DXMINPLUSONE=%u -DXMAX=%u -DYMIN=%u -DYMINPLUSONE=%u -DYMINPLUSTWO=%u "
                "-DYMAX=%u -DXMAXPLUSONE=%u -DXMAXPLUSTWO=%u -DXMAXPLUSTHREE=%u -DXMAXPLUSFOUR=%u " 
                "-DXMAXPLUSFIVE=%u -DYMAXPLUSONE=%u -DYMAXPLUSTWO=%u -DYMAXPLUSTHREE=%u -DWORKGROUP_SIZE=%u " 
                "-DWORKGROUP_SIZE_DIVTWO=%u -DGPU_REDUCTION -cl-strict-aliasing", 
                xmin, xmin+1, xmax, ymin, ymin+1, ymin+2, ymax, xmax+1, xmax+2, xmax+3, xmax+4, xmax+5, 
                ymax+1, ymax+2, ymax+3, CloverCL::fixed_wg_min_size_large_dim*CloverCL::fixed_wg_min_size_small_dim, 
                (CloverCL::fixed_wg_min_size_large_dim*CloverCL::fixed_wg_min_size_small_dim)/2
               );
    }
    else {

#ifdef OCL_VERBOSE
        std::cout << "Executing CPU specific kernels " << std::endl;
#endif

        sprintf(buildOptions, 
                "-DXMIN=%u -DXMINPLUSONE=%u -DXMAX=%u -DYMIN=%u -DYMINPLUSONE=%u -DYMINPLUSTWO=%u "
                "-DYMAX=%u -DXMAXPLUSONE=%u -DXMAXPLUSTWO=%u -DXMAXPLUSTHREE=%u -DXMAXPLUSFOUR=%u " 
                "-DXMAXPLUSFIVE=%u -DYMAXPLUSONE=%u -DYMAXPLUSTWO=%u -DYMAXPLUSTHREE=%u -DWORKGROUP_SIZE=%u "
                "-DWORKGROUP_SIZE_DIVTWO=%u", 
                xmin, xmin+1, xmax, ymin, ymin+1, ymin+2, ymax, xmax+1, xmax+2, xmax+3, xmax+4, xmax+5, ymax+1, 
                ymax+2, ymax+3, CloverCL::fixed_wg_min_size_large_dim*CloverCL::fixed_wg_min_size_small_dim, 
                (CloverCL::fixed_wg_min_size_large_dim*CloverCL::fixed_wg_min_size_small_dim)/2
               );
    }

    prog_err = clBuildProgram(*prog, 1, CloverCL::devices_list, buildOptions, NULL, NULL); 

    if (prog_err != CL_SUCCESS) {
        BUILD_LOG();
        reportError(prog_err, "Building the program");
    }


#ifdef OCL_VERBOSE
    cl_uint num_devices = -1; 
    clGetProgramInfo(*prog, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);
    size_t ret_buildoptions_size;

    std::cout << "Number of devices associated with program: " << num_devices << std::endl; 


    clGetProgramBuildInfo(*prog, device_c, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &ret_buildoptions_size); 
    
    char *ret_buildoptions = new char[ret_buildoptions_size]; 

    clGetProgramBuildInfo(*prog, device_c, CL_PROGRAM_BUILD_OPTIONS, ret_buildoptions_size, ret_buildoptions, NULL); 

    std::string ret_buildoptions_str = std::string(ret_buildoptions);
    std::cout << "Program built with following build options: " << ret_buildoptions_str << std::endl; 

    BUILD_LOG(); 
#endif 

}

void CloverCL::createKernelObjects() {

    cl_int err; 

    /*
     * Set up the kernels here!
     */
    //ideal_gas_predict_knl_c = clCreateKernel(ideal_vis_uh_prog, "ideal_gas_ocl_kernel", &err);
    //if (err != CL_SUCCESS) {
    //    reportError(err, "ideal_gas_predict_kernel"); 
    //}


    //ideal_gas_NO_predict_knl_c          = clCreateKernel(ideal_vis_uh_prog, "ideal_gas_ocl_kernel", &err);

    //viscosity_knl_c                     = clCreateKernel(ideal_vis_uh_prog, "viscosity_ocl_kernel", &err);

    //flux_calc_knl_c                     = clCreateKernel(pdv_fluxcalc_prog, "flux_calc_ocl_kernel", &err);

    //accelerate_knl_c                    = clCreateKernel(accel_revert_prog, "accelerate_ocl_kernel", &err);
    accelerate_knl_c                    = clCreateKernel(accelerate_prog, "accelerate_ocl_kernel", &err);


    //advec_cell_xdir_sweep1_sec1_knl_c = clCreateKernel(advec_cell_knl_xdir_sweep1_prog, "advec_cell_xdir_section1_sweep1_kernel", &err);
    //advec_cell_xdir_sweep1_sec2_knl_c = clCreateKernel(advec_cell_knl_xdir_sweep1_prog, "advec_cell_xdir_section2_kernel", &err);
    //advec_cell_xdir_sweep1_sec3_knl_c = clCreateKernel(advec_cell_knl_xdir_sweep1_prog, "advec_cell_xdir_section3_kernel", &err);

    //advec_cell_xdir_sweep2_sec1_knl_c = clCreateKernel(advec_cell_knl_xdir_sweep2_prog, "advec_cell_xdir_section1_sweep2_kernel", &err);
    //advec_cell_xdir_sweep2_sec2_knl_c = clCreateKernel(advec_cell_knl_xdir_sweep2_prog, "advec_cell_xdir_section2_kernel", &err);
    //advec_cell_xdir_sweep2_sec3_knl_c = clCreateKernel(advec_cell_knl_xdir_sweep2_prog, "advec_cell_xdir_section3_kernel", &err);

    //advec_cell_ydir_sweep1_sec1_knl_c = clCreateKernel(advec_cell_knl_ydir_sweep1_prog, "advec_cell_ydir_section1_sweep1_kernel", &err);
    //advec_cell_ydir_sweep1_sec2_knl_c = clCreateKernel(advec_cell_knl_ydir_sweep1_prog, "advec_cell_ydir_section2_kernel", &err);
    //advec_cell_ydir_sweep1_sec3_knl_c = clCreateKernel(advec_cell_knl_ydir_sweep1_prog, "advec_cell_ydir_section3_kernel", &err);

    //advec_cell_ydir_sweep2_sec1_knl_c = clCreateKernel(advec_cell_knl_ydir_sweep2_prog, "advec_cell_ydir_section1_sweep2_kernel", &err);
    //advec_cell_ydir_sweep2_sec2_knl_c = clCreateKernel(advec_cell_knl_ydir_sweep2_prog, "advec_cell_ydir_section2_kernel", &err);
    //advec_cell_ydir_sweep2_sec3_knl_c = clCreateKernel(advec_cell_knl_ydir_sweep2_prog, "advec_cell_ydir_section3_kernel", &err);

    //advec_cell_xdir_sec1_s1_knl_c       = clCreateKernel(advec_cell_knl_xdir_sweep1_prog, "advec_cell_xdir_section1_sweep1_kernel", &err);
    //advec_cell_xdir_sec1_s2_knl_c       = clCreateKernel(advec_cell_knl_xdir_sweep1_prog, "advec_cell_xdir_section1_sweep2_kernel", &err);
    //advec_cell_xdir_sec2_knl_c          = clCreateKernel(advec_cell_knl_xdir_sweep1_prog, "advec_cell_xdir_section2_kernel", &err);
    //advec_cell_xdir_sec3_knl_c          = clCreateKernel(advec_cell_knl_xdir_sweep1_prog, "advec_cell_xdir_section3_kernel", &err);
    //advec_cell_ydir_sec1_s1_knl_c       = clCreateKernel(advec_cell_knl_ydir_sweep1_prog, "advec_cell_ydir_section1_sweep1_kernel", &err);
    //advec_cell_ydir_sec1_s2_knl_c       = clCreateKernel(advec_cell_knl_ydir_sweep1_prog, "advec_cell_ydir_section1_sweep2_kernel", &err);
    //advec_cell_ydir_sec2_knl_c          = clCreateKernel(advec_cell_knl_ydir_sweep1_prog, "advec_cell_ydir_section2_kernel", &err);
    //advec_cell_ydir_sec3_knl_c          = clCreateKernel(advec_cell_knl_ydir_sweep1_prog, "advec_cell_ydir_section3_kernel", &err);


    //advec_mom_vol_knl_c                 = clCreateKernel(advec_mom_knl_vol_prog, "advec_mom_vol_ocl_kernel", &err);

    //advec_mom_node_x_knl_c              = clCreateKernel(advec_mom_knl_node_x_prog, "advec_mom_node_ocl_kernel_x", &err);

    //advec_mom_node_mass_pre_x_knl_c     = clCreateKernel(advec_mom_knl_node_mass_pre_x_prog, "advec_mom_node_mass_pre_ocl_kernel_x", &err);

    //advec_mom_flux_x_vec1_knl_c         = clCreateKernel(advec_mom_knl_mom_flux_x_vec1_prog, "advec_mom_flux_ocl_kernel_x_vec1", &err);

    //advec_mom_flux_x_vecnot1_knl_c      = clCreateKernel(advec_mom_knl_mom_flux_x_notvec1_prog, "advec_mom_flux_ocl_kernel_x_notvec1", &err);

    //advec_mom_vel_x_knl_c               = clCreateKernel(advec_mom_knl_vel_x_prog, "advec_mom_vel_ocl_kernel_x", &err);

    //advec_mom_node_y_knl_c              = clCreateKernel(advec_mom_knl_node_y_prog, "advec_mom_node_ocl_kernel_y", &err);

    //advec_mom_node_mass_pre_y_knl_c     = clCreateKernel(advec_mom_knl_node_mass_pre_y_prog, "advec_mom_node_mass_pre_ocl_kernel_y", &err);

    //advec_mom_flux_y_vec1_knl_c         = clCreateKernel(advec_mom_knl_mom_flux_y_vec1_prog, "advec_mom_flux_ocl_kernel_y_vec1", &err);
    //
    //advec_mom_flux_y_vecnot1_knl_c      = clCreateKernel(advec_mom_knl_mom_flux_y_notvec1_prog, "advec_mom_flux_ocl_kernel_y_notvec1", &err);

    //advec_mom_vel_y_knl_c               = clCreateKernel(advec_mom_knl_vel_y_prog, "advec_mom_vel_ocl_kernel_y", &err);       

    //pdv_correct_knl_c                   = clCreateKernel(pdv_fluxcalc_prog, "pdv_correct_ocl_kernel", &err);

    //pdv_predict_knl_c                   = clCreateKernel(pdv_fluxcalc_prog, "pdv_predict_ocl_kernel", &err);

    //dt_calc_knl_c                       = clCreateKernel(calcdt_minred_prog, "calc_dt_ocl_kernel", &err);

    //revert_knl_c                        = clCreateKernel(accel_revert_prog, "revert_ocl_kernel", &err);

    //reset_field_knl_c                   = clCreateKernel(field_sumred_reset_prog, "reset_field_ocl_kernel", &err);

    //generate_chunk_knl_c                = clCreateKernel(initialise_generate_chunk_prog, "generate_chunk_ocl_kernel", &err);

    //initialise_chunk_cell_x_knl_c       = clCreateKernel(initialise_generate_chunk_prog, "initialise_chunk_cell_x_ocl_kernel", &err);

    //initialise_chunk_cell_y_knl_c       = clCreateKernel(initialise_generate_chunk_prog, "initialise_chunk_cell_y_ocl_kernel", &err);

    //initialise_chunk_vertex_x_knl_c     = clCreateKernel(initialise_generate_chunk_prog, "initialise_chunk_vertex_x_ocl_kernel", &err);

    //initialise_chunk_vertex_y_knl_c     = clCreateKernel(initialise_generate_chunk_prog, "initialise_chunk_vertex_y_ocl_kernel", &err);

    //initialise_chunk_volume_area_knl_c  = clCreateKernel(initialise_generate_chunk_prog, "initialise_chunk_volume_area_ocl_kernel", &err);

    //field_summary_knl_c                 = clCreateKernel(field_sumred_reset_prog, "field_summary_ocl_kernel", &err);

    //update_halo_bottom_cell_knl_c       = clCreateKernel(ideal_vis_uh_prog, "update_halo_bottom_cell_ocl_kernel", &err);

    //update_halo_bottom_vel_knl_c        = clCreateKernel(ideal_vis_uh_prog, "update_halo_bottom_vel_ocl_kernel", &err);

    //update_halo_bottom_flux_x_knl_c     = clCreateKernel(ideal_vis_uh_prog, "update_halo_bottom_flux_x_ocl_kernel", &err);

    //update_halo_bottom_flux_y_knl_c     = clCreateKernel(ideal_vis_uh_prog, "update_halo_bottom_flux_y_ocl_kernel", &err);

    //update_halo_top_cell_knl_c          = clCreateKernel(ideal_vis_uh_prog, "update_halo_top_cell_ocl_kernel", &err);

    //update_halo_top_vel_knl_c           = clCreateKernel(ideal_vis_uh_prog, "update_halo_top_vel_ocl_kernel", &err); 

    //update_halo_top_flux_x_knl_c        = clCreateKernel(ideal_vis_uh_prog, "update_halo_top_flux_x_ocl_kernel", &err);

    //update_halo_top_flux_y_knl_c        = clCreateKernel(ideal_vis_uh_prog, "update_halo_top_flux_y_ocl_kernel", &err);

    //update_halo_right_cell_knl_c        = clCreateKernel(ideal_vis_uh_prog, "update_halo_right_cell_ocl_kernel", &err);

    //update_halo_right_vel_knl_c         = clCreateKernel(ideal_vis_uh_prog, "update_halo_right_vel_ocl_kernel", &err);

    //update_halo_right_flux_x_knl_c      = clCreateKernel(ideal_vis_uh_prog, "update_halo_right_flux_x_ocl_kernel", &err);

    //update_halo_right_flux_y_knl_c      = clCreateKernel(ideal_vis_uh_prog, "update_halo_right_flux_y_ocl_kernel", &err);

    //update_halo_left_cell_knl_c         = clCreateKernel(ideal_vis_uh_prog, "update_halo_left_cell_ocl_kernel", &err);

    //update_halo_left_vel_knl_c          = clCreateKernel(ideal_vis_uh_prog, "update_halo_left_vel_ocl_kernel", &err);

    //update_halo_left_flux_x_knl_c       = clCreateKernel(ideal_vis_uh_prog, "update_halo_left_flux_x_ocl_kernel", &err);

    //update_halo_left_flux_y_knl_c       = clCreateKernel(ideal_vis_uh_prog, "update_halo_left_flux_y_ocl_kernel", &err);

    //read_top_buffer_knl_c               = clCreateKernel(pack_comms_buffers_prog, "top_comm_buffer_pack", &err);

    //read_bottom_buffer_knl_c            = clCreateKernel(pack_comms_buffers_prog, "bottom_comm_buffer_pack", &err);

    //read_right_buffer_knl_c             = clCreateKernel(pack_comms_buffers_prog, "right_comm_buffer_pack", &err);

    //read_left_buffer_knl_c              = clCreateKernel(pack_comms_buffers_prog, "left_comm_buffer_pack", &err);

    //write_top_buffer_knl_c              = clCreateKernel(unpack_comms_buffers_prog, "top_comm_buffer_unpack", &err);

    //write_bottom_buffer_knl_c           = clCreateKernel(unpack_comms_buffers_prog, "bottom_comm_buffer_unpack", &err);

    //write_right_buffer_knl_c            = clCreateKernel(unpack_comms_buffers_prog, "right_comm_buffer_unpack", &err);

    //write_left_buffer_knl_c             = clCreateKernel(unpack_comms_buffers_prog, "left_comm_buffer_unpack", &err);

    std::cout << "at end of the kernel creation" << std::endl; 

}

void CloverCL::readVisualisationBuffers(
                int x_max,
                int y_max,
                double* vertexx,
                double* vertexy,
                double* density0,
                double* energy0,
                double* pressure,
                double* viscosity,
                double* xvel0,
                double* yvel0)
{

    //cl::Event event1, event2, event3, event4, event5, event6, event7, event8;
    //std::vector<cl::Event> events;
    cl_int err; 

    //queue.enqueueReadBuffer( CloverCL::vertexx_buffer, CL_FALSE, 0, (x_max+5)*sizeof(double), vertexx, NULL, &event1);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::vertexx_buffer_c, CL_FALSE, 0, (x_max+5)*sizeof(double), vertexx, 0, NULL, NULL); 
    checkErr(err, "readVisualisationBuffers() vertexx");

    //queue.enqueueReadBuffer( CloverCL::vertexy_buffer, CL_FALSE, 0, (y_max+5)*sizeof(double), vertexy, NULL, &event2);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::vertexy_buffer_c, CL_FALSE, 0, (y_max+5)*sizeof(double), vertexy, 0, NULL, NULL); 
    checkErr(err, "readVisualisationBuffers() vertexy");

    //queue.enqueueReadBuffer( CloverCL::density0_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), density0, NULL, &event3);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::density0_buffer_c, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), density0, 0, NULL, NULL ); 
    checkErr(err, "readVisualisationBuffers() density0");

    //queue.enqueueReadBuffer( CloverCL::energy0_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), energy0, NULL, &event4);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::energy0_buffer_c, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), energy0, 0, NULL, NULL); 
    checkErr(err, "readVisualisationBuffers() energy0");

    //queue.enqueueReadBuffer( CloverCL::pressure_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), pressure, NULL, &event5);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::pressure_buffer_c, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), pressure, 0, NULL, NULL); 
    checkErr(err, "readVisualisationBuffers() pressure");

    //queue.enqueueReadBuffer( CloverCL::viscosity_buffer, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), viscosity, NULL, &event6);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::viscosity_buffer_c, CL_FALSE, 0, (x_max+4)*(y_max+4)*sizeof(double), viscosity, 0, NULL, NULL); 
    checkErr(err, "readVisualisationBuffers() viscosity");

    //queue.enqueueReadBuffer( CloverCL::xvel0_buffer, CL_FALSE, 0, (x_max+5)*(y_max+5)*sizeof(double), xvel0, NULL, &event7);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::xvel0_buffer_c, CL_FALSE, 0, (x_max+5)*(y_max+5)*sizeof(double), xvel0, 0, NULL, NULL); 
    checkErr(err, "readVisualisationBuffers() xvel0");

    //queue.enqueueReadBuffer( CloverCL::yvel0_buffer, CL_FALSE, 0, (x_max+5)*(y_max+5)*sizeof(double), yvel0, NULL, &event8);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::yvel0_buffer_c, CL_FALSE, 0, (x_max+5)*(y_max+5)*sizeof(double), yvel0, 0, NULL, NULL); 
    checkErr(err, "readVisualisationBuffers() yvel0");

    //events.push_back(event1);
    //events.push_back(event2);
    //events.push_back(event3);
    //events.push_back(event4);
    //events.push_back(event5);
    //events.push_back(event6);
    //events.push_back(event7);
    //events.push_back(event8);

    //cl::Event::waitForEvents(events);
    
    err = clFinish(CloverCL::queue_c);
}

void CloverCL::readCommunicationBuffer(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge)
{
    //cl::Buffer* field_buffer;
    //cl::Buffer* comm_buffer;
    //cl::Event event1;
    //int buff_length;
    //int buff_min;

    //global_events.clear();
    //global_events.push_back(last_event);

    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //std::stringstream ss_rank;
    //ss_rank << rank;

    //switch(*field_name) {
    //    case FIELD_DENSITY0: field_buffer = &density0_buffer; break;
    //    case FIELD_DENSITY1: field_buffer = &density1_buffer; break;
    //    case FIELD_ENERGY0: field_buffer = &energy0_buffer; break;
    //    case FIELD_ENERGY1: field_buffer = &energy1_buffer; break;
    //    case FIELD_PRESSURE: field_buffer = &pressure_buffer; break;
    //    case FIELD_VISCOSITY: field_buffer = &viscosity_buffer; break;
    //    case FIELD_SOUNDSPEED: field_buffer = &soundspeed_buffer; break;
    //    case FIELD_XVEL0: field_buffer = &xvel0_buffer; break;
    //    case FIELD_XVEL1: field_buffer = &xvel1_buffer; break;
    //    case FIELD_YVEL0: field_buffer = &yvel0_buffer; break;
    //    case FIELD_YVEL1: field_buffer = &yvel1_buffer; break;
    //    case FIELD_VOL_FLUX_X: field_buffer = &vol_flux_x_buffer; break;
    //    case FIELD_VOL_FLUX_Y: field_buffer = &vol_flux_y_buffer; break;
    //    case FIELD_MASS_FLUX_X: field_buffer = &mass_flux_x_buffer; break;
    //    case FIELD_MASS_FLUX_Y: field_buffer = &mass_flux_y_buffer; break;
    //}

    //cl::size_t<3> b_origin;
    //cl::size_t<3> h_origin;
    //cl::size_t<3> region;

    //size_t b_row_pitch = sizeof(double) * (*xmax + *xinc + 4);
    //size_t b_slice_pitch = 0;
    //size_t h_row_pitch = 0;
    //size_t h_slice_pitch = 0;

    //h_origin[0] = 0;
    //h_origin[1] = 0;
    //h_origin[2] = 0;

    //switch(*which_edge) {
    //    case 1: comm_buffer = &(top_send_buffer);
    //            buff_length = *xmax + *xinc + (2 * *depth);
    //            buff_min = *xmin; 
    //            b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
    //            b_origin[1] = ((*ymax+1)-(*depth-1));
    //            b_origin[2] = 0;
    //            region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
    //            region[1] = *depth;
    //            region[2] = 1;
    //            break;
    //    case 2: comm_buffer = &right_send_buffer;
    //            buff_length = *ymax + *yinc + (2 * *depth);
    //            buff_min = *ymin; 
    //            b_origin[0] = ((*xmax+1)-(*depth-1))*sizeof(double);
    //            b_origin[1] = ((*ymin+1) - (*depth));
    //            b_origin[2] = 0;
    //            region[0] = (*depth)*sizeof(double);
    //            region[1] = (*ymax)+*yinc+(2* *depth);
    //            region[2] = 1;
    //            break;
    //    case 3: comm_buffer = &bottom_send_buffer;
    //            buff_length = *xmax + *xinc + (2 * *depth);
    //            buff_min = *xmin; 
    //            b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
    //            b_origin[1] = (*ymin+1+*yinc);
    //            b_origin[2] = 0;
    //            region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
    //            region[1] = *depth;
    //            region[2] = 1;
    //            break;
    //    case 4: comm_buffer = &left_send_buffer;
    //            buff_length = *ymax + *yinc + (2 * *depth);
    //            buff_min = *ymin; 
    //            b_origin[0] = ((*xmin+1+*xinc)*sizeof(double));
    //            b_origin[1] = ((*ymin+1) - (*depth));
    //            b_origin[2] = 0;
    //            region[0] = (*depth)*sizeof(double);
    //            region[1] = (*ymax)+*yinc+(2* *depth);
    //            region[2] = 1;
    //            break;
    //}

    //buff_length = buff_length * *depth;

    //try {
    //    queue.enqueueReadBufferRect( *field_buffer, CL_TRUE, b_origin, h_origin, region, b_row_pitch, 
    //                                 b_slice_pitch, h_row_pitch, h_slice_pitch, buffer, &global_events);
    //} catch (cl::Error err) {
    //    reportError(err, "readCommunicationBuffer enqueueReadBufferRect");
    //}
}

void CloverCL::writeCommunicationBuffer(
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* depth,
        int* xinc,
        int* yinc,
        int* field_name,
        double* buffer,
        int* which_edge)
{
    //cl::Buffer* field_buffer;
    //cl::Buffer* comm_buffer;
    //cl::Kernel* comm_kernel;
    //cl::Event event1;
    //int buff_length;
    //int buff_min;

    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //std::stringstream ss_rank;
    //ss_rank << rank;

    //switch(*field_name) {
    //    case FIELD_DENSITY0: field_buffer = &density0_buffer; break;
    //    case FIELD_DENSITY1: field_buffer = &density1_buffer; break;
    //    case FIELD_ENERGY0: field_buffer = &energy0_buffer; break;
    //    case FIELD_ENERGY1: field_buffer = &energy1_buffer; break;
    //    case FIELD_PRESSURE: field_buffer = &pressure_buffer; break;
    //    case FIELD_VISCOSITY: field_buffer = &viscosity_buffer; break;
    //    case FIELD_SOUNDSPEED: field_buffer = &soundspeed_buffer; break;
    //    case FIELD_XVEL0: field_buffer = &xvel0_buffer; break;
    //    case FIELD_XVEL1: field_buffer = &xvel1_buffer; break;
    //    case FIELD_YVEL0: field_buffer = &yvel0_buffer; break;
    //    case FIELD_YVEL1: field_buffer = &yvel1_buffer; break;
    //    case FIELD_VOL_FLUX_X: field_buffer = &vol_flux_x_buffer; break;
    //    case FIELD_VOL_FLUX_Y: field_buffer = &vol_flux_y_buffer; break;
    //    case FIELD_MASS_FLUX_X: field_buffer = &mass_flux_x_buffer; break;
    //    case FIELD_MASS_FLUX_Y: field_buffer = &mass_flux_y_buffer; break;
    //}

    //cl::size_t<3> b_origin;
    //cl::size_t<3> h_origin;
    //cl::size_t<3> region;

    //size_t b_row_pitch = sizeof(double) * (*xmax + *xinc + 4);
    //size_t b_slice_pitch = 0;
    //size_t h_row_pitch = 0;
    //size_t h_slice_pitch = 0;

    //h_origin[0] = 0;
    //h_origin[1] = 0;
    //h_origin[2] = 0;

    //switch(*which_edge) {
    //    case 1: comm_buffer = &(top_send_buffer);
    //            buff_length = *xmax + *xinc + (2 * *depth);
    //            buff_min = *xmin; 
    //            b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
    //            b_origin[1] = ((*ymax+1)+1+*yinc);
    //            b_origin[2] = 0;
    //            region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
    //            region[1] = *depth;
    //            region[2] = 1;
    //            break;
    //    case 2: comm_buffer = &right_send_buffer;
    //            buff_length = *ymax + *yinc + (2 * *depth);
    //            buff_min = *ymin; 
    //            b_origin[0] = ((*xmax+1)+1+*xinc)*sizeof(double);
    //            b_origin[1] = ((*ymin+1) - (*depth));
    //            b_origin[2] = 0;
    //            region[0] = (*depth)*sizeof(double);
    //            region[1] = (*ymax)+*yinc+(2* *depth);
    //            region[2] = 1;
    //            break;
    //    case 3: comm_buffer = &bottom_send_buffer;
    //            buff_length = *xmax + *xinc + (2 * *depth);
    //            buff_min = *xmin; 
    //            b_origin[0] = ((*xmin+1) - *depth)*sizeof(double);
    //            b_origin[1] = (*ymin+1)-*depth;
    //            b_origin[2] = 0;
    //            region[0] = ((*xmax)+*xinc+2*(*depth))*sizeof(double);
    //            region[1] = *depth;
    //            region[2] = 1;
    //            break;
    //    case 4: comm_buffer = &left_send_buffer;
    //            buff_length = *ymax + *yinc + (2 * *depth);
    //            buff_min = *ymin; 
    //            b_origin[0] = ((*xmin+1-(*depth))*sizeof(double));
    //            b_origin[1] = ((*ymin+1) - (*depth));
    //            b_origin[2] = 0;
    //            region[0] = (*depth)*sizeof(double);
    //            region[1] = (*ymax)+*yinc+(2* *depth);
    //            region[2] = 1;
    //            break;
    //}

    //buff_length = buff_length * *depth;

    //try {
    //    queue.enqueueWriteBufferRect( *field_buffer, CL_TRUE, b_origin, h_origin, region, b_row_pitch, 
    //                                  b_slice_pitch, h_row_pitch, h_slice_pitch, buffer); 
    //} catch (cl::Error err) {
    //    reportError(err, "writeCommunicationBuffer enqueueWriteBufferRect");
    //}
}

void CloverCL::readAllCommunicationBuffers(
        int* x_max,
        int* y_max,
        double* density0,
        double* density1,
        double* energy0,
        double* energy1,
        double* pressure,
        double* viscosity,
        double* soundspeed,
        double* xvel0,
        double* xvel1,
        double* yvel0,
        double* yvel1,
        double* vol_flux_x,
        double* vol_flux_y,
        double* mass_flux_x,
        double* mass_flux_y)
{

    //cl::Event event1;
    cl_int err; 

        //queue.enqueueReadBuffer( CloverCL::density0_buffer, CL_TRUE, 0, 
        //                        (*x_max+4)*(*y_max+4)*sizeof(double), density0, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::density0_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), density0, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() density0");

        //queue.enqueueReadBuffer( CloverCL::density1_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), density1, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::density1_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), density1, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() density1");

        //queue.enqueueReadBuffer( CloverCL::energy0_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), energy0, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::energy0_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), energy0, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() energy0");

        //queue.enqueueReadBuffer( CloverCL::energy1_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), energy1, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::energy1_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), energy1, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() energy1");

        //queue.enqueueReadBuffer( CloverCL::pressure_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), pressure, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::pressure_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), pressure, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() pressure");

        //queue.enqueueReadBuffer( CloverCL::viscosity_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), viscosity, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::viscosity_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), viscosity, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() viscosity");

        //queue.enqueueReadBuffer( CloverCL::soundspeed_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), soundspeed, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::soundspeed_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), soundspeed, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() soundspeed");

        //queue.enqueueReadBuffer( CloverCL::xvel0_buffer, CL_TRUE, 0, 
        //                         (*x_max+5)*(*y_max+5)*sizeof(double), xvel0, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::xvel0_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), xvel0, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() xvel0");

        //queue.enqueueReadBuffer( CloverCL::xvel1_buffer, CL_TRUE, 0, 
        //                         (*x_max+5)*(*y_max+5)*sizeof(double), xvel1, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::xvel1_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), xvel1, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() xvel1");

        //queue.enqueueReadBuffer( CloverCL::yvel0_buffer, CL_TRUE, 0, 
        //                         (*x_max+5)*(*y_max+5)*sizeof(double), yvel0, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::yvel0_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), yvel0, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() yvel0");

        //queue.enqueueReadBuffer( CloverCL::yvel1_buffer, CL_TRUE, 0, 
        //                         (*x_max+5)*(*y_max+5)*sizeof(double), yvel1, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::yvel1_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), yvel1, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() yvel1");

        //queue.enqueueReadBuffer( CloverCL::mass_flux_x_buffer, CL_TRUE, 0, 
        //                         (*x_max+5)*(*y_max+4)*sizeof(double), mass_flux_x, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::mass_flux_x_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+4)*sizeof(double), mass_flux_x, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() mass_flux_x");

        //queue.enqueueReadBuffer( CloverCL::vol_flux_x_buffer, CL_TRUE, 0, 
        //                         (*x_max+5)*(*y_max+4)*sizeof(double), vol_flux_x, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::vol_flux_x_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+4)*sizeof(double), vol_flux_x, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() vol_flux_x");

        //queue.enqueueReadBuffer( CloverCL::mass_flux_y_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+5)*sizeof(double), mass_flux_y, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::mass_flux_y_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+5)*sizeof(double), mass_flux_y, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() mass_flux_y");

        //queue.enqueueReadBuffer( CloverCL::vol_flux_y_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+5)*sizeof(double), vol_flux_y, NULL, &event1);
        err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::vol_flux_y_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+5)*sizeof(double), vol_flux_y, 0, NULL, NULL); 
        checkErr(err, "readAllCommunicationBuffers() vol_flux_y");
}

void CloverCL::writeAllCommunicationBuffers(
        int* x_max,
        int* y_max,
        double* density0,
        double* density1,
        double* energy0,
        double* energy1,
        double* pressure,
        double* viscosity,
        double* soundspeed,
        double* xvel0,
        double* xvel1,
        double* yvel0,
        double* yvel1,
        double* vol_flux_x,
        double* vol_flux_y,
        double* mass_flux_x,
        double* mass_flux_y)
{

    //cl::Event event1;
    cl_int err; 

        //queue.enqueueWriteBuffer( CloverCL::density0_buffer, CL_TRUE, 0, 
        //                          (*x_max+4)*(*y_max+4)*sizeof(double), density0, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::density0_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), density0, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() density0");

        //queue.enqueueWriteBuffer( CloverCL::density1_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), density1, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::density1_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), density1, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() density1");

        //queue.enqueueWriteBuffer( CloverCL::energy0_buffer, CL_TRUE, 0, 
        //                         (*x_max+4)*(*y_max+4)*sizeof(double), energy0, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::energy0_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), energy0, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() energy0");

        //queue.enqueueWriteBuffer( CloverCL::energy1_buffer, CL_TRUE, 0, 
        //                          (*x_max+4)*(*y_max+4)*sizeof(double), energy1, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::energy1_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), energy1, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() energy1");

        //queue.enqueueWriteBuffer( CloverCL::pressure_buffer, CL_TRUE, 0, 
        //                          (*x_max+4)*(*y_max+4)*sizeof(double), pressure, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::pressure_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), pressure, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() pressure");

        //queue.enqueueWriteBuffer( CloverCL::viscosity_buffer, CL_TRUE, 0, 
        //                          (*x_max+4)*(*y_max+4)*sizeof(double), viscosity, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::viscosity_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), viscosity, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() viscosity");

        //queue.enqueueWriteBuffer( CloverCL::soundspeed_buffer, CL_TRUE, 0, 
        //                          (*x_max+4)*(*y_max+4)*sizeof(double), soundspeed, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::soundspeed_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+4)*sizeof(double), soundspeed, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() soundspeed");

        //queue.enqueueWriteBuffer( CloverCL::xvel0_buffer, CL_TRUE, 0, 
        //                          (*x_max+5)*(*y_max+5)*sizeof(double), xvel0, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xvel0_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), xvel0, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() xvel0");

        //queue.enqueueWriteBuffer( CloverCL::xvel1_buffer, CL_TRUE, 0, 
        //                          (*x_max+5)*(*y_max+5)*sizeof(double), xvel1, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xvel1_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), xvel1, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() xvel1");

        //queue.enqueueWriteBuffer( CloverCL::yvel0_buffer, CL_TRUE, 0, 
        //                          (*x_max+5)*(*y_max+5)*sizeof(double), yvel0, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::yvel0_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), yvel0, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() yvel0");

        //queue.enqueueWriteBuffer( CloverCL::yvel1_buffer, CL_TRUE, 0, 
        //                          (*x_max+5)*(*y_max+5)*sizeof(double), yvel1, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::yvel1_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+5)*sizeof(double), yvel1, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() yvel1");

        //queue.enqueueWriteBuffer( CloverCL::mass_flux_x_buffer, CL_TRUE, 0, 
        //                          (*x_max+5)*(*y_max+4)*sizeof(double), mass_flux_x, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::mass_flux_x_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+4)*sizeof(double), mass_flux_x, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() mass_flux_x");

        //queue.enqueueWriteBuffer( CloverCL::vol_flux_x_buffer, CL_TRUE, 0, 
        //                          (*x_max+5)*(*y_max+4)*sizeof(double), vol_flux_x, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::vol_flux_x_buffer_c, CL_TRUE, 0, (*x_max+5)*(*y_max+4)*sizeof(double), vol_flux_x, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() vol_flux_x");

        //queue.enqueueWriteBuffer( CloverCL::mass_flux_y_buffer, CL_TRUE, 0, 
        //                          (*x_max+4)*(*y_max+5)*sizeof(double), mass_flux_y, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::mass_flux_y_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+5)*sizeof(double), mass_flux_y, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() mass_flux_y");

        //queue.enqueueWriteBuffer( CloverCL::vol_flux_y_buffer, CL_TRUE, 0, 
        //                          (*x_max+4)*(*y_max+5)*sizeof(double), vol_flux_y, NULL, &event1);
        err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::vol_flux_y_buffer_c, CL_TRUE, 0, (*x_max+4)*(*y_max+5)*sizeof(double), vol_flux_y, 0, NULL, NULL);  
        checkErr(err, "readAllCommunicationBuffers() vol_flux_y");
}

void CloverCL::enqueueKernel_nooffsets( cl_kernel kernel, int num_x, int num_y, double * event_time)
{
    cl_int err; 

    int x_rnd = (num_x / fixed_wg_min_size_large_dim ) * fixed_wg_min_size_large_dim;

    if ((x_rnd != num_x))
        x_rnd = x_rnd + fixed_wg_min_size_large_dim;


    int y_rnd = ( num_y / fixed_wg_min_size_small_dim ) * fixed_wg_min_size_small_dim;

    if (y_rnd != num_y) {
        y_rnd = y_rnd + fixed_wg_min_size_small_dim; 
    }

    //size_t global_wi [2] = {x_rnd, y_rnd}; 
    //size_t local_wi [2] = {fixed_wg_min_size_large_dim, fixed_wg_min_size_small_dim}; 
    size_t global_wi [2] = {961,3843}; 
    size_t local_wi [2] = {1,1}; 
                
        err = clEnqueueNDRangeKernel(queue_c, kernel, 2, NULL, global_wi, local_wi, 0, NULL, &last_event );


        //queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange(x_rnd, y_rnd), 
        //                            cl::NDRange(fixed_wg_min_size_large_dim,fixed_wg_min_size_small_dim), 
        //                            NULL, &last_event); 

    if ( err != CL_SUCCESS) {

        size_t kernel_name_size;
        char *kernel_name;

        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &kernel_name_size);
        kernel_name = new char[kernel_name_size];  
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, kernel_name_size, kernel_name, NULL);
        //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);

        std::string kernel_name_str = std::string(kernel_name);

        std::cout << "launching kernel: " << kernel_name_str << " xnum: " << x_rnd << " ynum: " << num_y 
                  << " wg_x: " << fixed_wg_min_size_large_dim << " wg_y: " << fixed_wg_min_size_small_dim << std::endl;

        reportError(err, kernel_name_str);
    }

#if PROFILE_OCL_KERNELS
    cl_ulong knl_start, knl_end;
    double diff;

    size_t kernel_name_size;
    char *kernel_name;

    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &kernel_name_size);
    kernel_name = new char[kernel_name_size];  
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, kernel_name_size, kernel_name, NULL);
    //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);

    std::string kernel_name_str = std::string(kernel_name);
    //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);


    clWaitForEvents(1, &last_event); 
    //last_event.wait();

    //last_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &knl_start);
    //last_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &knl_end);

    clGetEventProfilingInfo(last_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &knl_start, NULL); 
    clGetEventProfilingInfo(last_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &knl_end, NULL); 

    diff = (knl_end - knl_start)*CloverCL::NS_TO_SECONDS;

    std::cout << "[PROFILING]: " << kernel_name_str << " OpenCL kernel took "
              << diff << " seconds (device time)" << std::endl;

    *event_time = diff;
#endif
}

void CloverCL::enqueueKernel(cl_kernel kernel, int x_min, int x_max, int y_min, int y_max)
{

    int x_max_opt;
    int x_tot = (x_max - x_min) + 1;

    int x_rnd = (x_tot / fixed_wg_min_size_large_dim) * fixed_wg_min_size_large_dim;

    if ((x_rnd != x_tot))
        x_rnd = x_rnd + fixed_wg_min_size_large_dim;


    x_max_opt = x_rnd + x_min - 1;

    cl_int err;
    size_t global_wi [2] = {x_max_opt, y_max};  
    size_t offsets [2] = {x_min, y_min};
    size_t local_wi [2] = {fixed_wg_min_size_large_dim, fixed_wg_min_size_small_dim}; 

        //queue.enqueueNDRangeKernel( kernel, cl::NDRange(x_min, y_min), cl::NDRange(x_max_opt, y_max), 
        //                            cl::NDRange(fixed_wg_min_size_large_dim, fixed_wg_min_size_small_dim), 
        //                            NULL, &last_event);

        err = clEnqueueNDRangeKernel(queue_c, kernel, 2, offsets, global_wi, local_wi, 0, NULL, &last_event); 

    if (err != CL_SUCCESS) {

        char * kernel_name;
        size_t kernel_name_size; 

        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &kernel_name_size);
        kernel_name = new char[kernel_name_size];
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, kernel_name_size, kernel_name, NULL);
        //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);

        std::string kernel_name_str = std::string(kernel_name); 
        std::cout << "launching kernel: " << kernel_name_str << " xmin: " << x_min << " xmax: " << x_max_opt 
                                          << " ymin: " << y_min << " ymax: " << y_max << std::endl;
        reportError(err, kernel_name_str);
    }

#if PROFILE_OCL_KERNELS
    cl_ulong knl_start, knl_end;

    size_t kernel_name_size;
    char *kernel_name;

    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &kernel_name_size);
    kernel_name = new char[kernel_name_size];
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, kernel_name_size, kernel_name, NULL);
    //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);

    std::string kernel_name_str = std::string(kernel_name);
    //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);

    clWaitForEvents(1, &last_event);
    //last_event.wait();

    //last_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &knl_start);
    //last_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &knl_end);

    clGetEventProfilingInfo(last_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &knl_start, NULL);
    clGetEventProfilingInfo(last_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &knl_end, NULL);

    std::cout << "[PROFILING]: " << kernel_name_str << " OpenCL kernel took "
              << (knl_end - knl_start)*CloverCL::NS_TO_SECONDS
              << " seconds (device time)" << std::endl;
#endif
}

void CloverCL::enqueueKernel(cl_kernel kernel, int min, int max)
{
    cl_int err; 

    int tot = (max - min) + 1;

    int rnd = (tot / prefer_wg_multiple) * prefer_wg_multiple;

    if ((rnd != tot))
        rnd = rnd + prefer_wg_multiple;

    int min_opt, max_opt;

    min_opt = min;

    max_opt = rnd + min - 1;
    size_t offsets [1] = {min_opt};
    size_t global_wi [1] = {max_opt}; 

        //queue.enqueueNDRangeKernel( kernel, cl::NDRange(min_opt), cl::NDRange(max_opt), cl::NullRange, NULL, &last_event);

        err = clEnqueueNDRangeKernel(queue_c, kernel, 1, offsets, global_wi, NULL, 0, NULL, &last_event);

    if (err != CL_SUCCESS) {

        char* kernel_name;
        size_t kernel_name_size;
        
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &kernel_name_size); 
        kernel_name = new char[kernel_name_size];
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, kernel_name_size, kernel_name, NULL);
        //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);

        std::string kernel_name_str = std::string(kernel_name);
        reportError(err, kernel_name_str);
    }

#if PROFILE_OCL_KERNELS
    cl_ulong knl_start, knl_end;

    //kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &kernel_name);
    char* kernel_name;
    size_t kernel_name_size;
    
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &kernel_name_size);
    kernel_name = new char[kernel_name_size];
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, kernel_name_size, kernel_name, NULL);

    std::string kernel_name_str = std::string(kernel_name);

    //last_event.wait();
    clWaitForEvents(1, &last_event);

    //last_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &knl_start);
    //last_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &knl_end);

    clGetEventProfilingInfo(last_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &knl_start, NULL);
    clGetEventProfilingInfo(last_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &knl_end, NULL);

    std::cout << "[PROFILING]: " << kernel_name_str << " OpenCL kernel took "
              << (knl_end - knl_start)*CloverCL::NS_TO_SECONDS
              << " seconds (device time)" << std::endl;
#endif
}


void CloverCL::read_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume)
{
    cl_int err;

    //CloverCL::queue.finish();
    //CloverCL::outoforder_queue.finish(); 
    err = clFinish(CloverCL::queue_c);
    err = clFinish(CloverCL::outoforder_queue_c);
    

    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::density0_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::density1_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::energy0_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::energy1_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::pressure_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), pressure, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::viscosity_buffer,   CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), viscosity, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::soundspeed_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), soundspeed, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::xvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::xvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::yvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::yvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::vol_flux_x_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), vol_flux_x, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::vol_flux_y_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), vol_flux_y, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::mass_flux_x_buffer, CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), mass_flux_x, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::mass_flux_y_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), mass_flux_y, NULL, NULL);

    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::density0_buffer_c,    CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density0, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::density1_buffer_c,    CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::energy0_buffer_c,     CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy0, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::energy1_buffer_c,     CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::pressure_buffer_c,    CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), pressure, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::viscosity_buffer_c,   CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), viscosity, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::soundspeed_buffer_c,  CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), soundspeed, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::xvel0_buffer_c,       CL_FALSE, 0, 
                              (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::xvel1_buffer_c,       CL_FALSE, 0, 
                              (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::yvel0_buffer_c,       CL_FALSE, 0, 
                              (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::yvel1_buffer_c,       CL_FALSE, 0, 
                              (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::vol_flux_x_buffer_c,  CL_FALSE, 0, 
                              (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), vol_flux_x, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::vol_flux_y_buffer_c,  CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), vol_flux_y, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::mass_flux_x_buffer_c, CL_FALSE, 0, 
                              (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), mass_flux_x, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::mass_flux_y_buffer_c, CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), mass_flux_y, 0, NULL, NULL);

    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::celldx_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*sizeof(double), celldx, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::celldy_buffer, CL_FALSE, 0, (CloverCL::ymax_c+4)*sizeof(double), celldy, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueReadBuffer(CloverCL::volume_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), volume, NULL, NULL);

    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::celldx_buffer_c, CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*sizeof(double), celldx, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::celldy_buffer_c, CL_FALSE, 0, 
                              (CloverCL::ymax_c+4)*sizeof(double), celldy, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::outoforder_queue_c, CloverCL::volume_buffer_c, CL_FALSE, 0, 
                              (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), volume, 0, NULL, NULL);

    //CloverCL::outoforder_queue.finish();
    err = clFinish(CloverCL::outoforder_queue_c);
}


void CloverCL::write_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume )
{
    cl_int err; 

    //CloverCL::queue.finish();
    //CloverCL::outoforder_queue.finish(); 
    err = clFinish(CloverCL::queue_c);
    err = clFinish(CloverCL::outoforder_queue_c);

    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::density0_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::density1_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::energy0_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::energy1_buffer,     CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::pressure_buffer,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), pressure, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::viscosity_buffer,   CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), viscosity, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::soundspeed_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), soundspeed, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::xvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::xvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::yvel0_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::yvel1_buffer,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::vol_flux_x_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), vol_flux_x, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::vol_flux_y_buffer,  CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), vol_flux_y, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::mass_flux_x_buffer, CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), mass_flux_x, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::mass_flux_y_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), mass_flux_y, NULL, NULL);

    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::density0_buffer_c,    CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::density1_buffer_c,    CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::energy0_buffer_c,     CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::energy1_buffer_c,     CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), energy1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::pressure_buffer_c,    CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), pressure, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::viscosity_buffer_c,   CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), viscosity, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::soundspeed_buffer_c,  CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), soundspeed, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::xvel0_buffer_c,       CL_FALSE, 0, 
                               (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::xvel1_buffer_c,       CL_FALSE, 0, 
                               (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::yvel0_buffer_c,       CL_FALSE, 0, 
                               (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::yvel1_buffer_c,       CL_FALSE, 0, 
                               (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::vol_flux_x_buffer_c,  CL_FALSE, 0, 
                               (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), vol_flux_x, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::vol_flux_y_buffer_c,  CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), vol_flux_y, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::mass_flux_x_buffer_c, CL_FALSE, 0, 
                               (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), mass_flux_x, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::mass_flux_y_buffer_c, CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), mass_flux_y, 0, NULL, NULL);

    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::celldx_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*sizeof(double), celldx, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::celldy_buffer, CL_FALSE, 0, (CloverCL::ymax_c+4)*sizeof(double), celldy, NULL, NULL);
    //CloverCL::outoforder_queue.enqueueWriteBuffer(CloverCL::volume_buffer, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), volume, NULL, NULL);

    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::celldx_buffer_c, CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*sizeof(double), celldx, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::celldy_buffer_c, CL_FALSE, 0, 
                               (CloverCL::ymax_c+4)*sizeof(double), celldy, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::outoforder_queue_c, CloverCL::volume_buffer_c, CL_FALSE, 0, 
                               (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), volume, 0, NULL, NULL);

    //CloverCL::outoforder_queue.finish();
    err = clFinish(CloverCL::outoforder_queue_c);
}

void CloverCL::write_accelerate_buffers_tocard(double* density0, double* pressure, double* viscosity, 
                                               double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                               double* volume , double* xarea, double* yarea)
{
    cl_int err; 

    err = clFinish(CloverCL::queue_c);

    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::density0_buffer_c,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), density0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::pressure_buffer_c,    CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), pressure, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::viscosity_buffer_c,   CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), viscosity, 0, NULL, NULL);

    //err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xvel0_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, 0, NULL, NULL);
    //err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, 0, NULL, NULL);
    //err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::yvel0_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, 0, NULL, NULL);
    //err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::yvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xvel0_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+8)*(CloverCL::ymax_c+5)*sizeof(double), xvel0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+8)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::yvel0_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+8)*(CloverCL::ymax_c+5)*sizeof(double), yvel0, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::yvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+8)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, 0, NULL, NULL);


    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::volume_buffer_c, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+4)*sizeof(double), volume, 0, NULL, NULL);

    //err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xarea_buffer_c, CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+4)*sizeof(double), xarea, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::xarea_buffer_c, CL_FALSE, 0, (CloverCL::xmax_c+8)*(CloverCL::ymax_c+4)*sizeof(double), xarea, 0, NULL, NULL);

    err = clEnqueueWriteBuffer(CloverCL::queue_c, CloverCL::yarea_buffer_c, CL_FALSE, 0, (CloverCL::xmax_c+4)*(CloverCL::ymax_c+5)*sizeof(double), yarea, 0, NULL, NULL);

    err = clFinish(CloverCL::queue_c);
}

void CloverCL::read_accelerate_buffers_backfromcard(double* xvel1, double* yvel1)
{
    cl_int err;

    err = clFinish(CloverCL::queue_c);
    
    //err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::xvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::xvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+8)*(CloverCL::ymax_c+5)*sizeof(double), xvel1, 0, NULL, NULL);

    //err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::yvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+5)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(CloverCL::queue_c, CloverCL::yvel1_buffer_c,       CL_FALSE, 0, (CloverCL::xmax_c+8)*(CloverCL::ymax_c+5)*sizeof(double), yvel1, 0, NULL, NULL);


    err = clFinish(CloverCL::queue_c);
}

void CloverCL::call_clfinish()
{
    cl_int err;

    err = clFinish(queue_c);
}

inline void CloverCL::checkErr(cl_int err, std::string name)
{
    if (err != CL_SUCCESS) {
        std::cout << "ERROR: " << name << " (" << errToString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef OCL_VERBOSE
        std::cout << "SUCCESS: " << name << std::endl;
#endif
    }
}

void CloverCL::reportError( cl_int err, std::string message)
{
    std::cout << "[CloverCL] ERROR: " << message << " " << "(" 
              << CloverCL::errToString(err) << ")" << std::endl;
    exit(EXIT_FAILURE);
}

std::string CloverCL::errToString(cl_int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        //case CL_INVALID_PROPERTY:                   return "Invalid property";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default: return "Unknown";
    }
}

void CloverCL::dumpBinary() {

    cl_int err;

    const std::string binary_name = "cloverleaf_ocl_binary";

    printf("Dumping binary to %s:\n", binary_name.c_str());

        cl_uint ndevices;
        //program.getInfo(CL_PROGRAM_NUM_DEVICES, &ndevices);
        err = clGetProgramInfo(CloverCL::ideal_vis_uh_prog, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &ndevices, NULL);
        //printf(" ndevices in dumpBinary = %d\n", ndevices);
        
        //std::vector<size_t> sizes = std::vector<size_t>(ndevices);
        size_t sizes [ndevices];
        //program.getInfo(CL_PROGRAM_BINARY_SIZES, &sizes);
        err = clGetProgramInfo(CloverCL::ideal_vis_uh_prog, CL_PROGRAM_BINARY_SIZES, sizeof(sizes), &sizes, NULL);
        //printf("DumpBinary sizes.size() = %d, sizes[0] = %d\n", sizes.size(), sizes[0]);
        
        //std::vector<char*> binaries = std::vector<char*>(ndevices);
        char* binaries [ndevices];
        binaries[0] = new char[sizes[0]];
        //program.getInfo(CL_PROGRAM_BINARIES, &binaries);
        err = clGetProgramInfo(CloverCL::ideal_vis_uh_prog, CL_PROGRAM_BINARIES, sizeof(binaries), &binaries, NULL);
        
        //printf("Binary:\n%s\n", binaries[0]);
        FILE* file = fopen(binary_name.c_str(), "wb");
        fwrite(binaries[0], sizes[0], sizeof(char), file);
        fclose(file);
        
        delete binaries[0];
        
        return;
        #if 0
        std::vector<size_t> sizes
        program.getInfo(CL_PROGRAM_BINARY_SIZES);
        assert(sizes.size() == 1);
        printf(" sizes.size() = %ld, sizes[0] = %ld\n", sizes.size(), sizes[0]);
        std::vector<unsigned char*> binaries = program.getInfo<CL_PROGRAM_BINARIES>();
        assert(binaries.size() == 1);
        for (int i = 0; i < binaries.size(); i++) {
            printf("%c", binaries[0][i]);
        }
        #endif

    if (err != CL_SUCCESS) {
        reportError(err, "Dumping Binary");
    }
    
}
