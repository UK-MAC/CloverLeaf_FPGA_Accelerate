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
 *  @brief CloverCL static class header file.
 *  @author Andrew Mallinson, David Beckingsale
 *  @details Contains common functionality required by all OCL kernels 
*/

#ifndef CLOVER_CL_H_
#define CLOVER_CL_H_

#define __CL_ENABLE_EXCEPTIONS
#include <CL/opencl.h>

#include <string>
#include <vector>

/** 
 * @class CloverCL
 *
 * Class to wrap OpenCL functions, providing consistent access to devices,
 * contexts, kernels etc. throughout the code.
 */
class CloverCL { 
    public: 
        CloverCL(); 
        virtual ~CloverCL ();

        static int const fixed_wg_min_size_large_dim   = WG_SIZE_X; // x value passed in by preprocessor 
        static int const fixed_wg_min_size_small_dim   = WG_SIZE_Y; // y value passed in by preprocessor 
        static int xmax_plusfour_rounded;
        static int xmax_plusfive_rounded;
        static int ymax_plusfour_rounded;
        static int ymax_plusfive_rounded;

        static bool initialised;

        static cl_platform_id platform_c;
        static cl_context context_c;
        static cl_device_id device_c;
        static cl_device_id* devices_list;
        static cl_command_queue queue_c;
        static cl_command_queue outoforder_queue_c;

        //static cl_program program_c;

        static cl_program ideal_gas_prog;
        static cl_program accelerate_prog;                  
        static cl_program field_summary_prog;      
        static cl_program flux_calc_prog;          
        static cl_program reset_field_prog;
        static cl_program revert_prog;  
        static cl_program viscosity_prog;
        static cl_program calc_dt_prog;                   
        static cl_program pdv_prog;                
        static cl_program initialise_chunk_prog;   
        static cl_program min_reduction_prog;      
        static cl_program sum_reduction_prog;
        static cl_program update_halo_prog; 
        static cl_program generate_chunk_prog;     
       
        static cl_program advec_cell_knl_xdir_sec1_sweep1_prog; 
        static cl_program advec_cell_knl_xdir_sec1_sweep2_prog; 
        static cl_program advec_cell_knl_xdir_sec2_prog;        
        static cl_program advec_cell_knl_xdir_sec3_prog;        
        static cl_program advec_cell_knl_y_sec1_sweep1_prog;    
        static cl_program advec_cell_knl_y_sec1_sweep2_prog;     
        static cl_program advec_cell_knl_y_sec2_prog;            
        static cl_program advec_cell_knl_y_sec3_prog;            
        static cl_program advec_mom_knl_vol_prog;             
        static cl_program advec_mom_knl_node_x_prog;          
        static cl_program advec_mom_knl_node_y_prog;          
        static cl_program advec_mom_knl_node_mass_pre_x_prog; 
        static cl_program advec_mom_knl_node_mass_pre_y_prog; 
        static cl_program advec_mom_knl_mom_flux_x_vec1_prog;    
        static cl_program advec_mom_knl_mom_flux_x_notvec1_prog; 
        static cl_program advec_mom_knl_mom_flux_y_vec1_prog;    
        static cl_program advec_mom_knl_mom_flux_y_notvec1_prog; 
        static cl_program advec_mom_knl_vel_x_prog;           
        static cl_program advec_mom_knl_vel_y_prog;           
        
        static cl_program pack_comms_buffers_prog; 
        static cl_program unpack_comms_buffers_prog;
        static cl_program read_comm_buffers_prog; 
        static cl_program write_comm_buffers_prog;

        static int const chunk_left   = 1;
        static int const chunk_right  = 2;
        static int const chunk_bottom = 3;
        static int const chunk_top    = 4;
        static int const external_face=-1;

        static int const field_density0   = 1;
        static int const field_density1   = 2;
        static int const field_energy0    = 3;
        static int const field_energy1    = 4;
        static int const field_pressure   = 5;
        static int const field_viscosity  = 6;
        static int const field_soundspeed = 7;
        static int const field_xvel0      = 8;
        static int const field_xvel1      = 9;
        static int const field_yvel0      =10;
        static int const field_yvel1      =11;
        static int const field_vol_flux_x =12;
        static int const field_vol_flux_y =13;
        static int const field_mass_flux_x=14;
        static int const field_mass_flux_y=15;
        static int const num_fields       =15;

        static const double NS_TO_SECONDS = 1e-9;
        static const double US_TO_SECONDS = 1e-6;

        static const int FIELD_DENSITY0   = 1;
        static const int FIELD_DENSITY1   = 2;
        static const int FIELD_ENERGY0    = 3;
        static const int FIELD_ENERGY1    = 4;
        static const int FIELD_PRESSURE   = 5;
        static const int FIELD_VISCOSITY  = 6;
        static const int FIELD_SOUNDSPEED = 7;
        static const int FIELD_XVEL0      = 8;
        static const int FIELD_XVEL1      = 9;
        static const int FIELD_YVEL0      =10;
        static const int FIELD_YVEL1      =11;
        static const int FIELD_VOL_FLUX_X =12;
        static const int FIELD_VOL_FLUX_Y =13;
        static const int FIELD_MASS_FLUX_X=14;
        static const int FIELD_MASS_FLUX_Y=15;

        static size_t device_prefer_wg_multiple;
        static cl_uint native_wg_multiple;
        static size_t prefer_wg_multiple;
        static size_t max_reduction_wg_size;
        static cl_uint device_procs;
        static size_t device_max_wg_size;
        static cl_ulong device_local_mem_size;
        static cl_device_type device_type;

        static int number_of_red_levels;
        static cl_event last_event;

        static int mpi_rank; 
        static int xmax_c;
        static int ymax_c;

        static void init(
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
                double dtdiv_safe);

        static void determineWorkGroupSizeInfo();
        static void calculateKernelLaunchParams(int x_max, int y_max);

        static void allocateReductionInterBuffers();

        static void allocateLocalMemoryObjects();

        static void calculateReductionStructure( int xmax, int ymax);

        static void build_reduction_kernel_objects();

        static void printDeviceInformation();

        static void initPlatform(std::string name);

        static void initContext(std::string preferred_type);

        static void initDevice(int id);

        static void initCommandQueue();

        static void loadProgram(int xmin, int xmax, int ymin, int ymax);
        static void build_one_program(int xmin, int xmax, int ymin, int ymax, std::string filename, cl_program* prog);

        static void createKernelObjects(); 

        static void createBuffers( int x_max, int y_max, int num_states);

        static void checkErr( cl_int err, std::string name);

        static void reportError( cl_int err, std::string message);

        static void readVisualisationBuffers(
                int x_max,
                int y_max,
                double* vertexx,
                double* vertexy,
                double* density0,
                double* energy0,
                double* pressure,
                double* viscosity,
                double* xvel0,
                double* yvel0);

        static void readCommunicationBuffer(
                int* xmin,
                int* xmax,
                int* ymin,
                int* ymax,
                int* depth,
                int* xinc,
                int* yinc,
                int* field_name,
                double* buffer,
                int* which_edge);

        static void writeCommunicationBuffer(
                int* xmin,
                int* xmax,
                int* ymin,
                int* ymax,
                int* depth,
                int* xinc,
                int* yinc,
                int* field_name,
                double* buffer,
                int* which_edge);

        static void readAllCommunicationBuffers(
                int* x_max,
                int* y_max,
                double* denisty0,
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
                double* mass_flux_y);

        static void writeAllCommunicationBuffers(
                int* x_max,
                int* y_max,
                double* denisty0,
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
                double* mass_flux_y);

        static void enqueueKernel_nooffsets(cl_kernel kernel, int num_x, int num_y);

        static void enqueueKernel(cl_kernel kernel, int x_min, int x_max, int y_min, int y_max);

        static void enqueueKernel(cl_kernel kernel, int min, int max);

        static void initialiseKernelArgs(
                int x_min,
                int x_max,
                int y_min,
                int y_max,
                double g_small,
                double g_big,
                double dtmin,
                double dtc_safe,
                double dtu_safe,
                double dtv_safe,
                double dtdiv_safe);

        static std::string errToString(cl_int err);

        static void read_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume ); 

        static void write_back_all_ocl_buffers(double* density0, double* density1, double* energy0, double* energy1,
                                         double* pressure, double* viscosity, double* soundspeed,
                                         double* xvel0, double* xvel1, double* yvel0, double* yvel1,
                                         double* vol_flux_x, double* mass_flux_x,
                                         double* vol_flux_y, double* mass_flux_y,
                                         double* celldx, double* celldy, double* volume ); 

        static void dumpBinary();

        static cl_mem density0_buffer_c;
        static cl_mem density1_buffer_c;
        static cl_mem energy0_buffer_c;
        static cl_mem energy1_buffer_c;
        static cl_mem pressure_buffer_c;
        static cl_mem soundspeed_buffer_c;
        static cl_mem celldx_buffer_c;
        static cl_mem celldy_buffer_c;
        static cl_mem viscosity_buffer_c;
        static cl_mem xvel0_buffer_c;
        static cl_mem yvel0_buffer_c;
        static cl_mem xvel1_buffer_c;
        static cl_mem yvel1_buffer_c;
        static cl_mem xarea_buffer_c;
        static cl_mem yarea_buffer_c;
        static cl_mem vol_flux_x_buffer_c;
        static cl_mem vol_flux_y_buffer_c;
        static cl_mem mass_flux_x_buffer_c;
        static cl_mem mass_flux_y_buffer_c;
        static cl_mem stepbymass_buffer_c;
        static cl_mem volume_buffer_c;

        static cl_mem vol_tmp_buffer_c;
        static cl_mem mass_tmp_buffer_c;
        static cl_mem ie_tmp_buffer_c;
        static cl_mem ke_tmp_buffer_c;
        static cl_mem press_tmp_buffer_c;

        static cl_mem node_flux_buffer_c;
        static cl_mem node_mass_post_buffer_c;
        static cl_mem node_mass_pre_buffer_c;
        static cl_mem advec_vel_buffer_c;
        static cl_mem mom_flux_buffer_c;
        static cl_mem pre_vol_buffer_c;
        static cl_mem post_vol_buffer_c;

        static cl_mem vertexdx_buffer_c;
        static cl_mem vertexx_buffer_c;
        static cl_mem vertexdy_buffer_c;
        static cl_mem vertexy_buffer_c;
        static cl_mem cellx_buffer_c;
        static cl_mem celly_buffer_c;

        static cl_mem dt_min_val_array_buffer_c;
        static cl_mem dt_min_val_buffer_c;

        static cl_mem vol_sum_val_buffer_c;
        static cl_mem mass_sum_val_buffer_c;
        static cl_mem ie_sum_val_buffer_c;
        static cl_mem ke_sum_val_buffer_c;
        static cl_mem press_sum_val_buffer_c;

        static cl_mem pre_mass_buffer_c;
        static cl_mem post_mass_buffer_c;
        static cl_mem advec_vol_buffer_c;
        static cl_mem post_ener_buffer_c;
        static cl_mem ener_flux_buffer_c;

        static cl_mem state_density_buffer_c;
        static cl_mem state_energy_buffer_c;
        static cl_mem state_xvel_buffer_c;
        static cl_mem state_yvel_buffer_c;
        static cl_mem state_xmin_buffer_c;
        static cl_mem state_xmax_buffer_c;
        static cl_mem state_ymin_buffer_c;
        static cl_mem state_ymax_buffer_c;
        static cl_mem state_radius_buffer_c;
        static cl_mem state_geometry_buffer_c;

        static cl_mem top_send_buffer_c;
        static cl_mem top_recv_buffer_c;
        static cl_mem bottom_send_buffer_c;
        static cl_mem bottom_recv_buffer_c;
        static cl_mem left_send_buffer_c;
        static cl_mem left_recv_buffer_c;
        static cl_mem right_send_buffer_c;
        static cl_mem right_recv_buffer_c;

        static cl_mem cpu_min_red_buffer_c;
        static cl_mem cpu_vol_red_buffer_c;
        static cl_mem cpu_mass_red_buffer_c;
        static cl_mem cpu_ie_red_buffer_c;
        static cl_mem cpu_ke_red_buffer_c;
        static cl_mem cpu_press_red_buffer_c;

        static cl_kernel ideal_gas_predict_knl_c;
        static cl_kernel ideal_gas_NO_predict_knl_c;
        static cl_kernel viscosity_knl_c;
        static cl_kernel flux_calc_knl_c;
        static cl_kernel accelerate_knl_c;
        static cl_kernel advec_mom_vol_knl_c;
        static cl_kernel advec_mom_node_x_knl_c;
        static cl_kernel advec_mom_node_mass_pre_x_knl_c;
        static cl_kernel advec_mom_flux_x_vec1_knl_c;
        static cl_kernel advec_mom_flux_x_vecnot1_knl_c;
        static cl_kernel advec_mom_vel_x_knl_c;
        static cl_kernel advec_mom_node_y_knl_c;
        static cl_kernel advec_mom_node_mass_pre_y_knl_c;
        static cl_kernel advec_mom_flux_y_vec1_knl_c;
        static cl_kernel advec_mom_flux_y_vecnot1_knl_c;
        static cl_kernel advec_mom_vel_y_knl_c;
        static cl_kernel dt_calc_knl_c;
        static cl_kernel advec_cell_xdir_sec1_s1_knl_c;
        static cl_kernel advec_cell_xdir_sec1_s2_knl_c;
        static cl_kernel advec_cell_xdir_sec2_knl_c;
        static cl_kernel advec_cell_xdir_sec3_knl_c;
        static cl_kernel advec_cell_ydir_sec1_s1_knl_c;
        static cl_kernel advec_cell_ydir_sec1_s2_knl_c;
        static cl_kernel advec_cell_ydir_sec2_knl_c;
        static cl_kernel advec_cell_ydir_sec3_knl_c;
        static cl_kernel pdv_correct_knl_c;
        static cl_kernel pdv_predict_knl_c;
        static cl_kernel reset_field_knl_c;
        static cl_kernel revert_knl_c;
        static cl_kernel generate_chunk_knl_c;
        static cl_kernel initialise_chunk_cell_x_knl_c;
        static cl_kernel initialise_chunk_cell_y_knl_c;
        static cl_kernel initialise_chunk_vertex_x_knl_c;
        static cl_kernel initialise_chunk_vertex_y_knl_c;
        static cl_kernel initialise_chunk_volume_area_knl_c;
        static cl_kernel field_summary_knl_c;
        static cl_kernel update_halo_left_cell_knl_c;
        static cl_kernel update_halo_right_cell_knl_c;
        static cl_kernel update_halo_top_cell_knl_c;
        static cl_kernel update_halo_bottom_cell_knl_c;
        static cl_kernel update_halo_left_vel_knl_c;
        static cl_kernel update_halo_right_vel_knl_c;
        static cl_kernel update_halo_top_vel_knl_c;
        static cl_kernel update_halo_bottom_vel_knl_c;
        static cl_kernel update_halo_left_flux_x_knl_c;
        static cl_kernel update_halo_right_flux_x_knl_c;
        static cl_kernel update_halo_top_flux_x_knl_c;
        static cl_kernel update_halo_bottom_flux_x_knl_c;
        static cl_kernel update_halo_left_flux_y_knl_c;
        static cl_kernel update_halo_right_flux_y_knl_c;
        static cl_kernel update_halo_top_flux_y_knl_c;
        static cl_kernel update_halo_bottom_flux_y_knl_c;

        static cl_kernel read_top_buffer_knl_c;
        static cl_kernel read_right_buffer_knl_c;
        static cl_kernel read_bottom_buffer_knl_c;
        static cl_kernel read_left_buffer_knl_c;
        static cl_kernel write_top_buffer_knl_c;
        static cl_kernel write_right_buffer_knl_c;
        static cl_kernel write_bottom_buffer_knl_c;
        static cl_kernel write_left_buffer_knl_c;

        static std::vector<cl_kernel> min_reduction_kernels;
        static std::vector<cl_kernel> vol_sum_reduction_kernels;
        static std::vector<cl_kernel> mass_sum_reduction_kernels;
        static std::vector<cl_kernel> ie_sum_reduction_kernels;
        static std::vector<cl_kernel> ke_sum_reduction_kernels;
        static std::vector<cl_kernel> press_sum_reduction_kernels;

        static std::vector<size_t> num_workitems_tolaunch;
        static std::vector<size_t> num_workitems_per_wg;
        static std::vector<int> local_mem_size;
        static std::vector<int> size_limits;
        static std::vector<int> buffer_sizes;
        static std::vector<bool> input_even;
        static std::vector<int> num_elements_per_wi;

        static std::vector<cl_mem> min_interBuffers;
        static std::vector<cl_mem> vol_interBuffers;
        static std::vector<cl_mem> mass_interBuffers;
        static std::vector<cl_mem> ie_interBuffers;
        static std::vector<cl_mem> ke_interBuffers;
        static std::vector<cl_mem> press_interBuffers;

        static std::vector<int> min_local_memory_objects;
        static std::vector<int> vol_local_memory_objects;
        static std::vector<int> mass_local_memory_objects;
        static std::vector<int> ie_local_memory_objects;
        static std::vector<int> ke_local_memory_objects;
        static std::vector<int> press_local_memory_objects;

    private:
        //static std::vector<cl::Event> global_events;
};

#endif
