#define ENQUEUE_KERNEL_OOO_MACRO(kernel,x_num,y_num,x_wg_size,y_wg_size) \
    global_wi[0] = x_num; \
    global_wi[1] = y_num; \
    local_wi[0] = x_wg_size; \
    local_wi[1] = y_wg_size; \
    clEnqueueNDRangeKernel(CloverCL::outoforder_queue_c, kernel, 2, NULL, \
                           global_wi, local_wi, 0, NULL, NULL);



