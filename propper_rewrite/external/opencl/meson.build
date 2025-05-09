
opencl_sdk_path = 'E:/opt/OpenCL-SDK/git'
opencl_lib_path = opencl_sdk_path + '/lib'
opencl_inc_path = opencl_sdk_path + '/include'

opencl_dep = declare_dependency(
  dependencies : cc.find_library('OpenCL', dirs : opencl_lib_path),
  include_directories : include_directories(opencl_inc_path),
)

# end of fiel