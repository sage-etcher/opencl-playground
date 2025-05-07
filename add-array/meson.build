project(
  'test', 
  'c',
)

opencl_sdk = 'E:/opt/OpenCL-SDK/git'

executable(
  'cltest', 
  [
    'main.c',
    'compute.c',
  ],
  include_directories : include_directories([opencl_sdk+'/include']),
  c_args : [
  #  '/DNDEBUG',
  ],
  link_args : [
    '/LIBPATH:'+opencl_sdk+'/lib',
    'OpenCL.lib',
  ],
)

# end of file