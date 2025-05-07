project(
  'test', 
  'c',
)

opencl_sdk = 'E:/opt/OpenCL-SDK/git'

executable(
  'cltest', 
  [
    'main.c',
  ],
  include_directories : include_directories([opencl_sdk+'/include']),
  link_args : [
    '/LIBPATH:'+opencl_sdk+'/lib',
    'OpenCL.lib',
  ],
)

# end of file