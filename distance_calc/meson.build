project(
  'test', 
  'c',
)

opencl_sdk = 'E:/opt/OpenCL-SDK/git'

executable(
  'distcalc', 
  [
    'main.c',
  ],
  include_directories : include_directories([opencl_sdk+'/include']),
  c_args : [
    '/DCL_TARGET_OPENCL_VERSION=300',
    #'/DNDEBUG',
    #'/DDEBUG_DEFAULT_SHADER',
  ],
  link_args : [
    '/LIBPATH:'+opencl_sdk+'/lib',
    'OpenCL.lib',
  ],
)

# end of file