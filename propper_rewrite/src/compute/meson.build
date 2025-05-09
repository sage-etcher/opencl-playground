
compute_incdir = include_directories('.')
compute_library = static_library(
  'compute',
  files([
    'compute.c',
    'ftypes.c',
  ]),
  dependencies : [
    opencl_dep,
  ],
)

compute_dep = declare_dependency(
  link_with: compute_library,
  include_directories: compute_incdir,
  dependencies : [
    opencl_dep,
  ],
)

# end of file