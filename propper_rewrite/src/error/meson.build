
error_incdir = include_directories('.')
error_library = static_library(
  'error',
  files([
    'error.c',
  ]),
)

error_dep = declare_dependency(
  link_with: error_library,
  include_directories: error_incdir,
)

# end of file