
qsort_incdir = include_directories('.')
qsort_library = static_library(
  'qsort',
  files([
    'qsort.c',
  ]),
)

qsort_dep = declare_dependency(
  link_with: qsort_library,
  include_directories: qsort_incdir,
)

# end of file