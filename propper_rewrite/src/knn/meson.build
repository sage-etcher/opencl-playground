
knn_incdir = include_directories('.')
knn_library = static_library(
  'knn',
  files([
    'knn.c',
  ]),
)

knn_dep = declare_dependency(
  link_with: knn_library,
  include_directories: knn_incdir,
)

# end of file