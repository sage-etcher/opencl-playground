
csv_incdir = include_directories('.')
csv_library = static_library(
  'csv',
  files([
    'csv.c',
  ]),
)

csv_dep = declare_dependency(
  link_with: csv_library,
  include_directories: csv_incdir,
)

# end of file