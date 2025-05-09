
configure_file(
  input  : 'config.h.in',
  output : 'config.h',
  configuration : conf_data,
)

apptest = executable(
  'apptest',
  files([
    'config_ini.c',
    'main.c',
  ]),
  dependencies : [
    compute_dep,
    error_dep,
    inih_dep,
    portability_dep,
  ],
  install : true,
)

# end of file
