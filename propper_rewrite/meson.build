
project(
    'apptest', 'c',
    default_options : [
        'c_std=c89',
        'warning_level=3',
    ],
)

cc = meson.get_compiler('c')

# subprojects
inih_proj = subproject(
  'inih',
  default_options : [
    'with_INIReader=false',
    'with_test_suite=false',
  ],
)
inih_dep = inih_proj.get_variable('inih_dep')

# non meson external deps
subdir('external')

# project
apptest_datadir = get_option('datadir') / 'apptest'
conf_data = configuration_data()
conf_data.set('datadir', apptest_datadir)

subdir('resources')
subdir('assets')
subdir('src')