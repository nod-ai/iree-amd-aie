# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# https://stackoverflow.com/a/49216539/9045206
function(remove_flag_from_target _target _flag)
  get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
  if(_target_cxx_flags)
    list(REMOVE_ITEM _target_cxx_flags ${_flag})
    set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
  endif()
endfunction()

function(replace_string_in_file _file _match_string _replace_string)
  if(NOT (EXISTS ${_file}))
    message(FATAL_ERROR "file ${_file} does not exist")
  endif()
  file(READ "${_file}" _file_contents)
  if(_file_contents STREQUAL "")
    message(FATAL_ERROR "empty file contents for ${_file}")
  endif()
  string(REPLACE "${_match_string}" "${_replace_string}" _file_contents "${_file_contents}")
  if(_file_contents STREQUAL "")
    message(FATAL_ERROR "empty replacement contents for ${_file}")
  endif()
  file(WRITE "${_file}" "${_file_contents}")
endfunction()

