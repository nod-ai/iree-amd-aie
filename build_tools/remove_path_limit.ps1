# https://stackoverflow.com/a/68353105
# Cannot open compiler generated file: '': Invalid argument
# -DCMAKE_OBJECT_PATH_MAX=4096
# The registry value will be cached by the system (per process) after the first call to an affected Win32 file or directory function (see below for the list of functions). The registry value will not be reloaded during the lifetime of the process. In order for all apps on the system to recognize the value, a reboot might be required because some processes may have started before the key was set.
# https://github.com/ninja-build/ninja/pull/2225 (ninja 1.12)
# CMake 3.30.2
# https://developercommunity.visualstudio.com/t/compiler-cant-find-source-file-in-path-/10221576
# https://gitlab.kitware.com/cmake/cmake/-/issues/25936
# https://learn.microsoft.com/en-us/visualstudio/releases/2019/history#release-dates-and-build-numbers
# to solve stddef.h problem ^
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force