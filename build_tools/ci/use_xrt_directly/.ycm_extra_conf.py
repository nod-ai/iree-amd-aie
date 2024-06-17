def Settings( **kwargs ):
  return {
    'flags': [ '-x', 'c++',
              '-Wall',
              '-I/opt/xilinx/xrt/include',
              '-L/opt/xilinx/xrt/lib',
              '-luuid',
              '-lxrt_coreutil',
              '-lrt',
              '-lstdc++',
            ]
  }
