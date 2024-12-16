# generated using clang2py amdxdna_accel.h -o amdxdna_accel.py -k cdefstum
import ctypes


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith("PADDING_"):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):
    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, "_fields_"):
            return (f[0] for f in cls._fields_ if not f[0].startswith("PADDING"))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = type_(
                            (lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]
                            )
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_(
                        (lambda default_: lambda *args: default_)(default_)
                    )
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
                )
            )
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass


AMDXDNA_ACCEL_H_ = True  # macro
AMDXDNA_DRIVER_MAJOR = 1  # macro
AMDXDNA_DRIVER_MINOR = 0  # macro
AMDXDNA_INVALID_CMD_HANDLE = ~0  # macro
AMDXDNA_INVALID_ADDR = ~0  # macro
AMDXDNA_INVALID_CTX_HANDLE = 0  # macro
AMDXDNA_INVALID_BO_HANDLE = 0  # macro
AMDXDNA_INVALID_FENCE_HANDLE = 0  # macro
SYNC_DIRECT_TO_DEVICE = 0  # macro
SYNC_DIRECT_FROM_DEVICE = 1  # macro

# values for enumeration 'amdxdna_drm_ioctl_id'
amdxdna_drm_ioctl_id__enumvalues = {
    0: "DRM_AMDXDNA_CREATE_HWCTX",
    1: "DRM_AMDXDNA_DESTROY_HWCTX",
    2: "DRM_AMDXDNA_CONFIG_HWCTX",
    3: "DRM_AMDXDNA_CREATE_BO",
    4: "DRM_AMDXDNA_GET_BO_INFO",
    5: "DRM_AMDXDNA_SYNC_BO",
    6: "DRM_AMDXDNA_EXEC_CMD",
    7: "DRM_AMDXDNA_GET_INFO",
    8: "DRM_AMDXDNA_SET_STATE",
    9: "DRM_AMDXDNA_WAIT_CMD",
    10: "DRM_AMDXDNA_SUBMIT_WAIT",
    11: "DRM_AMDXDNA_SUBMIT_SIGNAL",
    12: "DRM_AMDXDNA_NUM_IOCTLS",
}
DRM_AMDXDNA_CREATE_HWCTX = 0
DRM_AMDXDNA_DESTROY_HWCTX = 1
DRM_AMDXDNA_CONFIG_HWCTX = 2
DRM_AMDXDNA_CREATE_BO = 3
DRM_AMDXDNA_GET_BO_INFO = 4
DRM_AMDXDNA_SYNC_BO = 5
DRM_AMDXDNA_EXEC_CMD = 6
DRM_AMDXDNA_GET_INFO = 7
DRM_AMDXDNA_SET_STATE = 8
DRM_AMDXDNA_WAIT_CMD = 9
DRM_AMDXDNA_SUBMIT_WAIT = 10
DRM_AMDXDNA_SUBMIT_SIGNAL = 11
DRM_AMDXDNA_NUM_IOCTLS = 12
amdxdna_drm_ioctl_id = ctypes.c_uint32  # enum

# values for enumeration 'amdxdna_device_type'
amdxdna_device_type__enumvalues = {
    -1: "AMDXDNA_DEV_TYPE_UNKNOWN",
    0: "AMDXDNA_DEV_TYPE_KMQ",
    1: "AMDXDNA_DEV_TYPE_UMQ",
}
AMDXDNA_DEV_TYPE_UNKNOWN = -1
AMDXDNA_DEV_TYPE_KMQ = 0
AMDXDNA_DEV_TYPE_UMQ = 1
amdxdna_device_type = ctypes.c_int32  # enum


class struct_amdxdna_qos_info(Structure):
    pass


struct_amdxdna_qos_info._pack_ = 1  # source:False
struct_amdxdna_qos_info._fields_ = [
    ("gops", ctypes.c_uint32),
    ("fps", ctypes.c_uint32),
    ("dma_bandwidth", ctypes.c_uint32),
    ("latency", ctypes.c_uint32),
    ("frame_exec_time", ctypes.c_uint32),
    ("priority", ctypes.c_uint32),
]


class struct_amdxdna_drm_create_hwctx(Structure):
    pass


struct_amdxdna_drm_create_hwctx._pack_ = 1  # source:False
struct_amdxdna_drm_create_hwctx._fields_ = [
    ("ext", ctypes.c_uint64),
    ("ext_flags", ctypes.c_uint64),
    ("qos_p", ctypes.c_uint64),
    ("umq_bo", ctypes.c_uint32),
    ("log_buf_bo", ctypes.c_uint32),
    ("max_opc", ctypes.c_uint32),
    ("num_tiles", ctypes.c_uint32),
    ("mem_size", ctypes.c_uint32),
    ("umq_doorbell", ctypes.c_uint32),
    ("handle", ctypes.c_uint32),
    ("PADDING_0", ctypes.c_ubyte * 4),
]


# DRM_IOCTL_AMDXDNA_CREATE_HWCTX = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_CREATE_HWCTX , struct_amdxdna_drm_create_hwctx ) # macro
class struct_amdxdna_drm_destroy_hwctx(Structure):
    pass


struct_amdxdna_drm_destroy_hwctx._pack_ = 1  # source:False
struct_amdxdna_drm_destroy_hwctx._fields_ = [
    ("handle", ctypes.c_uint32),
    ("pad", ctypes.c_uint32),
]


# DRM_IOCTL_AMDXDNA_DESTROY_HWCTX = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_DESTROY_HWCTX , struct_amdxdna_drm_destroy_hwctx ) # macro
class struct_amdxdna_cu_config(Structure):
    pass


struct_amdxdna_cu_config._pack_ = 1  # source:False
struct_amdxdna_cu_config._fields_ = [
    ("cu_bo", ctypes.c_uint32),
    ("cu_func", ctypes.c_ubyte),
    ("pad", ctypes.c_ubyte * 3),
]


def struct_amdxdna_hwctx_param_config_cu(num_cus, cu_configs):
    assert len(cu_configs) == num_cus

    class struct_amdxdna_hwctx_param_config_cu(Structure):
        pass

    struct_amdxdna_hwctx_param_config_cu._pack_ = 1  # source:False
    struct_amdxdna_hwctx_param_config_cu._fields_ = [
        ("num_cus", ctypes.c_uint16),
        ("pad", ctypes.c_uint16 * 3),
        ("cu_configs", struct_amdxdna_cu_config * num_cus),
    ]
    struc = struct_amdxdna_hwctx_param_config_cu()
    struc.num_cus = num_cus
    struc.cu_configs = (struct_amdxdna_cu_config * num_cus)(*cu_configs)
    return struc


# values for enumeration 'amdxdna_drm_config_hwctx_param'
amdxdna_drm_config_hwctx_param__enumvalues = {
    0: "DRM_AMDXDNA_HWCTX_CONFIG_CU",
    1: "DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF",
    2: "DRM_AMDXDNA_HWCTX_REMOVE_DBG_BUF",
    3: "DRM_AMDXDNA_HWCTX_CONFIG_NUM",
}
DRM_AMDXDNA_HWCTX_CONFIG_CU = 0
DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF = 1
DRM_AMDXDNA_HWCTX_REMOVE_DBG_BUF = 2
DRM_AMDXDNA_HWCTX_CONFIG_NUM = 3
amdxdna_drm_config_hwctx_param = ctypes.c_uint32  # enum


class struct_amdxdna_drm_config_hwctx(Structure):
    pass


struct_amdxdna_drm_config_hwctx._pack_ = 1  # source:False
struct_amdxdna_drm_config_hwctx._fields_ = [
    ("handle", ctypes.c_uint32),
    ("param_type", ctypes.c_uint32),
    ("param_val", ctypes.c_uint64),
    ("param_val_size", ctypes.c_uint32),
    ("pad", ctypes.c_uint32),
]

# DRM_IOCTL_AMDXDNA_CONFIG_HWCTX = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_CONFIG_HWCTX , struct_amdxdna_drm_config_hwctx ) # macro

# values for enumeration 'amdxdna_bo_type'
amdxdna_bo_type__enumvalues = {
    0: "AMDXDNA_BO_INVALID",
    1: "AMDXDNA_BO_SHMEM",
    2: "AMDXDNA_BO_DEV_HEAP",
    3: "AMDXDNA_BO_DEV",
    4: "AMDXDNA_BO_CMD",
    5: "AMDXDNA_BO_DMA",
}
AMDXDNA_BO_INVALID = 0
AMDXDNA_BO_SHMEM = 1
AMDXDNA_BO_DEV_HEAP = 2
AMDXDNA_BO_DEV = 3
AMDXDNA_BO_CMD = 4
AMDXDNA_BO_DMA = 5
amdxdna_bo_type = ctypes.c_uint32  # enum


class struct_amdxdna_drm_create_bo(Structure):
    pass


struct_amdxdna_drm_create_bo._pack_ = 1  # source:False
struct_amdxdna_drm_create_bo._fields_ = [
    ("flags", ctypes.c_uint64),
    ("type", ctypes.c_uint32),
    ("_pad", ctypes.c_uint32),
    ("vaddr", ctypes.c_uint64),
    ("size", ctypes.c_uint64),
    ("handle", ctypes.c_uint32),
    ("PADDING_0", ctypes.c_ubyte * 4),
]


# DRM_IOCTL_AMDXDNA_CREATE_BO = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_CREATE_BO , struct_amdxdna_drm_create_bo ) # macro
class struct_amdxdna_drm_get_bo_info(Structure):
    pass


struct_amdxdna_drm_get_bo_info._pack_ = 1  # source:False
struct_amdxdna_drm_get_bo_info._fields_ = [
    ("ext", ctypes.c_uint64),
    ("ext_flags", ctypes.c_uint64),
    ("handle", ctypes.c_uint32),
    ("_pad", ctypes.c_uint32),
    ("map_offset", ctypes.c_uint64),
    ("vaddr", ctypes.c_uint64),
    ("xdna_addr", ctypes.c_uint64),
]


# DRM_IOCTL_AMDXDNA_GET_BO_INFO = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_GET_BO_INFO , struct_amdxdna_drm_get_bo_info ) # macro
class struct_amdxdna_drm_sync_bo(Structure):
    pass


struct_amdxdna_drm_sync_bo._pack_ = 1  # source:False
struct_amdxdna_drm_sync_bo._fields_ = [
    ("handle", ctypes.c_uint32),
    ("direction", ctypes.c_uint32),
    ("offset", ctypes.c_uint64),
    ("size", ctypes.c_uint64),
]

# DRM_IOCTL_AMDXDNA_SYNC_BO = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_SYNC_BO , struct_amdxdna_drm_sync_bo ) # macro

# values for enumeration 'amdxdna_cmd_type'
amdxdna_cmd_type__enumvalues = {
    0: "AMDXDNA_CMD_SUBMIT_EXEC_BUF",
    1: "AMDXDNA_CMD_SUBMIT_DEPENDENCY",
    2: "AMDXDNA_CMD_SUBMIT_SIGNAL",
}
AMDXDNA_CMD_SUBMIT_EXEC_BUF = 0
AMDXDNA_CMD_SUBMIT_DEPENDENCY = 1
AMDXDNA_CMD_SUBMIT_SIGNAL = 2
amdxdna_cmd_type = ctypes.c_uint32  # enum


class struct_amdxdna_drm_exec_cmd(Structure):
    pass


struct_amdxdna_drm_exec_cmd._pack_ = 1  # source:False
struct_amdxdna_drm_exec_cmd._fields_ = [
    ("ext", ctypes.c_uint64),
    ("ext_flags", ctypes.c_uint64),
    ("hwctx", ctypes.c_uint32),
    ("type", ctypes.c_uint32),
    ("cmd_handles", ctypes.c_uint64),
    ("args", ctypes.c_uint64),
    ("cmd_count", ctypes.c_uint32),
    ("arg_count", ctypes.c_uint32),
    ("seq", ctypes.c_uint64),
]


# DRM_IOCTL_AMDXDNA_EXEC_CMD = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_EXEC_CMD , struct_amdxdna_drm_exec_cmd ) # macro
class struct_amdxdna_drm_wait_cmd(Structure):
    pass


struct_amdxdna_drm_wait_cmd._pack_ = 1  # source:False
struct_amdxdna_drm_wait_cmd._fields_ = [
    ("hwctx", ctypes.c_uint32),
    ("timeout", ctypes.c_uint32),
    ("seq", ctypes.c_uint64),
]


# DRM_IOCTL_AMDXDNA_WAIT_CMD = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_WAIT_CMD , struct_amdxdna_drm_wait_cmd ) # macro
class struct_amdxdna_drm_query_aie_status(Structure):
    pass


struct_amdxdna_drm_query_aie_status._pack_ = 1  # source:False
struct_amdxdna_drm_query_aie_status._fields_ = [
    ("buffer", ctypes.c_uint64),
    ("buffer_size", ctypes.c_uint32),
    ("cols_filled", ctypes.c_uint32),
]


class struct_amdxdna_drm_query_aie_version(Structure):
    pass


struct_amdxdna_drm_query_aie_version._pack_ = 1  # source:False
struct_amdxdna_drm_query_aie_version._fields_ = [
    ("major", ctypes.c_uint32),
    ("minor", ctypes.c_uint32),
]


class struct_amdxdna_drm_query_aie_tile_metadata(Structure):
    pass


struct_amdxdna_drm_query_aie_tile_metadata._pack_ = 1  # source:False
struct_amdxdna_drm_query_aie_tile_metadata._fields_ = [
    ("row_count", ctypes.c_uint16),
    ("row_start", ctypes.c_uint16),
    ("dma_channel_count", ctypes.c_uint16),
    ("lock_count", ctypes.c_uint16),
    ("event_reg_count", ctypes.c_uint16),
    ("pad", ctypes.c_uint16 * 3),
]


class struct_amdxdna_drm_query_aie_metadata(Structure):
    pass


struct_amdxdna_drm_query_aie_metadata._pack_ = 1  # source:False
struct_amdxdna_drm_query_aie_metadata._fields_ = [
    ("col_size", ctypes.c_uint32),
    ("cols", ctypes.c_uint16),
    ("rows", ctypes.c_uint16),
    ("version", struct_amdxdna_drm_query_aie_version),
    ("core", struct_amdxdna_drm_query_aie_tile_metadata),
    ("mem", struct_amdxdna_drm_query_aie_tile_metadata),
    ("shim", struct_amdxdna_drm_query_aie_tile_metadata),
]


class struct_amdxdna_drm_query_clock(Structure):
    pass


struct_amdxdna_drm_query_clock._pack_ = 1  # source:False
struct_amdxdna_drm_query_clock._fields_ = [
    ("name", ctypes.c_ubyte * 16),
    ("freq_mhz", ctypes.c_uint32),
    ("pad", ctypes.c_uint32),
]


class struct_amdxdna_drm_query_clock_metadata(Structure):
    _pack_ = 1  # source:False
    _fields_ = [
        ("mp_npu_clock", struct_amdxdna_drm_query_clock),
        ("h_clock", struct_amdxdna_drm_query_clock),
    ]


# values for enumeration 'amdxdna_sensor_type'
amdxdna_sensor_type__enumvalues = {
    0: "AMDXDNA_SENSOR_TYPE_POWER",
}
AMDXDNA_SENSOR_TYPE_POWER = 0
amdxdna_sensor_type = ctypes.c_uint32  # enum


class struct_amdxdna_drm_query_sensor(Structure):
    pass


struct_amdxdna_drm_query_sensor._pack_ = 1  # source:False
struct_amdxdna_drm_query_sensor._fields_ = [
    ("label", ctypes.c_ubyte * 64),
    ("input", ctypes.c_uint32),
    ("max", ctypes.c_uint32),
    ("average", ctypes.c_uint32),
    ("highest", ctypes.c_uint32),
    ("status", ctypes.c_ubyte * 64),
    ("units", ctypes.c_ubyte * 16),
    ("unitm", ctypes.c_byte),
    ("type", ctypes.c_ubyte),
    ("pad", ctypes.c_ubyte * 6),
]


class struct_amdxdna_drm_query_hwctx(Structure):
    pass


struct_amdxdna_drm_query_hwctx._pack_ = 1  # source:False
struct_amdxdna_drm_query_hwctx._fields_ = [
    ("context_id", ctypes.c_uint32),
    ("start_col", ctypes.c_uint32),
    ("num_col", ctypes.c_uint32),
    ("pad", ctypes.c_uint32),
    ("pid", ctypes.c_int64),
    ("command_submissions", ctypes.c_uint64),
    ("command_completions", ctypes.c_uint64),
    ("migrations", ctypes.c_uint64),
    ("preemptions", ctypes.c_uint64),
    ("errors", ctypes.c_uint64),
]


class struct_amdxdna_drm_aie_mem(Structure):
    pass


struct_amdxdna_drm_aie_mem._pack_ = 1  # source:False
struct_amdxdna_drm_aie_mem._fields_ = [
    ("col", ctypes.c_uint32),
    ("row", ctypes.c_uint32),
    ("addr", ctypes.c_uint32),
    ("size", ctypes.c_uint32),
    ("buf_p", ctypes.c_uint64),
]


class struct_amdxdna_drm_aie_reg(Structure):
    pass


struct_amdxdna_drm_aie_reg._pack_ = 1  # source:False
struct_amdxdna_drm_aie_reg._fields_ = [
    ("col", ctypes.c_uint32),
    ("row", ctypes.c_uint32),
    ("addr", ctypes.c_uint32),
    ("val", ctypes.c_uint32),
]

# values for enumeration 'amdxdna_power_mode_type'
amdxdna_power_mode_type__enumvalues = {
    0: "POWER_MODE_DEFAULT",
    1: "POWER_MODE_LOW",
    2: "POWER_MODE_MEDIUM",
    3: "POWER_MODE_HIGH",
}
POWER_MODE_DEFAULT = 0
POWER_MODE_LOW = 1
POWER_MODE_MEDIUM = 2
POWER_MODE_HIGH = 3
amdxdna_power_mode_type = ctypes.c_uint32  # enum


class struct_amdxdna_drm_get_power_mode(Structure):
    pass


struct_amdxdna_drm_get_power_mode._pack_ = 1  # source:False
struct_amdxdna_drm_get_power_mode._fields_ = [
    ("power_mode", ctypes.c_ubyte),
    ("pad", ctypes.c_ubyte * 7),
]


class struct_amdxdna_drm_query_firmware_version(Structure):
    pass


struct_amdxdna_drm_query_firmware_version._pack_ = 1  # source:False
struct_amdxdna_drm_query_firmware_version._fields_ = [
    ("major", ctypes.c_uint32),
    ("minor", ctypes.c_uint32),
    ("patch", ctypes.c_uint32),
    ("build", ctypes.c_uint32),
]

# values for enumeration 'amdxdna_drm_get_param'
amdxdna_drm_get_param__enumvalues = {
    0: "DRM_AMDXDNA_QUERY_AIE_STATUS",
    1: "DRM_AMDXDNA_QUERY_AIE_METADATA",
    2: "DRM_AMDXDNA_QUERY_AIE_VERSION",
    3: "DRM_AMDXDNA_QUERY_CLOCK_METADATA",
    4: "DRM_AMDXDNA_QUERY_SENSORS",
    5: "DRM_AMDXDNA_QUERY_HW_CONTEXTS",
    6: "DRM_AMDXDNA_READ_AIE_MEM",
    7: "DRM_AMDXDNA_READ_AIE_REG",
    8: "DRM_AMDXDNA_QUERY_FIRMWARE_VERSION",
    9: "DRM_AMDXDNA_GET_POWER_MODE",
    10: "DRM_AMDXDNA_NUM_GET_PARAM",
}
DRM_AMDXDNA_QUERY_AIE_STATUS = 0
DRM_AMDXDNA_QUERY_AIE_METADATA = 1
DRM_AMDXDNA_QUERY_AIE_VERSION = 2
DRM_AMDXDNA_QUERY_CLOCK_METADATA = 3
DRM_AMDXDNA_QUERY_SENSORS = 4
DRM_AMDXDNA_QUERY_HW_CONTEXTS = 5
DRM_AMDXDNA_READ_AIE_MEM = 6
DRM_AMDXDNA_READ_AIE_REG = 7
DRM_AMDXDNA_QUERY_FIRMWARE_VERSION = 8
DRM_AMDXDNA_GET_POWER_MODE = 9
DRM_AMDXDNA_NUM_GET_PARAM = 10
amdxdna_drm_get_param = ctypes.c_uint32  # enum


class struct_amdxdna_drm_get_info(Structure):
    pass


struct_amdxdna_drm_get_info._pack_ = 1  # source:False
struct_amdxdna_drm_get_info._fields_ = [
    ("param", ctypes.c_uint32),
    ("buffer_size", ctypes.c_uint32),
    ("buffer", ctypes.c_uint64),
]


# DRM_IOCTL_AMDXDNA_GET_INFO = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_GET_INFO , struct_amdxdna_drm_get_info ) # macro
class struct_amdxdna_drm_set_power_mode(Structure):
    pass


struct_amdxdna_drm_set_power_mode._pack_ = 1  # source:False
struct_amdxdna_drm_set_power_mode._fields_ = [
    ("power_mode", ctypes.c_ubyte),
    ("pad", ctypes.c_ubyte * 7),
]

# values for enumeration 'amdxdna_drm_set_param'
amdxdna_drm_set_param__enumvalues = {
    0: "DRM_AMDXDNA_SET_POWER_MODE",
    1: "DRM_AMDXDNA_WRITE_AIE_MEM",
    2: "DRM_AMDXDNA_WRITE_AIE_REG",
    3: "DRM_AMDXDNA_NUM_SET_PARAM",
}
DRM_AMDXDNA_SET_POWER_MODE = 0
DRM_AMDXDNA_WRITE_AIE_MEM = 1
DRM_AMDXDNA_WRITE_AIE_REG = 2
DRM_AMDXDNA_NUM_SET_PARAM = 3
amdxdna_drm_set_param = ctypes.c_uint32  # enum


class struct_amdxdna_drm_set_state(Structure):
    pass


struct_amdxdna_drm_set_state._pack_ = 1  # source:False
struct_amdxdna_drm_set_state._fields_ = [
    ("param", ctypes.c_uint32),
    ("buffer_size", ctypes.c_uint32),
    ("buffer", ctypes.c_uint64),
]


# DRM_IOCTL_AMDXDNA_SET_STATE = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_SET_STATE , struct_amdxdna_drm_set_state ) # macro
class struct_amdxdna_drm_syncobjs(Structure):
    pass


struct_amdxdna_drm_syncobjs._pack_ = 1  # source:False
struct_amdxdna_drm_syncobjs._fields_ = [
    ("handles", ctypes.c_uint64),
    ("points", ctypes.c_uint64),
    ("count", ctypes.c_uint32),
    ("pad", ctypes.c_uint32),
]


def struct_amdxdna_cmd_chain(command_count):
    class struct_amdxdna_cmd_chain(Structure):
        pass

    struct_amdxdna_cmd_chain._pack_ = 1  # source:False
    struct_amdxdna_cmd_chain._fields_ = [
        ("command_count", ctypes.c_uint32),
        ("submit_index", ctypes.c_uint32),
        ("error_index", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
        ("data", ctypes.c_uint64 * command_count),
    ]
    return struct_amdxdna_cmd_chain


def struct_amdxdna_cmd(count):
    class struct_amdxdna_cmd(Structure):
        pass

    struct_amdxdna_cmd._pack_ = 1  # source:False
    struct_amdxdna_cmd._fields_ = [
        ("state", ctypes.c_uint32, 4),
        ("unused", ctypes.c_uint32, 6),
        ("extra_cu_masks", ctypes.c_uint32, 2),
        ("count", ctypes.c_uint32, 11),
        ("opcode", ctypes.c_uint32, 5),
        ("reserved", ctypes.c_uint32, 4),
        ("data", ctypes.c_uint32 * count),
    ]
    return struct_amdxdna_cmd


# DRM_IOCTL_AMDXDNA_SUBMIT_WAIT = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_SUBMIT_WAIT , struct_amdxdna_drm_syncobjs ) # macro
# DRM_IOCTL_AMDXDNA_SUBMIT_SIGNAL = DRM_IOWR ( DRM_COMMAND_BASE + DRM_AMDXDNA_SUBMIT_SIGNAL , struct_amdxdna_drm_syncobjs ) # macro
__all__ = [
    "AMDXDNA_ACCEL_H_",
    "AMDXDNA_BO_CMD",
    "AMDXDNA_BO_DEV",
    "AMDXDNA_BO_DEV_HEAP",
    "AMDXDNA_BO_DMA",
    "AMDXDNA_BO_INVALID",
    "AMDXDNA_BO_SHMEM",
    "AMDXDNA_CMD_SUBMIT_DEPENDENCY",
    "AMDXDNA_CMD_SUBMIT_EXEC_BUF",
    "AMDXDNA_CMD_SUBMIT_SIGNAL",
    "AMDXDNA_DEV_TYPE_KMQ",
    "AMDXDNA_DEV_TYPE_UMQ",
    "AMDXDNA_DEV_TYPE_UNKNOWN",
    "AMDXDNA_DRIVER_MAJOR",
    "AMDXDNA_DRIVER_MINOR",
    "AMDXDNA_INVALID_ADDR",
    "AMDXDNA_INVALID_BO_HANDLE",
    "AMDXDNA_INVALID_CMD_HANDLE",
    "AMDXDNA_INVALID_CTX_HANDLE",
    "AMDXDNA_INVALID_FENCE_HANDLE",
    "AMDXDNA_SENSOR_TYPE_POWER",
    "DRM_AMDXDNA_CONFIG_HWCTX",
    "DRM_AMDXDNA_CREATE_BO",
    "DRM_AMDXDNA_CREATE_HWCTX",
    "DRM_AMDXDNA_DESTROY_HWCTX",
    "DRM_AMDXDNA_EXEC_CMD",
    "DRM_AMDXDNA_GET_BO_INFO",
    "DRM_AMDXDNA_GET_INFO",
    "DRM_AMDXDNA_GET_POWER_MODE",
    "DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF",
    "DRM_AMDXDNA_HWCTX_CONFIG_CU",
    "DRM_AMDXDNA_HWCTX_CONFIG_NUM",
    "DRM_AMDXDNA_HWCTX_REMOVE_DBG_BUF",
    "DRM_AMDXDNA_NUM_GET_PARAM",
    "DRM_AMDXDNA_NUM_IOCTLS",
    "DRM_AMDXDNA_NUM_SET_PARAM",
    "DRM_AMDXDNA_QUERY_AIE_METADATA",
    "DRM_AMDXDNA_QUERY_AIE_STATUS",
    "DRM_AMDXDNA_QUERY_AIE_VERSION",
    "DRM_AMDXDNA_QUERY_CLOCK_METADATA",
    "DRM_AMDXDNA_QUERY_FIRMWARE_VERSION",
    "DRM_AMDXDNA_QUERY_HW_CONTEXTS",
    "DRM_AMDXDNA_QUERY_SENSORS",
    "DRM_AMDXDNA_READ_AIE_MEM",
    "DRM_AMDXDNA_READ_AIE_REG",
    "DRM_AMDXDNA_SET_POWER_MODE",
    "DRM_AMDXDNA_SET_STATE",
    "DRM_AMDXDNA_SUBMIT_SIGNAL",
    "DRM_AMDXDNA_SUBMIT_WAIT",
    "DRM_AMDXDNA_SYNC_BO",
    "DRM_AMDXDNA_WAIT_CMD",
    "DRM_AMDXDNA_WRITE_AIE_MEM",
    "DRM_AMDXDNA_WRITE_AIE_REG",
    "POWER_MODE_DEFAULT",
    "POWER_MODE_HIGH",
    "POWER_MODE_LOW",
    "POWER_MODE_MEDIUM",
    "SYNC_DIRECT_FROM_DEVICE",
    "SYNC_DIRECT_TO_DEVICE",
    "amdxdna_bo_type",
    "amdxdna_cmd_type",
    "amdxdna_device_type",
    "amdxdna_drm_config_hwctx_param",
    "amdxdna_drm_get_param",
    "amdxdna_drm_ioctl_id",
    "amdxdna_drm_set_param",
    "amdxdna_power_mode_type",
    "amdxdna_sensor_type",
    "struct_amdxdna_cu_config",
    "struct_amdxdna_drm_aie_mem",
    "struct_amdxdna_drm_aie_reg",
    "struct_amdxdna_drm_config_hwctx",
    "struct_amdxdna_drm_create_bo",
    "struct_amdxdna_drm_create_hwctx",
    "struct_amdxdna_drm_destroy_hwctx",
    "struct_amdxdna_drm_exec_cmd",
    "struct_amdxdna_drm_get_bo_info",
    "struct_amdxdna_drm_get_info",
    "struct_amdxdna_drm_get_power_mode",
    "struct_amdxdna_drm_query_aie_metadata",
    "struct_amdxdna_drm_query_aie_status",
    "struct_amdxdna_drm_query_aie_tile_metadata",
    "struct_amdxdna_drm_query_aie_version",
    "struct_amdxdna_drm_query_clock",
    "struct_amdxdna_drm_query_clock_metadata",
    "struct_amdxdna_drm_query_firmware_version",
    "struct_amdxdna_drm_query_hwctx",
    "struct_amdxdna_drm_query_sensor",
    "struct_amdxdna_drm_set_power_mode",
    "struct_amdxdna_drm_set_state",
    "struct_amdxdna_drm_sync_bo",
    "struct_amdxdna_drm_syncobjs",
    "struct_amdxdna_drm_wait_cmd",
    "struct_amdxdna_hwctx_param_config_cu",
    "struct_amdxdna_qos_info",
    "struct_amdxdna_cmd_chain",
    "struct_amdxdna_cmd",
]
