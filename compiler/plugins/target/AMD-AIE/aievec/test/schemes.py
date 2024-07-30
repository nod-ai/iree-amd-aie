import numpy as np


def n_mults(data_size, coeff_size, data_complex=False, coeff_complex=False):
    m = 1
    if data_size == 32:
        m *= 2
    if coeff_size == 32:
        m *= 2
    if data_complex:
        m *= 2
    if coeff_complex:
        m *= 2

    return m


def permute_square(idx, rows, cols, square):
    idx = np.asarray(idx)
    idx = idx.reshape(rows, cols)
    for r in range(rows // 2):
        r *= 2
        for c in range(cols // 2):
            c *= 2
            idx_ = idx[r : r + 2, c : c + 2].flatten()
            idx_ = idx_[square]
            idx[r : r + 2, c : c + 2] = idx_.reshape(2, 2)

    return idx.flatten().tolist()


# v8acc48   mul8 (v64int16 xbuff, int xstart, unsigned int xoffsets, int xstep, unsigned int xsquare, v16int16 zbuff, int zstart, unsigned int zoffsets, int zstep)
def general_scheme(
    lanes, cols, start, step, offs, offs_hi, samples_in_buffer, square=None
):
    rows = lanes
    idx = [None] * rows * cols
    for i in range(rows * cols):
        c = i % cols
        r = i // cols

        offs_ = lambda x: offs[x]
        if r >= 8:
            offs_ = lambda x: offs_hi[x - 8]

        idx[i] = (start + offs_(r) + step * c) % samples_in_buffer

    if square is not None:
        idx = permute_square(idx, rows, cols, square)

    return idx


def sixteenbx16b_data_scheme(
    lanes, cols, start, step, offs, offs_hi, samples_in_buffer, square=None
):
    rows = lanes
    idx = [None] * rows * cols
    for i in range(rows * cols):
        c = i % cols
        r = i // cols

        offs_ = lambda x: offs[x]
        if r >= 8:
            offs_ = lambda x: offs_hi[x - 8]

        if r % 2 == 0:
            offset = offs_(r) * 2
        else:
            offset = (offs_(r) * 2) + (offs_(r - 1) + 1) * 2

        x_step = (c // 2) * step + (c % 2)

        idx[i] = (start + offset + x_step) % samples_in_buffer

    if square is not None:
        idx = permute_square(idx, rows, cols, square)

    return idx


def sixteenbx8b_coefficient_scheme(
    lanes, cols, start, step, offs, offs_hi, samples_in_buffer, square=None
):
    rows = lanes
    idx = [None] * rows * cols
    for i in range(rows * cols):
        c = i % cols
        r = i // cols

        offs_ = lambda x: offs[x]
        if r >= 8:
            offs_ = lambda x: offs_hi[x - 8]

        offset = offs_(r) * 2

        step = c // 2 * step + c % 2

        idx[i] = (start + offset + step) % (samples_in_buffer)

    if square is not None:
        idx = permute_square(idx, rows, cols, square)

    return idx


def eightbx16b_data_scheme(
    lanes, cols, start, step, offs, samples_in_buffer, square=None
):
    rows = lanes
    idx = [None] * rows * cols
    for i in range(rows * cols):
        c = i % cols
        r = i // cols

        rx = r // 2
        rr = r % 4

        if rr == 0:
            offset = offs[rx] * 4
        elif rr == 1:
            offset = offs[rx] * 4 + 1
        elif rr == 2:
            offset = offs[rx] * 4 + (offs[rx - 1] + 1) * 4
        elif rr == 3:
            offset = offs[rx] * 4 + (offs[rx - 1] + 1) * 4 + 1

        step = c // 2 * step + (c % 2) * 2

        idx[i] = (start + offset + step) % (samples_in_buffer)

    if square is not None:
        idx = permute_square(idx, rows, cols, square)

    return idx


def eightbx8b_coefficient_scheme(
    lanes, cols, start, step, offs, offs_hi, samples_in_buffer, square=None
):
    rows = lanes
    idx = [None] * rows * cols

    for i in range(rows * cols):
        c = i % cols
        r = i // cols
        rz = (r / 4) * 2 + (r % 2)

        offs_ = lambda x: offs[x]
        if r >= 8:
            offs_ = lambda x: offs_hi[x - 8]

        offset = offs_(rz) * 2

        step = c / 2 * step + (c % 2)

        idx[i] = (start + offset + step) % (samples_in_buffer)

    if square is not None:
        idx = permute_square(idx, rows, cols, square)

    return idx


# v8acc48   mul8 (v64int16 xbuff, int xstart, unsigned int xoffsets, int xstep, unsigned int xsquare, v16int16 zbuff, int zstart, unsigned int zoffsets, int zstep)
# acc = mul8 (xbuff, 0, 0x03020100, 0x2110, coef, 0, 0x00000000, 1)

#     xstart = 0
#     xstep = 2
#     xoffset = 0x03020100
idxs = sixteenbx16b_data_scheme(
    8,
    4,
    0,
    2,
    [0, 3, 0, 2, 0, 1, 0, 0][::-1],
    None,
    samples_in_buffer=64,
    square=[2, 1, 1, 0][::-1],
)
for r in range(8):
    print(idxs[4 * r : 4 * (r + 1)])

# [0, 1, 2, 3]
# [1, 2, 3, 4]
# [2, 3, 4, 5]
# [3, 4, 5, 6]
# [4, 5, 6, 7]
# [5, 6, 7, 8]
# [6, 7, 8, 9]
# [7, 8, 9, 10]

print()
#     zstart = 0
#     zstep = 1
#     zoffset = 0x0
idxs = general_scheme(
    8, 4, 0, 1, [0, 0, 0, 0, 0, 0, 0, 0][::-1], None, samples_in_buffer=16
)
for r in range(8):
    print(idxs[4 * r : 4 * (r + 1)])

# [0, 1, 2, 3]
# [0, 1, 2, 3]
# [0, 1, 2, 3]
# [0, 1, 2, 3]
# [0, 1, 2, 3]
# [0, 1, 2, 3]
# [0, 1, 2, 3]
# [0, 1, 2, 3]

print()

# v16acc48 mul16 (v32int16 xbuff, int xstart, unsigned int xoffsets, int xoffsets_hi, int xysquare, v16int16 zbuff, int zstart, int zoffsets, int zoffsets_hi, int zstep)
# acc = mul16 (xbuff, 0, 0x03020100, 0x47362514 , 0x2110, coef, 0, 0x00000000, 0x00000000, 1)
idxs = sixteenbx16b_data_scheme(
    16,
    2,
    0,
    1,
    offs=[0, 3, 0, 2, 0, 1, 0, 0][::-1],
    offs_hi=[4, 7, 3, 6, 2, 5, 1, 4][::-1],
    samples_in_buffer=32,
    square=[2, 1, 1, 0][::-1],
)
for r in range(16):
    print(idxs[2 * r : 2 * (r + 1)])

# [0, 1]
# [1, 2]
# [2, 3]
# [3, 4]
# [4, 5]
# [5, 6]
# [6, 7]
# [7, 8]
# [8, 9]
# [9, 12]
# [10, 11]
# [11, 16]
# [12, 13]
# [13, 20]
# [14, 15]
# [15, 24]
