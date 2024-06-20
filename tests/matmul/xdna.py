def get_name():
    return "amd-aie"

def get_permutations():
    return {'driver': ['xrt'], 'pipeline': ['pad', 'pack', 'simple-pack']}
