def get_name():
    return "cuda"

def get_permutations():
    return {'driver': ['cuda'], 'arch': ['sm_35', 'sm_60', 'sm_70', 'sm_80']}
