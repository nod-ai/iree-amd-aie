#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import cpu
import cuda

permute_config = {'backend': [],
                'input_op': ['stablehlo',
                             {'linalg_matmul': {'accumulate': [True, False],
                                                'transpose': [None, 'a', 'b']}}],
                'iterations': [1, 10],
                'dtype': ['i8', 'i32', 'f32', 'f16', 'bf16']
                }

all_backends = [cpu, cuda]

def get_permutations(pconfig):
    '''Compute the permutations from the permutation configuration structure `pconfig` - essentially a
cartesian product of all the available choices, but accounts for the nesting of options.'''
    if isinstance(pconfig, list):
        for x in pconfig:
            if isinstance(x, dict):
                for y in get_permutations(x):
                    yield y
            else:
                yield x
        return
    elif isinstance(pconfig, dict):
        pconfig = iter(pconfig.items())

    try:
        key, value = next(pconfig)
    except StopIteration:
        yield {}
        return
    for x in get_permutations(pconfig):
        for y in get_permutations(value):
            yield {key: y} | x

def get_fields(perms, prefix=''):
    if isinstance(perms, list) and len(perms):
        options = []
        for x in perms:
            if isinstance(x, dict):
                name = list(x.keys())[0]
                options.append(name)
                for y in get_fields(x, prefix):
                    yield y
            else:
                options.append(x)
        yield prefix, options
    elif isinstance(perms, dict):
        for k, v in perms.items():
            new_prefix = '.'.join((prefix, str(k))) if prefix else str(k)
            for x in get_fields(v, new_prefix):
                yield x

def parse_perm_opt(opt):
    if opt.lower() == 'none':
        return None
    elif opt.lower() == 'true':
        return True
    elif opt.lower() == 'false':
        return False
    try:
        return int(opt)
    except ValueError:
        pass

    bounds = opt.split('-')
    if len(bounds) == 2:
        try:
            start = int(bounds[0])
            stop = int(bounds[1]) + 1
            return range(start, stop)
        except ValueError:
            pass
    return opt

def apply_perm_opt(s, perms):
    field, user_values = str(s).split('=')
    field = field.split('.')
    user_values = user_values.split(',')
    for f in field[:-1]:
        if isinstance(perms, list):
            for x in perms:
                if isinstance(x, dict) and f in x:
                    perms = x[f]
        else:
            perms = perms[f]

    field = field[-1]
    old_values = perms[field]
    new_values = []
    for x in user_values:
        x = parse_perm_opt(x)
        if x == '':
            continue
        if isinstance(x, range):
            new_values += list(x)
            continue
        added = False
        for ov in old_values:
            if isinstance(ov, dict) and x in ov:
                new_values.append(ov)
                added = True
        if not added:
            new_values.append(x)
    perms[field] = new_values

def parse_exclusion(pat):
    clauses = pat.split('&')
    clauses = [clause.split('=') for clause in clauses]
    clauses = [(field.split('.'), value) for field, value in clauses]
    return clauses

def get_field(perm, field):
    for f in field:
        perm = perm[f]
    if isinstance(perm, dict) and len(perm) == 1:
        return list(perm.keys())[0]
    return perm

def match_exclusion(perm, exclusion):
    return all([get_field(perm, field) == str(value).lower()
                for field, value in exclusion])

def apply_exclusions(perms, exclusions):
    for p in perms:
        if any([match_exclusion(p, q) for q in exclusions]):
            continue
        yield p

def add_arguments(parser):
    parser.add_argument('--load', nargs='+', help="Load an external backend (python module)")
    parser.add_argument('-p', '--set', nargs='+', help="Set permutation options: field=v1,v2")
    parser.add_argument('-q', '--exclude', nargs='+', help="Cut permutations matching the pattern: backend=cpu\\&dtype=bf16")
    parser.add_argument('--show-fields', action='store_true', help="Show all of the permutation fields and their available options and exit")
    parser.add_argument('--show-perms', action='store_true', help="List every permutation and exit")
    parser.add_argument('-n', '--dry-run', action='store_true', help="Don't compile or run anything")
    parser.add_argument('--compile-only', action='store_true', help="Only compile, don't run the kernels")
    pass

def main(argv):
    parser = argparse.ArgumentParser(description='Test and benchmark different matmul configurations')
    add_arguments(parser)
    args = parser.parse_args(argv[1:])

    if args.load:
        for extra_backend in args.load:
            all_backends.append(__import__(extra_backend))

    for backend in all_backends:
        permute_config['backend'].append({backend.__name__: backend.get_permutations()})

    if args.set:
        for p in args.set:
            apply_perm_opt(p, permute_config)

    exclusions = []
    for q in args.exclude or []:
        exclusions.append(parse_exclusion(q))
    print(exclusions)

    if args.show_fields:
        for name, value in get_fields(permute_config):
            print(name, '=', value)
    permutations = list(apply_exclusions(get_permutations(permute_config), exclusions))
    if args.show_perms:
        for i, p in enumerate(permutations):
            print("#" + str(i) + ":", p)
    if args.show_fields or args.show_perms:
        return 0

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
