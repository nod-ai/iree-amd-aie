# Find appearances of keys, and replace with equivalent value. Consider all files in the current directory.

import os
import re
import copy

keys = ["DoLightMagic", "DOLIGHTMAGIC", "do_light_magic", "do-light-magic"]

# User: change these to what your target pass is called
values = ["DoDarkMagic", "DODARKMAGIC", "do_dark_magic", "do-dark-magic"]


base_key = keys[0]
base_value = values[0]

bold_key = keys[1]
bold_value = values[1]

underscore_key = keys[2]
underscore_value = values[2]

dashing_key = keys[3]
dashing_value = values[3]

# Internal check that values are internally consistent:
# 1 underscore_value and dashing_value should be the same, up to '-' and '_'
assert (
    underscore_value.replace("_", "-") == dashing_value
    ), f"{underscore_value} != {dashing_value}"

# 2 base_value and bold_value should be the same, up to case
assert (
    base_value.lower() == bold_value.lower()
    ), f"{base_value} != {bold_value}"


raise RuntimeError("Remove this line if you're sure you're golden")

def replace_keys(file, keys, values):
    with open(file, "r") as f:
        old_text = f.read()
        text = copy.copy(old_text)
    for key, value in zip(keys, values):
        text = re.sub(key, value, text)
    if old_text != text:
        with open(file, "w") as f:
            f.write(text)
            print(f"Replaced keys in {file}")


# Walk files in this directory recursively, include files of type [.txt, .c, .cpp, .h, .hpp, .mlir]
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith((".txt", ".c", ".cpp", ".h", ".hpp", ".mlir", ".td")):
            replace_keys(os.path.join(root, file), keys, values)

        # If the file ends with 'base_key'.cpp, rename it to 'base_value'.cpp
        if file.endswith(f"{base_key}.cpp"):
            new_file = file.replace(f"{base_key}.cpp", f"{base_value}.cpp")
            os.rename(os.path.join(root, file), os.path.join(root, new_file))
            print(f"Renamed {file} to {new_file}")

        if file.endswith(f"{underscore_key}.mlir"):
            new_file = file.replace(
                f"{underscore_key}.mlir", f"{underscore_value}.mlir"
            )
            os.rename(os.path.join(root, file), os.path.join(root, new_file))
            print(f"Renamed {file} to {new_file}")
