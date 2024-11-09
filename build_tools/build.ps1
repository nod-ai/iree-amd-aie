$this_dir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$repo_root = Resolve-Path -Path "$this_dir/.."
$iree_dir = Resolve-Path -Path "$repo_root/third_party/iree"
$build_dir = "$repo_root/iree-build"

echo "Building all"
echo "------------"
& cmake --build $build_dir -- -k 0