// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xdna-oplib %s | FileCheck %s

func.func @test() {
  return
}
// CHECK: Hello from XDNAOpLib
