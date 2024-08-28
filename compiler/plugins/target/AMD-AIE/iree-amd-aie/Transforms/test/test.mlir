module @test_single {
  func.func private @callee(%i: index, %j: index)
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %tile_0_2 = amdaie.tile(%c0, %c2)
  %core_0_2 = amdaie.core(%tile_0_2, in : [], out : []) {
    scf.forall (%i, %j) in (2, 2) {
      func.call @callee(%i, %j) : (index, index) -> ()
    }
    amdaie.end
  }
}
