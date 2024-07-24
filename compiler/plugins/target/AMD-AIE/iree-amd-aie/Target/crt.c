/// This is ported from llvm-aie's libc/startup/baremetal/aie2/crt0.S and
/// libc/startup/baremetal/aie2/crt1.cc. The purpose is to have a "baremetal"
/// crt for peano. Tip peano (currently) automatically links their own crt but
/// older peano (eg what we have in CI as of 7/23/204) does not.
/// Note, this has nothing to do with chess (which also automatically links its
/// crt).
R"crt(
__asm(
    ".global __start\n"
    ".type __start, STT_FUNC\n"
    ".global _sp_start_value_DM_stack\n"
    ".global _main_init\n"
    "__start:\n"
    "JL  #_main_init\n"
    "MOVXM sp, #_sp_start_value_DM_stack\n"
    "NOP\n"
    "NOP\n"
    "NOP\n"
    "NOP\n");

_Noreturn void done (void) {
  __builtin_aiev2_sched_barrier();
  __builtin_aiev2_done();
  __builtin_aiev2_sched_barrier();
}

extern int main(int, char**);
_Noreturn void _Exit (int val) {
  (void)val;
  done();
}

#define NULL ((void*)0)

void _main_init() {

  _Exit(main(0, NULL));
}

#ifdef __cplusplus
}
#endif  // __cplusplus
)crt"
