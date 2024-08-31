iss::create %PROCESSORNAME% iss
iss program load ./test.exe -disassemble -nmlpath /home/mlevental/dev_projects/iree-amd-aie/Vitis/2024.2/aietools/data/aie_ml/lib -extradisassembleopts +Mdec -do_not_set_entry_pc 1 -do_not_load_sp 1 -pm_check first -load_offsets {} -software_breakpoints_allowed on -hardware_breakpoints_allowed on
iss fileinput add SCD 0 -field -file ./i16_max_reduce.mlir -interval_files {} -position 0 -type {} -radix decimal -filter {} -break_on_wrap 0 -cycle_based 0 -format integer -gen_vcd_event 0 -structured 0 -bin_nbr_bytes 1 -bin_lsb_first 0
iss step -1
iss profile save test.prf
iss close
exit

