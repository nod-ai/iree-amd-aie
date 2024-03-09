
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x0020000
   data (!RX) : ORIGIN = 0x75400, LENGTH = 0xAC00
}
ENTRY(_main_init)
SECTIONS
{
  . = 0x0;
  .text : {
     /* the _main_init symbol from me_basic.o has to come at address zero. */
     *me_basic.o(.text)
     . = 0x200;
     _ctors_start = .;
     _init_array_start = .;
     KEEP(SORT(*.init_array))
     _ctors_end = .;
     _init_array_end = .;
     _dtors_start = .;
     _dtors_end = .;
     *(.text)
  } > program
  .data : {
     *(.data*);
     *(.rodata*)
  } > data
. = 0x70000;
_sp_start_value_DM_stack = .;
. += 0x400; /* stack */
. = 0x40400;
buf9 = .;
. += 0x1000;
. = 0x41400;
buf8 = .;
. += 0x1000;
. = 0x42400;
buf7 = .;
. += 0x1000;
. = 0x43400;
buf6 = .;
. += 0x1000;
. = 0x44400;
buf5 = .;
. += 0x1000;
/* No tile with memory exists to the west. */
. = 0x50000;
. += 0x10000;
. = 0x60400;
buf19 = .;
. += 0x1000;
. = 0x61400;
buf18 = .;
. += 0x1000;
. = 0x62400;
buf17 = .;
. += 0x1000;
. = 0x63400;
buf16 = .;
. += 0x1000;
. = 0x64400;
buf15 = .;
. += 0x1000;
. = 0x70400;
buf14 = .;
. += 0x1000;
. = 0x71400;
buf13 = .;
. += 0x1000;
. = 0x72400;
buf12 = .;
. += 0x1000;
. = 0x73400;
buf11 = .;
. += 0x1000;
. = 0x74400;
buf10 = .;
. += 0x1000;
  .bss : { *(.bss) } > data
  .bss.DMb.4 : { *(.bss.DMb.4) } > data
}
PROVIDE(_main = core_0_4);
