/******************************************************************************
* Copyright 2015-2022 Xilinx, Inc.
* Copyright 2022-2023 Advanced Micro Devices, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
******************************************************************************/

#ifndef _SYS_ELF_H
#define	_SYS_ELF_H

/*
-------------------------------------------------------------------------------
***********************************************   H E A D E R   F I L E S   ***
-------------------------------------------------------------------------------
*/
#include <stdint.h>


/* ELF 32-bit */
typedef uint32_t    Elf32_Addr;
typedef uint16_t    Elf32_Half;
typedef uint32_t    Elf32_Off;
typedef int32_t     Elf32_Sword;
typedef uint32_t    Elf32_Xword;
typedef uint32_t    Elf32_Word;

/* ELF 64-bit */
typedef uint64_t    Elf64_Addr;
typedef uint64_t    Elf64_Off;
typedef int64_t     Elf64_Sxword;
typedef uint64_t    Elf64_Xword;
typedef int32_t     Elf32_Sword;
typedef uint32_t    Elf64_Word;
typedef uint16_t    Elf64_Half;

#ifdef __cplusplus
extern "C" {
#endif


/*
-------------------------------------------------------------------------------
*********************************************   P R E P R O C E S S O R S   ***
-------------------------------------------------------------------------------
*/
#define EI_NIDENT           16
#define ELF32_FSZ_ADDR      4
#define ELF32_FSZ_HALF      2
#define ELF32_FSZ_OFF       4
#define ELF32_FSZ_SWORD     4
#define ELF32_FSZ_WORD      4

/* e_ident[] indexes */
#define EI_MAG0             0
#define EI_MAG1             1
#define EI_MAG2             2
#define EI_MAG3             3
#define EI_CLASS            4
#define EI_DATA             5
#define EI_VERSION          6
#define EI_PAD              7

/* EI_MAG */
#define ELFMAG0             0x7f
#define ELFMAG1             'E'
#define ELFMAG2             'L'
#define ELFMAG3             'F'
#define ELFMAG              "\177ELF"
#define SELFMAG             4

/* EI_DATA */
#define ELFDATANONE         0
#define ELFDATA2LSB         1
#define ELFDATA2MSB         2
#define ELFDATANUM          3

/* e_type */
#define ET_NONE             0
#define ET_REL              1
#define ET_EXEC             2
#define ET_DYN              3
#define ET_CORE             4
#define ET_NUM              5

/* processor specific range */
#define ET_LOPROC           0xff00
#define ET_HIPROC           0xffff

/* e_machine */
#define EM_NONE             0
#define EM_M32              1       /* AT&T WE 32100 */
#define EM_SPARC            2       /* Sun SPARC */
#define EM_386              3       /* Intel 80386 */
#define EM_68K              4       /* Motorola 68000 */
#define EM_88K              5       /* Motorola 88000 */
#define EM_486              6       /* Intel 80486 */
#define EM_860              7       /* Intel i860 */
#define EM_MIPS             8       /* MIPS RS3000 Big-Endian */
#define EM_UNKNOWN9         9
#define EM_MIPS_RS3_LE      10      /* MIPS RS3000 Little-Endian */
#define EM_RS6000           11      /* RS6000 */
#define EM_UNKNOWN12        12
#define EM_UNKNOWN13        13
#define EM_UNKNOWN14        14
#define EM_PA_RISC          15      /* PA-RISC */
#define EM_nCUBE            16      /* nCUBE */
#define EM_VPP500           17      /* Fujitsu VPP500 */
#define EM_SPARC32PLUS      18      /* Sun SPARC 32+ */
#define EM_UNKNOWN19        19
#define EM_PPC              20      /* PowerPC */
#define EM_NUM              21

/* e_version, EI_VERSION */
#define EV_NONE             0
#define EV_CURRENT          1
#define EV_NUM              2

/* p_type */
#define PT_NULL             0
#define PT_LOAD             1
#define PT_DYNAMIC          2
#define PT_INTERP           3
#define PT_NOTE             4
#define PT_SHLIB            5
#define PT_PHDR             6
#define PT_NUM              7

/* processor specific range */
#define PT_LOPROC           0x70000000
#define PT_HIPROC           0x7fffffff

/* p_flags */
#define PF_R                0x4
#define PF_W                0x2
#define PF_X                0x1

/* processor specific values */
#define PF_MASKPROC         0xf0000000

/* sh_type */
#define SHT_NULL            0
#define SHT_PROGBITS        1
#define SHT_SYMTAB          2
#define SHT_STRTAB          3
#define SHT_RELA            4
#define SHT_HASH            5
#define SHT_DYNAMIC         6
#define SHT_NOTE            7
#define SHT_NOBITS          8
#define SHT_REL             9
#define SHT_SHLIB           10
#define SHT_DYNSYM          11
#define SHT_NUM             12

#define SHT_LOSUNW          0x6ffffffd
#define SHT_SUNW_verdef     0x6ffffffd
#define SHT_SUNW_verneed    0x6ffffffe
#define SHT_SUNW_versym     0x6fffffff
#define SHT_HISUNW          0x6fffffff

/* processor specific range */
#define SHT_LOPROC          0x70000000
#define SHT_HIPROC          0x7fffffff
#define SHT_LOUSER          0x80000000
#define SHT_HIUSER          0xffffffff

/* sh_flags */
#define SHF_WRITE           0x1
#define SHF_ALLOC           0x2
#define SHF_EXECINSTR       0x4

/* processor specific values */
#define SHF_MASKPROC        0xf0000000

/* special section numbers */
#define SHN_UNDEF           0
#define SHN_LORESERVE       0xff00
#define SHN_ABS             0xfff1
#define SHN_COMMON          0xfff2
#define SHN_HIRESERVE       0xffff

/* processor specific range */
#define SHN_LOPROC          0xff00
#define SHN_HIPROC          0xff1f

#define STN_UNDEF           0


/*
-------------------------------------------------------------------------------
***************************************************   S T R U C T U R E S   ***
-------------------------------------------------------------------------------
*/
/******************************************************************************/
typedef struct {
    unsigned char e_ident[EI_NIDENT];   /* ident bytes */
    Elf32_Half e_type;                  /* file type */
    Elf32_Half e_machine;               /* target machine */
    Elf32_Word e_version;               /* file version */
    Elf32_Addr e_entry;                 /* start address */
    Elf32_Off  e_phoff;                 /* phdr file offset */
    Elf32_Off  e_shoff;                 /* shdr file offset */
    Elf32_Word e_flags;                 /* file flags */
    Elf32_Half e_ehsize;                /* sizeof ehdr */
    Elf32_Half e_phentsize;             /* sizeof phdr */
    Elf32_Half e_phnum;                 /* number phdrs */
    Elf32_Half e_shentsize;             /* sizeof shdr */
    Elf32_Half e_shnum;                 /* number shdrs */
    Elf32_Half e_shstrndx;              /* shdr string index */
} Elf32_Ehdr;

/******************************************************************************/
typedef struct {
    unsigned char e_ident[EI_NIDENT];   /* ident bytes */
    Elf64_Half e_type;                  /* file type */
    Elf64_Half e_machine;               /* target machine */
    Elf64_Word e_version;               /* file version */
    Elf64_Addr e_entry;                 /* start address */
    Elf64_Off  e_phoff;                 /* phdr file offset */
    Elf64_Off  e_shoff;                 /* shdr file offset */
    Elf64_Word e_flags;                 /* file flags */
    Elf64_Half e_ehsize;                /* sizeof ehdr */
    Elf64_Half e_phentsize;             /* sizeof phdr */
    Elf64_Half e_phnum;                 /* number phdrs */
    Elf64_Half e_shentsize;             /* sizeof shdr */
    Elf64_Half e_shnum;                 /* number shdrs */
    Elf64_Half e_shstrndx;              /* shdr string index */
} Elf64_Ehdr;

/******************************************************************************/
typedef struct 
{
    Elf32_Word p_type;                  /* entry type */
    Elf32_Off p_offset;                 /* file offset */
    Elf32_Addr p_vaddr;                 /* virtual address */
    Elf32_Addr p_paddr;                 /* physical address */
    Elf32_Word p_filesz;                /* file size */
    Elf32_Word p_memsz;                 /* memory size */
    Elf32_Word p_flags;                 /* entry flags */
    Elf32_Word p_align;                 /* memory/file alignment */
} Elf32_Phdr;

/******************************************************************************/
typedef struct 
{
    Elf32_Word sh_name;                 /* section name */
    Elf32_Word sh_type;                 /* SHT_... */
    Elf32_Word sh_flags;                /* SHF_... */
    Elf32_Addr sh_addr;                 /* virtual address */
    Elf32_Off sh_offset;                /* file offset */
    Elf32_Word sh_size;                 /* section size */
    Elf32_Word sh_link;                 /* misc info */
    Elf32_Word sh_info;                 /* misc info */
    Elf32_Word sh_addralign;            /* memory alignment */
    Elf32_Word sh_entsize;              /* entry size if table */
} Elf32_Shdr;

/******************************************************************************/
typedef struct 
{
    Elf32_Word st_name;
    Elf32_Addr st_value;
    Elf32_Word st_size;
    unsigned char st_info;              /* bind, type: ELF_32_ST_... */
    unsigned char st_other;
    Elf32_Half st_shndx;                /* SHN_... */
} Elf32_Sym;

#ifdef __cplusplus
}
#endif

#endif
