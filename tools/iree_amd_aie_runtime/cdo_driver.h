/******************************************************************************
* Copyright 2019-2022 Xilinx, Inc.
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

#ifndef _CDO_DRIVER_H_
#define _CDO_DRIVER_H_

#include <stdbool.h>
#include <stdint.h>

enum byte_ordering{Little_Endian, Big_Endian};

enum CDO_COMMAND {
    CDO_CMD_DMA_WRITE     = 0x105U,
    CDO_CMD_MASK_POLL64   = 0x106U,
    CDO_CMD_MASK_WRITE64  = 0x107U,
    CDO_CMD_WRITE64       = 0x108U,
    CDO_CMD_NO_OPERATION  = 0x111U
};

typedef struct cdoHeader {

	/* Format:
	 *  0x0 - No. of remaining words in header,
	 *  0x4 - Identification Word,
	 *  0x8 - Version,
	 *  0xC - CDO Length,
	 *  0x10- Checksum 	 */

	const uint32_t NumWords;
	const uint32_t IdentWord;
	const uint32_t Version;

	uint32_t CDOLength; // Length of CDO object in words excluding header.
	uint32_t CheckSum; // one's complement of sum of all fields in header i.e. ~(NumWords+IdentWord+Version+Length)

} cdoHeader;

void startCDOFileStream(const char* cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void SectionHeader();
void configureHeader();
unsigned int getPadBytesForDmaWrCmdAlignment(uint32_t DmaCmdLength);
void insertNoOpCommand(unsigned int numPadBytes);
void insertDmaWriteCmdHdr(uint32_t DmaCmdLength);
void disableDmaCmdAlignment();


#endif /* _CDO_DRIVER_H_ */
