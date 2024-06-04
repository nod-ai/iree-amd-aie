/******************************************************************************
* Copyright 2018-2022 Xilinx, Inc.
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

/*****************************************************************************/
/**
 * @file cdo_driver.cc
 * @{
 * This file contains set of APIs used to generate AIE-CDO file.
 *
 * */
/***************************** Include Files *********************************/

#include "cdo_driver.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>

/************************** Macro Definitions *****************************/

/************************** Variable Definitions *****************************/
static bool axi_mm_debug = false;
static FILE *fp = NULL;
cdoHeader hdr = {.NumWords = 0x4, .IdentWord = 0x004F4443, .Version = 0x00000200, .CDOLength = 0x0, .CheckSum = 0x0 };
static bool disableNoOpInsertionForDMACmd = false;
/************************** Function Definitions *****************************/

size_t (*fwrite_wrapper)(const void *ptr, size_t size, size_t count, FILE *stream);

void reverseEndianness(unsigned char* buffer, size_t count) {

	/*Note: As per the requirement CDO is generated in little/big endain with boundary of
	 * 4 bytes (1 word). If Address or any other value to be written is 64 bit, then it is split into
	 * 2 words before endianness is swapped. So uint64_t Addr is always sent as Addr1,Addr0 of type unit32_t.
	 * Because all registers are 32 bit, and LE CDOs could be used without bootgen also. */

	for (int i = 0; i < count; i++) {
		unsigned char a = buffer[4 * i];
		unsigned char b = buffer[4 * i + 1];
		buffer[4 * i] = buffer[4 * i + 3];
		buffer[4 * i + 1] = buffer[4 * i + 2];
		buffer[4 * i + 2] = b;
		buffer[4 * i + 3] = a;
	}

}


size_t LEfwrite(const void *ptr, size_t size, size_t count, FILE *stream) {

	size_t num =0;
	if (size != 4)
		printf("Warning: It is expected to have each write of 4 bytes in CDO file. \n");

	void * copy = malloc(size*count);
	if (copy != NULL)
		memcpy(copy, ptr, (size*count));

	unsigned char * buffer = (unsigned char*) copy;
	int x = 1;
	if (*((char*) &x) == 1) {
		/* Little endian machine, fwrite directly */
		num = fwrite(buffer, size, count, stream);
	} else {
		/* Big endian machine, process and  fwrite in little endian */
		unsigned char *buffer = (unsigned char*) ptr;
		reverseEndianness(buffer, count);
		num = fwrite(buffer, size, count, stream);
	}

	free(copy);
	copy=NULL;
	buffer=NULL;
	return (num);
}

size_t BEfwrite(const void *ptr, size_t size, size_t count, FILE *stream) {

	size_t num =0;
	if (size != 4)
		printf("Warning: It is expected to have each write of 4 bytes in CDO file. \n");

	void * copy = malloc(size*count);
	if (copy != NULL)
		memcpy(copy, ptr, (size*count));

	unsigned char * buffer = (unsigned char*) copy;
	int x = 1;
	if (*((char*) &x) == 1) {
		/* Little endian machine, process and  fwrite in big endian */
		reverseEndianness(buffer, count);
		num = fwrite(buffer, size, count, stream);
	} else {
		/* Big endian machine, fwrite directly*/
		num = fwrite(buffer, size, count, stream);

	}
	free(copy);
	copy=NULL;
	buffer=NULL;
	return (num);
}

void EnAXIdebug() {
	axi_mm_debug = true;

}

void disableDmaCmdAlignment() {
  disableNoOpInsertionForDMACmd = true;
}


void setEndianness(bool endianness) {

	if(endianness == Little_Endian)  /*Little Endian */
		fwrite_wrapper = &LEfwrite;
	else /*Big Endian */
		fwrite_wrapper = &BEfwrite;

}


void startCDOFileStream(const char* cdoFileName)
{
    fp = fopen(cdoFileName, "wb+");
    if (fp == NULL) {
        printf("File could not be opened, fopen Error: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    printf("Generating: %s\n", cdoFileName);
}

void endCurrentCDOFileStream()
{
    fclose(fp);
    if(fp!=NULL) {
        fp = NULL;
    }
}

void FileHeader() {

	fwrite_wrapper(&hdr.NumWords, sizeof hdr.NumWords, 1, fp);
	fwrite_wrapper(&hdr.IdentWord, sizeof hdr.IdentWord, 1, fp);
	fwrite_wrapper(&hdr.Version, sizeof hdr.Version, 1, fp);
	fwrite_wrapper(&hdr.CDOLength, sizeof hdr.CDOLength, 1, fp);
	fwrite_wrapper(&hdr.CheckSum, sizeof hdr.CheckSum, 1, fp);


}


void cdo_Write32(uint64_t Addr, uint32_t Data) {

	/* Format: Reserved[31:24] | Length[23:16] | Handler[15:8] | API-ID[7:0] */
	const uint32_t Write64CmdHdr = ((0x03U << 16) | CDO_CMD_WRITE64);  /* 3 Words - PLM - CMD_WRITE */
	if (axi_mm_debug)
		printf("(Write64): Address:  0x%016lX Data:  0x%08X  \n", Addr, Data);
	uint32_t Addr0 = (uint32_t)(Addr);
	uint32_t Addr1 = (uint32_t)(Addr>>32);
	fwrite_wrapper(&Write64CmdHdr, sizeof Write64CmdHdr, 1, fp);
	fwrite_wrapper(&Addr1, sizeof Addr1, 1, fp);
	fwrite_wrapper(&Addr0, sizeof Addr0, 1, fp);
	fwrite_wrapper(&Data, sizeof Data, 1, fp);

}

void cdo_MaskWrite32(uint64_t Addr, uint32_t Mask, uint32_t Data) {

	/* Format: Reserved[31:24] | Length[23:16] | Handler[15:8] | API-ID[7:0] */
	const uint32_t MaskWrite64CmdHdr = ((0x04U << 16) | CDO_CMD_MASK_WRITE64); /* 4 Words - PLM - CMD_MASK_WRITE */
	if (axi_mm_debug)
		printf("(MaskWrite64): Address: 0x%016lX  Mask: 0x%08X  Data: 0x%08X \n", Addr, Mask, Data);
	uint32_t Addr0 = (uint32_t)(Addr);
	uint32_t Addr1 = (uint32_t)(Addr>>32);
	fwrite_wrapper(&MaskWrite64CmdHdr, sizeof MaskWrite64CmdHdr, 1, fp);
	fwrite_wrapper(&Addr1, sizeof Addr1, 1, fp);
	fwrite_wrapper(&Addr0, sizeof Addr0, 1, fp);
	fwrite_wrapper(&Mask, sizeof Mask, 1, fp);
	fwrite_wrapper(&Data, sizeof Data, 1, fp);

}

// Determine number of bytes required for padding, and insert NOP CMD to make
// source side address of DMA WRITE CMD pay-load to be 16 byte aligned.
// BOOTGEN post-processes CDO partitions to make it align, but as per discussion
// with "Prashant Malladi" user can disable post-processing, and they suggest
// that individual CDO should take care of alignment for efficient DMA operation.
unsigned int getPadBytesForDmaWrCmdAlignment(uint32_t DmaCmdLength)
{
    size_t DmaCmdSizeExceptPayload = (3 * sizeof(uint32_t)); // Header + Addr0 + Addr1 in bytes
    if(DmaCmdLength > 255) {
        DmaCmdSizeExceptPayload += sizeof(uint32_t); // 1 word for length in long command
    }

    // Compute number of padding bytes required for DMA WRITE CMD payload alignment
    long int currByteOffset = ftell(fp);
    if(currByteOffset == -1L)
    {
        perror("ftell()");
        fprintf(stderr,"INTERNAL ERROR: Failed to align DMA writes\n");
        exit(EXIT_FAILURE);
     }

    unsigned int numPadBytes = 0;
    if(((currByteOffset + DmaCmdSizeExceptPayload ) % 16) != 0) // Only if start of payload is not already aligned
        numPadBytes = (16 - ((currByteOffset + DmaCmdSizeExceptPayload) % 16));

    return (numPadBytes);

}


void insertNoOpCommand(unsigned int numPadBytes)
{
    // Minimum padding has to be 1 word, at-least header w/o command payload
    if (numPadBytes < 4)
        return;

    uint32_t NoOpCmdLength = (numPadBytes - sizeof(uint32_t))/sizeof(uint32_t);
    uint32_t NoOpCmdHdr = ((NoOpCmdLength << 16) | CDO_CMD_NO_OPERATION);
    uint32_t Data = 0;

    if (axi_mm_debug)
        printf("(NOP Command): Payload Length: %d \n", NoOpCmdLength);

    fwrite_wrapper(&NoOpCmdHdr, sizeof(NoOpCmdHdr), 1,fp);
    for(uint32_t i=0; i < NoOpCmdLength; i++) {
        fwrite_wrapper(&Data, sizeof(Data), 1, fp);
    }
}

void insertDmaWriteCmdHdr(uint32_t DmaCmdLength)
{
    uint32_t DmaWriteCmdHdr;
    if(DmaCmdLength < 255) {
        DmaWriteCmdHdr = ((DmaCmdLength << 16) | CDO_CMD_DMA_WRITE);
        fwrite_wrapper(&DmaWriteCmdHdr, sizeof DmaWriteCmdHdr, 1, fp);
    }
    else {
        DmaWriteCmdHdr = ((255 << 16) | CDO_CMD_DMA_WRITE);
        fwrite_wrapper(&DmaWriteCmdHdr, sizeof DmaWriteCmdHdr, 1, fp);
        fwrite_wrapper(&DmaCmdLength, sizeof(DmaCmdLength), 1, fp);
    }
}


void cdo_BlockWrite32(uint64_t Addr, uint32_t* pData, uint32_t size)
{

    uint32_t DmaCmdLength = size + 2; // Length in words including address

    // Insert No Operation command
    if (!disableNoOpInsertionForDMACmd) {
        unsigned int numPadBytes = getPadBytesForDmaWrCmdAlignment(DmaCmdLength);

        if ((numPadBytes != 0))
            insertNoOpCommand(numPadBytes);
    }

    if (axi_mm_debug)
    {
        printf("(BlockWrite-DMAWriteCmd): Start Address: 0x%016lX  Size: %" PRIu32 "\n", Addr, size);
        for(uint32_t i = 0; i < size; i ++) {
            printf("    Address: 0x%016lX  Data@ 0x%" PRIxPTR " is: 0x%08X \n", (Addr + 4*i), (uintptr_t)(pData + i), *(pData + i));
        }
        printf("\n");
    }

    insertDmaWriteCmdHdr(DmaCmdLength);
    uint32_t Addr0 = (uint32_t)(Addr);
    uint32_t Addr1 = (uint32_t)(Addr >> 32);
    fwrite_wrapper(&Addr1, sizeof Addr1, 1, fp);
    fwrite_wrapper(&Addr0, sizeof Addr0, 1, fp);

    // Write the block of data pointed by pData
    fwrite_wrapper(pData, sizeof(uint32_t), size, fp);

}


void cdo_BlockSet32(uint64_t Addr, uint32_t Data, uint32_t size)
{
    uint32_t DmaCmdLength = size + 2; // Length in words including address

    // Insert No Operation command
    if (!disableNoOpInsertionForDMACmd) {
        unsigned int numPadBytes = getPadBytesForDmaWrCmdAlignment(DmaCmdLength);

        if ((numPadBytes != 0))
            insertNoOpCommand(numPadBytes);
    }

    if (axi_mm_debug)
    {
        printf("(BlockSet-DMAWriteCmd):Start Address: 0x%016lX  Size: %" PRIu32 "\n", Addr, size);
        for(uint32_t i = 0; i < size; i ++) {
            printf("    Address: 0x%016lX  Data is: 0x%08X \n", (Addr + 4*i), Data);
        }
        printf("\n");
    }

    insertDmaWriteCmdHdr(DmaCmdLength);
    uint32_t Addr0 = (uint32_t)(Addr);
    uint32_t Addr1 = (uint32_t)(Addr >> 32);
    fwrite_wrapper(&Addr1, sizeof Addr1, 1, fp);
    fwrite_wrapper(&Addr0, sizeof Addr0, 1, fp);

    // Write the specified value for initializing entire address block
    for(uint32_t i = 0; i < size; i++)
        fwrite_wrapper(&Data, sizeof Data, 1, fp);

}

void cdo_MaskPoll(uint64_t Addr, uint32_t Mask, uint32_t Expected_Value, uint32_t TimeoutInMS) {

	/* Format: Reserved[31:24] | Length[23:16] | Handler[15:8] | API-ID[7:0] */
	const uint32_t MaskPoll64CmdHdr = ((0x05U << 16) | CDO_CMD_MASK_POLL64); /* 5 Words - PLM - CMD_MASK_POLL64 */
	if (axi_mm_debug)
		printf("(MaskPoll64): Address: 0x%016lX  Mask: 0x%08X  Expected Value: 0x%08X  Timeout(ms): 0x%08X \n", Addr, Mask, Expected_Value, TimeoutInMS);
	uint32_t Addr0 = (uint32_t)(Addr);
	uint32_t Addr1 = (uint32_t)(Addr>>32);
	fwrite_wrapper(&MaskPoll64CmdHdr, sizeof MaskPoll64CmdHdr, 1, fp);
	fwrite_wrapper(&Addr1, sizeof Addr1, 1, fp);
	fwrite_wrapper(&Addr0, sizeof Addr0, 1, fp);
	fwrite_wrapper(&Mask, sizeof Mask, 1, fp);
	fwrite_wrapper(&Expected_Value, sizeof Expected_Value, 1, fp);
	fwrite_wrapper(&TimeoutInMS, sizeof TimeoutInMS, 1, fp);

}


void configureHeader() {

	fseek(fp, 0, SEEK_SET);
	unsigned int numWords = 0;
	for (;;) {
		uint32_t buf;
		size_t n = fread(&buf, 4, 1, fp);
		numWords = numWords + n;
		if (n < 1) {
			break;
		}
	}

	hdr.CDOLength = (numWords - 5); // Length of CDO object in words excluding header length
	if( fseek(fp, 3 * 4, SEEK_SET) == 0) // Move ahead 3 words and update length of header in words and checksum
	{
	  if (ferror(fp))
	  {
	    perror("fseek()");
	    fprintf(stderr,"INTERNAL_ERROR: Failed to configure AIE-CDO Header \n");
	    exit(EXIT_FAILURE);
	  }
	}

	fwrite_wrapper(&hdr.CDOLength, sizeof hdr.CDOLength, 1, fp);
	hdr.CheckSum = ~(hdr.NumWords + hdr.IdentWord + hdr.Version + hdr.CDOLength);
	fwrite_wrapper(&hdr.CheckSum, sizeof hdr.CheckSum, 1, fp);
}

