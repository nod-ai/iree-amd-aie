#ifndef IREE_EXPERIMENTAL_HIP_HELPER_H_
#define IREE_EXPERIMENTAL_HIP_HELPER_H_

// #define TRACE_HSA

#ifdef TRACE_HSA
#include <stdio.h>
#define HSA_LOG(message)                                                                 \
    do {                                                                                 \
        printf("%s\n\t%s:%i\n\t%s\n", message, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
    } while(0)                                                                            
#else
#define HSA_LOG(message) ((void)0)
#endif

#undef TRACE_HSA

#endif // IREE_EXPERIMENTAL_HIP_HELPER_H_


