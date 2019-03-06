/*******************************************************************************
 * Copyright (C) 2019 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/


#include <cuda.h>

void cuda_check_internal(char const* file, int line, cudaError v, char const* reason = "");

#define cuda_check(...) cuda_check_internal(__FILE__, __LINE__, __VA_ARGS__)
