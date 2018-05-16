/*******************************************************************************
 * Copyright (C) 2018 David Piper <david.piper@tu-dortmund.de
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

# pragma once

//helper functions for GSIZE, an array for variable-length numbers:
//Let GSIZE[i]=k and GSIZE[j]=l (i < j) be any two numbers in GSIZE.
//Then, the algorithm always ensures that i+k <= j holds.
//All operations except calloc and free are supposed to be O(1).
void *       gsize_calloc( unsigned int n );
void         gsize_free( void *g );
void         gsize_set( void *g, unsigned int pos, unsigned int val );
unsigned int gsize_get( const void *g, unsigned int pos );
void         gsize_clear( void *g, unsigned int pos ); //sets gsize at pos to 0
unsigned int gsize_dec_get( void *g, unsigned int pos ); //first decrement, then read
void         gsize_inc( void *g, unsigned int pos ); //increment

//// GSIZE OPERATIONS /////////////////////////////////////////////////////////
void *gsize_calloc( unsigned int n ) {
    return calloc(n, sizeof(unsigned int));
}

void gsize_free( void *g ) {
    free(g);
}

void gsize_set( void *g, unsigned int pos, unsigned int val ) {
    ((unsigned int *)g)[pos] = val;
}

unsigned int gsize_get( const void *g, unsigned int pos ) {
    return ((const unsigned int *)g)[pos];
}

void gsize_clear( void *g, unsigned int pos ) {
    ((unsigned int *)g)[pos] = 0;
}

unsigned int gsize_dec_get( void *g, unsigned int pos ) {
    return --((unsigned int *)g)[pos];
}

void gsize_inc( void *g, unsigned int pos ) {
    ++((unsigned int *)g)[pos];
}
