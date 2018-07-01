#ifndef BWT_H
#define BWT_H

//=============================================================================================
// BWT demo using the MSufSort algorithm.
//
// Author: M.A. Maniscalco
// Date: 7/30/04
// email: michael@www.michael-maniscalco.com
//
// This code is free for non commercial use only.
//
//=============================================================================================



#include "MSufSort.h"




class BWT
{
public:
	BWT();

	virtual ~BWT();

	unsigned int Forward(SYMBOL_TYPE * data, unsigned int length);

	void Reverse(SYMBOL_TYPE * data, unsigned int length, unsigned int index);

	unsigned int MSufSortTime(){return m_suffixSorter->GetElapsedSortTime();}

	bool VerifySort(){return m_suffixSorter->VerifySort();}
private:

	MSufSort *			m_suffixSorter;
};




#endif