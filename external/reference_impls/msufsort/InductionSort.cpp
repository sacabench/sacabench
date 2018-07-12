#include "InductionSort.h"

InductionSortObject::InductionSortObject(unsigned int inductionPosition, unsigned int inductionValue, 
										 unsigned int suffixIndex)
{
	// sort value is 64 bits long.
	// bits are ...
	// 63 - 60: induction position (0 - 15)
	// 59 - 29: induction value at induction position (0 - (2^30 -1))
	// 28 - 0:  suffix index for the suffix sorted by induction (0 - (2^30) - 1)
	m_sortValue[0] = inductionPosition << 28;
	m_sortValue[0] |= ((inductionValue & 0x3fffffff) >> 2);
	m_sortValue[1] = (inductionValue << 30);
	m_sortValue[1] |= suffixIndex;
}