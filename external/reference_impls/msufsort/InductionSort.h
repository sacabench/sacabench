#ifndef MSUFSORT_INDUCTION_SORTING_H
#define MSUFSORT_INDUCTION_SORTING_H

#include "IntroSort.h"


class InductionSortObject
{
public:
	InductionSortObject(unsigned int inductionPosition = 0, unsigned int inductionValue = 0, unsigned int suffixIndex = 0);

	bool operator <= (InductionSortObject & object);

	bool operator == (InductionSortObject & object);

	InductionSortObject& operator = (InductionSortObject & object);

	bool operator >= (InductionSortObject & object);

	bool operator > (InductionSortObject & object);

	bool operator < (InductionSortObject & object);

	unsigned int	m_sortValue[2];
};


inline bool InductionSortObject::operator <= (InductionSortObject & object)
{
	if (m_sortValue[0] < object.m_sortValue[0])
		return true;
	else
		if (m_sortValue[0] == object.m_sortValue[0])
			return (m_sortValue[1] <= object.m_sortValue[1]);
	return false;
}



inline bool InductionSortObject::operator == (InductionSortObject & object)
{
	return ((m_sortValue[0] == object.m_sortValue[0]) && (m_sortValue[1] == object.m_sortValue[1]));
}



inline bool InductionSortObject::operator >= (InductionSortObject & object)
{
	if (m_sortValue[0] > object.m_sortValue[0])
		return true;
	else
		if (m_sortValue[0] == object.m_sortValue[0])
			return (m_sortValue[1] >= object.m_sortValue[1]);
	return false;
}



inline InductionSortObject & InductionSortObject::operator = (InductionSortObject & object)
{
	m_sortValue[0] = object.m_sortValue[0];
	m_sortValue[1] = object.m_sortValue[1];
	return *this;
}




inline bool InductionSortObject::operator > (InductionSortObject & object)
{
	if (m_sortValue[0] > object.m_sortValue[0])
		return true;
	else
		if (m_sortValue[0] == object.m_sortValue[0])
			return (m_sortValue[1] > object.m_sortValue[1]);
	return false;
}



inline bool InductionSortObject::operator < (InductionSortObject & object)
{
	if (m_sortValue[0] < object.m_sortValue[0])
		return true;
	else
		if (m_sortValue[0] == object.m_sortValue[0])
			return (m_sortValue[1] < object.m_sortValue[1]);
	return false;
}




#endif
