#include <gtest/gtest.h>

#include <parallel-range.cpp>

typedef uint uint;



TEST(BV, first_set) {
	BV bv(140);
	bv.fill(false);
	bv.set(7);
	bv.set(99);
	bv.set(131);
	EXPECT_EQ(7, bv.first_set(0,140));
	EXPECT_EQ(7, bv.first_set(7,140));
	EXPECT_EQ(99, bv.first_set(8,140));
	EXPECT_EQ(131, bv.first_set(100,140));
	EXPECT_EQ(99, bv.first_set(61,131));
	EXPECT_EQ(130, bv.first_set(100,130));
}

TEST(BV, reverse_first_set) {
	BV bv(140);
	bv.fill(false);
	bv.set(7);
	bv.set(99);
	bv.set(131);
	EXPECT_EQ(131, bv.reverse_first_set(139,-1));
	EXPECT_EQ(99, bv.reverse_first_set(130,-1));
	EXPECT_EQ(7, bv.reverse_first_set(98,-1));
	EXPECT_EQ(-1, bv.reverse_first_set(6,-1));
	EXPECT_EQ(99, bv.reverse_first_set(99,8));
	EXPECT_EQ(8, bv.reverse_first_set(98,8));
	EXPECT_EQ(7, bv.reverse_first_set(98,7));
}

TEST(BV, reverse_first_set_2) {
	BV bv(140);
	bv.fill(false);
	bv.set(0);
	EXPECT_EQ(0, bv.reverse_first_set(98,-1));
}

TEST(SegmentInfo, constructor) {
	segment_info<uint, 5> segs(7, NULL, NULL);
	EXPECT_EQ(segs.n, 7);	
	EXPECT_EQ(segs.num_blocks, 2);	
	BV expected(vector<bool>({1,0,0,0,0,0,1}));
	EXPECT_EQ(segs.bitvector, expected);	
	EXPECT_EQ(segs.popcount_sum, vector<int64_t>({0,1}));
}

TEST(SegmentInfo, next_one) {
	segment_info<uint, 5> segs(7, NULL, NULL);
	segs.bitvector.init(vector<bool>({1,0,1,0,1,0,1}));	
	segs.update_structure();
	int64_t pos = 0;
	EXPECT_EQ(true, segs.next_one(pos));
	EXPECT_EQ(2, pos);
	EXPECT_EQ(true, segs.next_one(pos));
	EXPECT_EQ(4, pos);
	EXPECT_EQ(true, segs.next_one(pos));
	EXPECT_EQ(6, pos);
	EXPECT_EQ(false, segs.next_one(pos));
}

TEST(SegmentInfo, update_structure) {
	segment_info<uint, 5> segs(9, NULL, NULL);
	segs.bitvector.init(vector<bool>({0,1,1,1,0,0,0,1,0}));
	segs.update_structure();
	EXPECT_EQ(vector<int64_t>({0,3}), segs.popcount_sum);
	segs.bitvector.init(vector<bool>({0,1,1,0,0,0,0,0,0}));
	segs.update_structure();
	EXPECT_EQ(segs.popcount_sum, vector<int64_t>({0,2}));
}

TEST(SegmentInfo, find_first_open) {
	segment_info<uint, 5> segs(9, NULL, NULL);
	segs.bitvector.init(vector<bool>({0,1,1,1,0,0,0,1,0}));	
	segs.update_structure();
	int64_t pos = -42; // Should not matter.
	EXPECT_EQ(true, segs.find_first_open_in_block(pos, 0));
	EXPECT_EQ(1, pos);
	// Last block only contains end not start!.
	EXPECT_EQ(false, segs.find_first_open_in_block(pos, 1));
}


TEST(SegmentInfo, iterate_segments) {
	segment_info<uint, 5> segs(9, NULL, NULL);
	segs.bitvector.init(vector<bool>({1,1,0,1,0,0,0,1,0}));	
	segs.update_structure();
	vector<vector<uint>> result;
	segs.iterate_segments([&result](uint s, uint e) { result.push_back({s,e});});
	EXPECT_EQ( vector<vector<uint>>({{0,1},{3,7}}), result);
}

TEST(SegmentInfo, iterate_segments_blocked) {
	segment_info<uint, 5> segs(9, NULL, NULL);
	segs.bitvector.init(vector<bool>({1,1,0,1,0,0,0,1,0}));
	segs.update_structure();
	vector<vector<uint>> result;
	segs.iterate_segments_blocked([&result](uint s, uint e, uint gs, uint ge) {
			result.push_back({s,e, gs, ge});
			});
	EXPECT_EQ( vector<vector<uint>>({{0,1,0,1},{3,4,3,7}, {5,7,3,7}}), result);
}

TEST(SegmentInfo, iterate_segments_blocked_without_update) {
	segment_info<uint, 5> segs(10, NULL, NULL);
	segs.bitvector.init(vector<bool>({1,0,0,0,0,0,0,0,0,1}));
	vector<vector<uint>> result;
	segs.iterate_segments_blocked([&result](uint s, uint e, uint gs, uint ge) {
			result.push_back({s,e, gs, ge});
			});
	EXPECT_EQ( vector<vector<uint>>({{0,4,0,9},{5,9,0,9}}), result);
}

TEST(SegmentInfo, update_segments) {
	vector<uint> ISA = {0,0,0,0,4,5,5,5};
	vector<uint> SA = {0,1,2,3,4,5,6,7};
	segment_info<uint, 5> segs(8, SA.data(), ISA.data());
	segs.update_segments(0);
	EXPECT_EQ(BV(vector<bool>({1,0,0,1,0,1,0,1})), segs.bitvector);
	EXPECT_EQ(4, segs.ISA[4]);
}

TEST(SegmentInfo, update_names_1) {
	vector<uint> ISA = {0,0,0,0,0,0,6,7};
	vector<uint> SA = {0,1,2,3,4,5,6,7};
	segment_info<uint, 5> segs(8, SA.data(), ISA.data());
	segs.bitvector.init(vector<bool>({0,1,0,1,1,1,0,0}));
	segs.update_structure();
	segs.update_names_1();
	vector<uint> result({0,1,2,3,4,5,6,7});
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_EQ(result[i], segs.ISA[i]);	
	}
}

TEST(SegmentInfo, update_names_2) {
	vector<uint> ISA = {0,0,0,0,0,0,6,7};
	vector<uint> SA = {0,1,2,3,4,5,6,7};
	segment_info<uint, 5> segs(8, SA.data(), ISA.data());
	segs.bitvector.init(vector<bool>({0,1,0,1,1,1,0,0}));
	segs.update_structure();
	segs.update_names_2();
	vector<uint> result({0,1,1,1,4,4,6,7});
	for (size_t i = 0; i < result.size(); ++i) {
		EXPECT_EQ(result[i], segs.ISA[i]);	
	}
}

TEST(SegmentInfo, suffix_sort) {
	vector<uint> ISA = {3,2,1,0,0,0,7,6};
	vector<uint> SA = {0,1,2,3,4,5,6,7};
	segment_info<uint, 5> segs(8, SA.data(), ISA.data());
	segs.bitvector.init(vector<bool>({1,0,0,1,0,0,1,1}));
	segs.update_structure();
	segs.prefix_sort(0);
	EXPECT_EQ(vector<uint>({3,2,1,0,4,5,7,6}), SA);
}

TEST(PackText, simple) {
	vector<uint> text({1,1,1});
	pack_text<uint>(text.data(), text.size());
	EXPECT_EQ((1 << 31), text[2]);
	EXPECT_EQ(((1 << 31) | (1 << 30)), text[1]);
	EXPECT_EQ(((1 << 31) | (1 << 30) | (1 << 29)), text[0]);
}

TEST(PackText, different_chars) {
	vector<uint> text({0,3,1});
	pack_text<uint>(text.data(), text.size());
	EXPECT_EQ((1 << 30), text[2]);
	EXPECT_EQ(((3 << 30) | (1 << 28)), text[1]);
	EXPECT_EQ(((3 << 28) | (1 << 26)), text[0]);
}


TEST(ParallelTrSort, error) {
	// 	   SA: 0123456789
	string text = "banananaaa";
	vector<uint> ISA,SA;
	for (char c : text) {
		ISA.push_back(c);
		SA.push_back(SA.size());
	}
	paralleltrsort(ISA.data(), SA.data(), (uint)SA.size());
	vector<uint> expectSA = {9,8,7,5,3,1,0,6,4,2};
	EXPECT_EQ(expectSA, SA);
}

TEST(ParallelTrSort, 64bit) {
	// 	   SA: 0123456789
	string text = "banananaaa";
	vector<uint64_t> ISA,SA;
	for (char c : text) {
		ISA.push_back(c);
		SA.push_back(SA.size());
	}
	paralleltrsort(ISA.data(), SA.data(), (uint64_t)SA.size());
	vector<uint64_t> expectSA = {9,8,7,5,3,1,0,6,4,2};
	EXPECT_EQ(expectSA, SA);
}

TEST(ParallelTrSort, Repetition) {
	// 	   SA: 999....0
	string text(1000, 'a'); // = a^1000
	vector<uint> ISA,SA;
	for (char c : text) {
		ISA.push_back(c);
		SA.push_back(SA.size());
	}
	paralleltrsort(ISA.data(), SA.data(), (uint)SA.size());
	vector<uint> expectSA(1000);
	for (uint i = 0; i < 1000; ++i)
		expectSA[i] = 999-i;
	EXPECT_EQ(expectSA, SA);
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
