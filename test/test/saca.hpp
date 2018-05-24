/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#pragma once

#include <gtest/gtest.h>
#include <util/alphabet.hpp>
#include <util/sa_check.hpp>

namespace test {
    /// Helper function to run a SA construction algorithm
    /// on a number of short test strings.
    ///
    /// The list of test strings can be freely extended as needed,
    /// since every algorithm should always be able to handle any Input.
    ///
    /// Example:
    /// ```cpp
    /// test::saca_corner_cases<MyAlgorithm>();
    /// ```
    template<typename Algorithm, typename sa_index_type = size_t>
    void saca_corner_cases() {
        using namespace sacabench::util;

        auto test = [](string_span text) {
            if (text.size() > 40) {
                std::cout << "Test SACA on '" << text.slice(0, 40) << "[...]'\n";
            } else {
                std::cout << "Test SACA on '" << text << "'\n";
            }

            auto copy = make_string(text);
            auto alphabet = apply_effective_alphabet(copy);
            auto output = make_container<sa_index_type>(text.size());
            span<sa_index_type> output_span = output;

            Algorithm::construct_sa(copy, alphabet.size, output_span);

            auto fast_result = sa_check(output_span, text);
            if (fast_result != sa_check_result::ok) {
                auto slow_result = sa_check_naive(output_span, text);
                ASSERT_EQ(fast_result, slow_result) << "BUG IN SA CHECKER DETECTED!";
                ASSERT_EQ(fast_result, sa_check_result::ok);
            }
        };

        test(""_s);
        test("hello world"_s);
        test("caabaccaabacaa"_s);
        test("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"_s);

        // All test strings below are take fromn tudocomp
        test("abcdebcdeabc"_s);
        test("a"_s);

        test("aaaaaaaaa"_s); \
        test("banana"_s); \
        test("ananas"_s); \
        test("abcdefgh#defgh_abcde"_s); \

        test("abcdebcdeabcd"_s);
        test("foobar"_s);
        test("abcabcabcabc"_s);

        test("abc abc  abc"_s);

        test("abaaabbababb"_s);

        test(
            "asdfasctjkcbweasbebvtiwetwcnbwbbqnqxernqzezwuqwezuet"
            "qcrnzxbneqebwcbqwicbqcbtnqweqxcbwuexcbzqwezcqbwecqbw"
            "dassdasdfzdfgfsdfsdgfducezctzqwebctuiqwiiqcbnzcebzqc"_s);

        test("ประเทศไทย中华Việt Nam"_s);

        test(
            "Lorem ipsum dolor sit amet, sea ut etiam solet salut"
            "andi, sint complectitur et his, ad salutandi imperdi"
            "et gubergren per mei."_s);

        test(
            "Лорэм атоморюм ут хаж, эа граэки емпыдит ёудёкабет "
            "мэль, декам дежпютатионй про ты. Нэ ёужто жэмпэр"
            " жкрибэнтур векж, незл коррюмпит."_s);

        test(
            "報チ申猛あち涙境ワセ周兵いわ郵入せすをだ漏告されて話巡わッき"
            "や間紙あいきり諤止テヘエラ鳥提フ健2銀稿97傷エ映田ヒマ役請多"
            "暫械ゅにうて。関国ヘフヲオ場三をおか小都供セクヲ前俳著ゅ向深"
            "まも月10言スひす胆集ヌヱナ賀提63劇とやぽ生牟56詰ひめつそ総愛"
            "ス院攻せいまて報当アラノ日府ラのがし。"_s);

        test(
            "Εαμ ανσιλλαε περισυλα συαφιθαθε εξ, δυο ιδ ρεβυμ σομ"
            "μοδο. Φυγιθ ηομερω ιυς ατ, ει αυδιρε ινθελλεγαμ νες."
            " Ρεκυε ωμνιυμ μανδαμυς κυο εα. Αδμοδυμ σωνσεκυαθ υθ "
            "φιξ, εσθ ετ πρωβατυς συαφιθαθε ραθιονιβυς, ταντας αυ"
            "διαμ ινστρυσθιορ ει σεα."_s);

        test("struct Foo { uint8_t bar }"_s);

        test("ABBCBCABA"_s);

        test("abcabca"_s);

        test("abbbbbbbbbbcbbbbbbbbbb"_s);
        test("abbbcbbb"_s);

        test("0	100009425	0.1661:0.1661	#businessfor"_s);
    }
}
