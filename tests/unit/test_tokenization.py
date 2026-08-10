from __future__ import annotations

import unittest

from rag_kb.document_processing.tokenization import (
    count_chunk_tokens,
    split_by_tokens,
)


class TokenizationTests(unittest.TestCase):
    def test_mixed_unicode_windows_remain_within_the_token_limit(self) -> None:
        # OCR output commonly mixes CJK characters, ASCII and combining marks.
        # A raw token window can begin or end inside one of those characters.
        text = (
            "ͫP̛仭专乶乲yQ丠京Y]亍丐他o习亟乴&仵产乻D丱亸丅͜jD̅专不R亳乿亁丨乤乁V̸̜̓"
            "举仦̍产万8丒͜1丙乌仌丷̈́C͡亩[4̐͐仯乱>A̅̾丶丐亯亁̫仟ͥ͠仡仕)亨亅仠五丬乬仈͝6"
            "乊̢乫亪=亾亇以亴̹乎仲丏义ͭ乂三今仼丙亝ͧ买̵I丼仹̘͂ͬ亻a亊͘亃井乗ͩI不仡业͌x亹"
            "仗丼今̲̏乴̉亪V*乷乶亶̘仪丳乻乀̿ͩ̃"
        )

        parts = split_by_tokens(text, max_tokens=160, overlap_tokens=50)

        self.assertGreater(len(parts), 1)
        self.assertTrue(all(count_chunk_tokens(part) <= 160 for part in parts))
        self.assertTrue(all(part in text for part in parts))
        self.assertNotIn("\ufffd", "".join(parts))


if __name__ == "__main__":
    unittest.main()
