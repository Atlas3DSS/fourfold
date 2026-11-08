"""tests for fourfold/tools.py: run with  python3 -m unittest tests.test_tools  (or pytest)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import AgentFS, compute, READ_CAP  # noqa: E402


class TestAgentFS(unittest.TestCase):
    def setUp(self):
        self.fs = AgentFS()

    def test_write_and_read(self):
        self.assertEqual(self.fs.call("write_file", name="lists/shopping.txt", text="milk\neggs"), "saved\n-lists/shopping.txt")
        self.assertEqual(self.fs.call("read_file", name="lists/shopping.txt"), "milk\neggs")
        self.assertEqual(self.fs.call("write_file", name="lists/shopping.txt", text="bread"), "saved\n-lists/shopping.txt")
        self.assertEqual(self.fs.call("read_file", name="lists/shopping.txt"), "bread")   # replaces

    def test_append_on_new_line(self):
        self.fs.call("write_file", name="a.txt", text="milk")
        self.assertEqual(self.fs.call("append_file", name="a.txt", text="bread"), "saved\n-a.txt")
        self.assertEqual(self.fs.call("read_file", name="a.txt"), "milk\nbread")
        self.assertEqual(self.fs.call("append_file", name="new.txt", text="x"), "saved\n-new.txt")   # creates
        self.assertEqual(self.fs.call("read_file", name="new.txt"), "x")

    def test_read_missing_and_bad_paths(self):
        self.assertEqual(self.fs.call("read_file", name="notes/ideas.txt"), "not found")
        self.assertTrue(self.fs.call("write_file", name="../x", text="t").startswith("error"))
        self.assertTrue(self.fs.call("write_file", name="", text="t").startswith("error"))
        self.assertEqual(self.fs.call("write_file", name="/people/sam.txt/", text="t"), "saved\n-people/sam.txt")

    def test_list_files_order_and_empty(self):
        self.assertEqual(self.fs.call("list_files"), "empty")
        for p in ("lists/shopping.txt", "people/sam.txt", "projects/website.txt"):
            self.fs.call("write_file", name=p, text=p)
        self.assertEqual(self.fs.call("list_files"), "-lists/shopping.txt\n-people/sam.txt\n-projects/website.txt")

    def test_links_and_backlinks(self):
        self.fs.call("write_file", name="people/sam.txt", text="Sam: engineer in Berlin")
        self.fs.call("write_file", name="projects/website.txt", text="launch May 5")
        self.assertEqual(self.fs.call("link_files", name="people/sam.txt", target="projects/website.txt"),
                         "linked\n-people/sam.txt\nto\n-projects/website.txt")
        self.assertEqual(self.fs.call("link_files", name="people/sam.txt", target="projects/website.txt"),
                         "linked\n-people/sam.txt\nto\n-projects/website.txt")       # no-op repeat, still one link
        self.assertEqual(len(self.fs.links), 1)
        self.assertEqual(self.fs.call("list_links", file="people/sam.txt"), "-projects/website.txt")
        self.assertEqual(self.fs.call("list_backlinks", path="projects/website.txt"), "-people/sam.txt")
        self.assertEqual(self.fs.call("list_backlinks", path="people/sam.txt"), "empty")
        self.assertEqual(self.fs.call("unlink_files", name="people/sam.txt", target="projects/website.txt"),
                         "removed\n-people/sam.txt\nto\n-projects/website.txt")
        self.assertEqual(self.fs.call("list_links", file="people/sam.txt"), "empty")
        self.assertEqual(self.fs.call("link_files", name="people/sam.txt", target="nope.txt"), "not found")
        self.assertEqual(self.fs.call("list_links", file="nope.txt"), "not found")

    def test_search_whole_word_case_insensitive_and_current_text(self):
        self.fs.call("write_file", name="people/sam.txt", text="Sam is an engineer in Berlin.")
        self.fs.call("write_file", name="notes.txt", text="Berliner pastries")
        self.assertEqual(self.fs.call("search_files", query="berlin"), "-people/sam.txt")   # whole word, any case
        self.assertEqual(self.fs.call("search_files", query="Paris"), "empty")
        self.fs.call("write_file", name="people/sam.txt", text="Sam moved to Tokyo.")
        self.assertEqual(self.fs.call("search_files", query="Berlin"), "empty")             # old word is gone
        self.assertEqual(self.fs.call("search_files", query="Tokyo"), "-people/sam.txt")
        self.assertEqual(self.fs.call("search_files", query=""), "empty")

    def test_file_cap(self):
        for i in range(AgentFS.MAX_FILES):
            self.assertEqual(self.fs.call("write_file", name=f"f/{i}.txt", text=str(i)), f"saved\n-f/{i}.txt")
        r = self.fs.call("write_file", name="one-too-many.txt", text="x")
        self.assertTrue(r.startswith("error") and "128" in r, r)
        self.assertTrue(self.fs.call("append_file", name="also-new.txt", text="x").startswith("error"))
        self.assertEqual(self.fs.call("write_file", name="f/0.txt", text="rewrite"), "saved\n-f/0.txt")   # existing is fine
        self.assertEqual(self.fs.call("append_file", name="f/1.txt", text="more"), "saved\n-f/1.txt")
        self.assertEqual(len(self.fs.files), AgentFS.MAX_FILES)

    def test_read_cap(self):
        self.fs.call("write_file", name="big.txt", text="x" * (READ_CAP + 100))
        r = self.fs.call("read_file", name="big.txt")
        self.assertTrue(r.startswith("x" * READ_CAP) and "[cut: 100 more characters]" in r)

    def test_unknown_tool_and_bad_args(self):
        self.assertTrue(self.fs.call("delete_file", name="x").startswith("error: unknown tool"))
        self.assertTrue(self.fs.call("write_file", name="x").startswith("error: write_file takes"))
        self.assertTrue(self.fs.call("read_file").startswith("error"))

    def test_json_round_trip(self):
        self.fs.call("write_file", name="a.txt", text="alpha\nline two")
        self.fs.call("write_file", name="b/c.txt", text="beta")
        self.fs.call("link_files", name="a.txt", target="b/c.txt")
        d = json.loads(json.dumps(self.fs.to_json()))
        fs2 = AgentFS.from_json(d)
        self.assertEqual(fs2.call("list_files"), "-a.txt\n-b/c.txt")
        self.assertEqual(fs2.call("read_file", name="a.txt"), "alpha\nline two")
        self.assertEqual(fs2.call("list_links", file="a.txt"), "-b/c.txt")
        self.assertEqual(fs2.to_json(), self.fs.to_json())
        self.assertEqual(AgentFS.from_json({}).call("list_files"), "empty")


class TestCompute(unittest.TestCase):
    def test_basics(self):
        self.assertEqual(compute("1234"), "= 1234")
        self.assertEqual(compute("2 + 3 * 4"), "= 14")
        self.assertEqual(compute("(2 + 3) * 4"), "= 20")
        self.assertEqual(compute("7 / 3"), "= 7/3")
        self.assertEqual(compute("6 / 3"), "= 2")
        self.assertEqual(compute("1/3 + 1/6"), "= 1/2")
        self.assertEqual(compute("-7 // 2"), "= -4")
        self.assertEqual(compute("7 % 3"), "= 1")
        self.assertEqual(compute("2 ** 10"), "= 1024")
        self.assertEqual(compute("2 ** -2"), "= 1/4")
        self.assertEqual(compute("-(3 - 5)"), "= 2")
        self.assertEqual(compute("+4"), "= 4")

    def test_functions(self):
        self.assertEqual(compute("sum(1, 2, 3)"), "= 6")
        self.assertEqual(compute("sum(1, sum(2, 3), max(4, 5) - min(1, 2))"), "= 10")
        self.assertEqual(compute("min(1/2, 1/3)"), "= 1/3")
        self.assertEqual(compute("max(2**3, 3**2)"), "= 9")
        self.assertTrue(compute("sum()").startswith("error"))
        self.assertTrue(compute("abs(-1)").startswith("error"))

    def test_big_integers(self):
        self.assertEqual(compute("123456789012345678901234567890 + 1"), "= 123456789012345678901234567891")
        self.assertTrue(compute("1234567890123456789012345678901").startswith("error"))   # 31-digit literal
        self.assertEqual(compute("10 ** 64"), "= 1" + "0" * 64)

    def test_errors(self):
        self.assertEqual(compute("1 / 0"), "error: division by zero")
        self.assertEqual(compute("1 // 0"), "error: division by zero")
        self.assertEqual(compute("5 % 0"), "error: division by zero")
        self.assertEqual(compute("0 ** -1"), "error: division by zero")
        self.assertTrue(compute("2 ** 65").startswith("error: exponent"))
        self.assertTrue(compute("10 ** 64 ** 2").startswith("error"))          # huge exponent, nothing computed
        self.assertTrue(compute("(10**30) ** 7").startswith("error: result exceeds"))
        self.assertTrue(compute("2 ** (1/2)").startswith("error"))
        self.assertTrue(compute("1.5 + 1").startswith("error"))
        self.assertTrue(compute("x + 1").startswith("error"))
        self.assertTrue(compute("__import__('os')").startswith("error"))
        self.assertTrue(compute("1 if 2 else 3").startswith("error"))
        self.assertTrue(compute("[1,2]").startswith("error"))
        self.assertTrue(compute("1 +").startswith("error: cannot parse"))
        self.assertTrue(compute("").startswith("error"))
        self.assertTrue(compute("1" + " + 1" * 150).startswith("error: expression longer"))
        self.assertTrue(compute("(" * 200 + "1" + ")" * 200).startswith("error"))
        self.assertTrue(compute(42).startswith("error"))

    def test_deterministic(self):
        self.assertEqual(compute("sum(1/3, 1/3, 1/3) * 3 - 1"), "= 2")
        self.assertEqual(compute("sum(1/3, 1/3, 1/3) * 3 - 1"), "= 2")


if __name__ == "__main__":
    unittest.main()
