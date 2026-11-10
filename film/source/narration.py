# "Pass It Up": the fourfold story, part two (2026-11-10). Re-authored from "The Long Way Round" (media/long_way), which is the
# rough draft. Sources: fourfold/TREE-RING-1109.md, FLOODING-BRIEF-1109.md, the run records under fourfold/runs/*-1110.
# No benchmark is named.
# Each line: id, spoken text, caption text (None = same as spoken), gap after (s).
_L = [
 ("title", None, None, 1.4),
 # A. Where we left off
 ("a1", "Last night, sixteen monkeys found the exact sum of their private numbers. Each one is a language model.",
  "Last night, sixteen monkeys found the exact sum of their private numbers. Each one is a language model: Qwen3.8-27B.", .15),
 ("a2", "They did it by flooding. Every monkey collected every other monkey's number, two to a port.",
  None, .15),
 ("a3", "So we made the grid bigger.", None, .4),
 # B. The wall
 ("b1", "One hundred monkeys. Sixty ticks later, not one had the sum.",
  "One hundred monkeys. Sixty ticks later, not one had the sum: 0 of 100.", .15),
 ("b2", "Each held about ninety of the hundred numbers. Several announced the network was complete.",
  "Each held about 90 of the 100 numbers. Several announced the network was complete.", .15),
 ("b3", "It wasn't the monkeys. A port carries two numbers a tick. Even perfect agents need fifty ticks to flood a hundred.",
  "It wasn't the monkeys. A port carries two numbers a tick. Even perfect agents need 50 ticks to flood 100.", .15),
 ("b4", "And every monkey's notes grow with the crowd.",
  "And every monkey's notes grow with the crowd: prompts went from 1,300 tokens to 5,600.", .4),
 # C. The old idea
 ("c1", "The fix is from nineteen eighty-two. Don't make everyone know everything. Add it up along a tree.",
  "The fix is from 1982. Don't make everyone know everything. Add it up along a tree.", .15),
 ("c2", "One monkey starts a wave. Whoever it reaches takes that neighbour as a parent, and passes the wave on.",
  None, .15),
 ("c3", "When everyone below a monkey has answered, it sends up one message: a sum, and a count. The host does the adding.",
  None, .15),
 ("c4", "The root gets the total, and knows it's complete. Then the answer runs back down.",
  None, .4),
 # D. The result
 ("d1", "Same hundred monkeys. Same model. Tick twenty-three: the first exact total. Tick thirty-three: all one hundred.",
  "Same hundred monkeys. Same model. Tick 23: the first exact total. Tick 33: all one hundred.", .15),
 ("d2", "Forty-nine thousand, six hundred eighty-nine. No wrong answers. A perfect tree takes thirty-two ticks. They took thirty-three.",
  "49,689. No wrong answers. A perfect tree takes 32 ticks. They took 33.", .15),
 ("d3", "Each tick ran in half the time, and the prompts stayed small.",
  "Each tick ran in half the time (202 s against 375), and the prompts stayed small.", .4),
 # E. The weakness
 ("e1", "A tree has one weakness. One slip, and a whole branch vanishes without a sound.", None, .15),
 ("e2", "It happened. One monkey waved at its own parent, and eleven monkeys dropped out of the count. The total looked fine. It was wrong.",
  "It happened, in a Claude Haiku run. One monkey waved at its own parent, and 11 dropped out of the count: 44,294 over 89.", .15),
 ("e3", "So grow two trees from two roots, and trust the answer only when they agree. The second tree caught it.",
  None, .15),
 ("e4", "Now the host refuses that write. The rerun: one hundred of one hundred.",
  "Now the host refuses that write. The rerun: 100 of 100, both trees agreeing.", .4),
 # G. Other minds
 ("g1", "It isn't one model's trick. A swarm of one hundred twelve Claude Haiku agents, across seven arrays joined by slow seams: all exact, for half the tokens.",
  "It isn't one model's trick. A swarm of 112 Claude Haiku agents, across seven arrays joined by slow seams: 112 of 112 exact, for half the tokens.", .4),
 # F. The tesseract: the cliffhanger
 ("f1", "Then the big one. Two hundred fifty-six monkeys on a tesseract: four dimensions, one ring through every port.",
  "Then the big one. 256 monkeys on a tesseract: four dimensions, one ring through every port.", .15),
 ("f2", "No tree here. Each monkey sends one entry around its ring, and everyone it passes adds their own. Four rings, and you hold the total.",
  None, .15),
 ("f3", "Perfect agents need seventeen ticks. Flooding would need one hundred twenty-eight.",
  "Perfect agents need 17 ticks. Flooding would need 128.", .15),
 ("f4", "It broke. One monkey's number was counted twice, and every monkey after it passed the mistake along.",
  None, .15),
 ("f5", "One hundred eighty-six of two hundred fifty-six exact. Seventy confidently wrong. And the twenty-seven B failed with only sixteen.",
  "Claude Haiku: 186 of 256 exact, 70 confidently wrong. And Qwen3.8-27B failed with only sixteen.", .15),
 ("f6", "A ring has no second tree to catch a slip. So next, the host checks every write. That's tomorrow.",
  None, .5),
 # H. The lesson
 ("h1", "Nobody has to know everything. But somebody has to check.", None, .3),
 ("h2", "Pass it up, little monkeys. We keep the ledger.", None, 1.2),
]
LINES = [(i, s, c if c else s, g) for i, s, c, g in _L]
