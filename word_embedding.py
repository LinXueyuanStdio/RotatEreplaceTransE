import polyglot
from polyglot.text import Text, Word
word = Word("Obama", language="en")
print("\n\nthe {} dimensions\n".format(word.vector.shape[0]))
print(word.vector[:200])