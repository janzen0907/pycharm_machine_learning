from pathlib import Path

path = Path('python.txt')
path.write_text("We're learning Python!")
# write_text opens the file, writes the text, closes the file
path.write_text("Learning more Python")

# Build up the string before we write it to the file
text_to_write = "We're learning Python"
text_to_write += "\nAnd Now we are learning more about it "
text_to_write += "\n"
path.write_text(text_to_write)
# Must covnert ints to strings, cannnot take in ints
path.write_text(str(333))