from pathlib import Path
import random
import shutil

"""
Raw data : ERON-SPAM

V. Metsis, I. Androutsopoulos and G. Paliouras, "Spam Filtering with 
Naive Bayes - Which Naive Bayes?". Proceedings of the 3rd Conference 
on Email and Anti-Spam (CEAS 2006), Mountain View, CA, USA, 2006.
"""


base_dir = Path(__file__).resolve().parent
source_dir = base_dir / "raw data" / "groupe"
target_dir = base_dir / "test"


def generate_test_data() -> None:
	files = [path for path in source_dir.iterdir() if path.is_file()]
	random.SystemRandom().shuffle(files)

	target_dir.mkdir(parents=True, exist_ok=True)

	for index, source_file in enumerate(files, start=1):
		shutil.copy2(source_file, target_dir / f"email{index}.txt")


if __name__ == "__main__":
	generate_test_data()
