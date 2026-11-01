from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Human:
	"""A human with an integer identifier and a photo path."""

	id: int
	photo: Union[str, Path]

	def __post_init__(self) -> None:
		if not isinstance(self.id, int):
			raise TypeError("id must be an int")
		self.photo = Path(self.photo)

raise

list_of_humans = []

def load_humans_from_directory(directory: Union[str, Path]) -> None:
	"""Load humans from a directory containing images.

	Args:
		directory (Union[str, Path]): The directory containing human images.
	"""
	directory = Path(directory)
	if not directory.is_dir():
		raise ValueError(f"{directory} is not a valid directory.")

	for image_path in sorted(directory.iterdir()):
		if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
			continue

		human_id = len(list_of_humans) + 1
		human = Human(id=human_id, photo=image_path)
		list_of_humans.append(human)
        


