from glob import glob
from dataclasses import dataclass
from os.path import abspath, expanduser, join

from scal3r.utils.image_utils import apply_frame_range_to_sorted_paths

@dataclass
class ImageFolderDataset:
    input_dir: str
    image_patterns: tuple[str, ...]
    max_images: int | None = None
    start_frame: int = 0
    end_frame: int = -1
    interval: int = 1

    def list_images(self) -> list[str]:
        images: list[str] = []
        for pattern in self.image_patterns:
            images.extend(glob(join(self.input_dir, pattern)))
        resolved = sorted({abspath(expanduser(path)) for path in images})
        resolved = apply_frame_range_to_sorted_paths(
            resolved,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            interval=self.interval,
        )
        if self.max_images:
            return resolved[: self.max_images]
        return resolved
