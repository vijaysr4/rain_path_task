import py7zr
import os
from pathlib import Path



def extract_7z_archive(archive_path: Path, output_dir: Path) -> None:
    """
    Extracts a .7z archive to the given output directory.

    :param archive_path: Path to the .7z file.
    :param output_dir: Directory where contents will be extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(str(archive_path), mode='r') as archive:
        archive.extractall(path=str(output_dir))


if __name__ == "__main__":
    # Paths
    archive = Path("/Data/rainpath/imgs_HEMIT/HEMIT.7z")
    destination = Path("/Data/rainpath/HEMIT")

    extract_7z_archive(archive, destination)
    print(f"Extraction complete: {destination}")


