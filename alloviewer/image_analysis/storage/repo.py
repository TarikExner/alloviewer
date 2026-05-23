from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from ..structs import ParsedPlateLayout, dc_to_dict, parsed_plate_from_dict


class LayoutRepo:
    """File-based repository for parsed plate layouts.

    Layouts are stored as JSON files under ``root``. Each layout is saved by
    ``upload_id`` and indexed by SHA-256 checksum through a small link file.

    Parameters
    ----------
    root : str
        Directory used to store layout JSON files and SHA link files.
    """

    def __init__(self, root: str):
        """Initialize the repository directory.

        Parameters
        ----------
        root : str
            Storage directory. Created if it does not already exist.
        """
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path_for_id(self, upload_id: str) -> str:
        """Return the JSON path for an upload ID.

        Parameters
        ----------
        upload_id : str
            Layout upload ID.

        Returns
        -------
        str
            Path to the JSON file for the layout.
        """
        return os.path.join(self.root, f"{upload_id}.json")

    def _path_for_sha(self, sha256: str) -> str:
        """Return the link-file path for a SHA-256 checksum.

        Parameters
        ----------
        sha256 : str
            SHA-256 checksum of the uploaded layout source.

        Returns
        -------
        str
            Path to the SHA link file.
        """
        return os.path.join(self.root, f"sha_{sha256}.link")

    def save_layout(self, layout: ParsedPlateLayout) -> str:
        """Save a parsed plate layout.

        Parameters
        ----------
        layout : ParsedPlateLayout
            Parsed layout object to save. If ``layout.upload_id`` is empty, a
            new ID is generated and written back to the object.

        Returns
        -------
        str
            Upload ID of the saved layout.

        Notes
        -----
        Saving creates or replaces two files: the layout JSON file and the
        SHA-256 link file.
        """
        if not layout.upload_id:
            layout.upload_id = f"pl_{uuid.uuid4().hex[:10]}"

        with open(self._path_for_id(layout.upload_id), "w", encoding="utf-8") as f:
            json.dump(dc_to_dict(layout), f, ensure_ascii=False, indent=2)

        with open(self._path_for_sha(layout.sha256), "w", encoding="utf-8") as f:
            f.write(layout.upload_id)

        return layout.upload_id

    def get_by_id(self, upload_id: str) -> Optional[ParsedPlateLayout]:
        """Load a parsed plate layout by upload ID.

        Parameters
        ----------
        upload_id : str
            Layout upload ID.

        Returns
        -------
        ParsedPlateLayout or None
            Parsed layout object, or ``None`` if no matching layout file
            exists.
        """
        p = self._path_for_id(upload_id)

        if not os.path.exists(p):
            return None

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        return parsed_plate_from_dict(data)

    def find_by_sha(self, sha256: str) -> Optional[ParsedPlateLayout]:
        """Load a parsed plate layout by SHA-256 checksum.

        Parameters
        ----------
        sha256 : str
            SHA-256 checksum of the uploaded layout source.

        Returns
        -------
        ParsedPlateLayout or None
            Parsed layout object, or ``None`` if no SHA link file exists.
        """
        link = self._path_for_sha(sha256)

        if not os.path.exists(link):
            return None

        with open(link, "r", encoding="utf-8") as f:
            upload_id = f.read().strip()

        return self.get_by_id(upload_id)
