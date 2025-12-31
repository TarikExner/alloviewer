from __future__ import annotations
from typing import Optional
import json
import os
import uuid
from ..structs import ParsedPlateLayout, dc_to_dict, parsed_plate_from_dict

class LayoutRepo:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path_for_id(self, upload_id: str) -> str:
        return os.path.join(self.root, f"{upload_id}.json")

    def _path_for_sha(self, sha256: str) -> str:
        return os.path.join(self.root, f"sha_{sha256}.link")

    def save_layout(self, layout: ParsedPlateLayout) -> str:
        if not layout.upload_id:
            layout.upload_id = f"pl_{uuid.uuid4().hex[:10]}"
        with open(self._path_for_id(layout.upload_id), "w", encoding="utf-8") as f:
            json.dump(dc_to_dict(layout), f, ensure_ascii=False, indent=2)
        with open(self._path_for_sha(layout.sha256), "w", encoding="utf-8") as f:
            f.write(layout.upload_id)
        return layout.upload_id

    def get_by_id(self, upload_id: str) -> Optional[ParsedPlateLayout]:
        p = self._path_for_id(upload_id)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return parsed_plate_from_dict(data)

    def find_by_sha(self, sha256: str) -> Optional[ParsedPlateLayout]:
        link = self._path_for_sha(sha256)
        if not os.path.exists(link):
            return None
        with open(link, "r", encoding="utf-8") as f:
            upload_id = f.read().strip()
        return self.get_by_id(upload_id)

