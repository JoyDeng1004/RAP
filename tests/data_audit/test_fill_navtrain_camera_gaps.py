import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/data_audit/fill_navtrain_camera_gaps.py"
SPEC = importlib.util.spec_from_file_location("gap_fill", SCRIPT)
gap_fill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gap_fill)


def jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="JPEG")
    return buffer.getvalue()


class CameraGapTests(unittest.TestCase):
    def test_allowlist_contains_only_46_camera_gaps(self):
        manifest = gap_fill.load_manifest()
        self.assertEqual(len(manifest["paths"]), 46)
        self.assertEqual({row["shard"] for row in manifest["archives"]}, {52, 98, 171})
        self.assertEqual(sum(row["size"] for row in manifest["archives"]), 17268830574)
        self.assertFalse(any("missing_camera" in path for path in manifest["paths"]))

    def test_archive_extracts_only_requested_jpeg(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "camera.tgz"
            requested = "log/CAM_F0/wanted.jpg"
            content = jpeg_bytes()
            with tarfile.open(archive, "w:gz") as bundle:
                for name in ("../../" + requested, "prefix/log/RADAR_BACK_RIGHT/radar.bin",
                             "prefix/log/CAM_F0/extra.jpg", "prefix/" + requested):
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    bundle.addfile(member, io.BytesIO(content))
            gap_fill.extract_selected(archive, [requested], root / "stage")
            self.assertEqual((root / "stage" / requested).read_bytes(), content)
            self.assertFalse((root / "stage/log/RADAR_BACK_RIGHT").exists())
            self.assertFalse((root / "stage/log/CAM_F0/extra.jpg").exists())

    def test_publish_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "image.jpg"
            target.write_bytes(b"original")
            with self.assertRaises(FileExistsError):
                gap_fill.publish(io.BytesIO(jpeg_bytes()), target)
            self.assertEqual(target.read_bytes(), b"original")

    def test_bad_image_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "image.jpg"
            with self.assertRaises(OSError):
                gap_fill.publish(io.BytesIO(b"not an image"), target)
            self.assertFalse(target.exists())

    def test_path_cannot_escape_via_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "stage"
            root.mkdir()
            (root / "log").symlink_to(Path(temporary), target_is_directory=True)
            with self.assertRaises(ValueError):
                gap_fill.checked_path(root, "log/CAM_F0/image.jpg")

    def test_stage_refuses_archive_download_without_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = gap_fill.argparse.Namespace(mode="stage", sensor_root=str(root / "live"),
                staging=str(root / "stage"), file_urls=None, allow_archive_transfer=False)
            with patch.dict(gap_fill.os.environ, {"JOB_ID": "test"}), patch.object(gap_fill, "fetch") as fetch:
                with self.assertRaisesRegex(ValueError, "refusing redundant archive transfer"):
                    gap_fill.run(args)
                fetch.assert_not_called()

    def test_present_images_skip_all_transfers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = jpeg_bytes()
            for relative in gap_fill.load_manifest()["paths"]:
                target = root / "live" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            args = gap_fill.argparse.Namespace(mode="stage", sensor_root=str(root / "live"),
                staging=str(root / "stage"), file_urls=None, allow_archive_transfer=False)
            with patch.dict(gap_fill.os.environ, {"JOB_ID": "test"}), patch.object(gap_fill, "fetch") as fetch:
                gap_fill.run(args)
                fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
