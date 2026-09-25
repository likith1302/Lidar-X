"""Multi-format LiDAR point cloud parser and validator.

Supports:
- .bin: SemanticKITTI float32 [x, y, z, intensity] binary (16 bytes/point)
- .pcd: Point Cloud Library PCD format (ASCII and binary data modes)
- .xyz: ASCII coordinate rows [x, y, z, (intensity)]
- .ply: Polygon File Format (ASCII and binary little-endian vertex clouds)
"""

import hashlib
import io
import re
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np

from ..models.schemas import Point3D, BoundingBox3D, JsonPointCloudPayload


class LidarParserError(Exception):
    """Exception raised when parsing LiDAR input fails."""
    pass


class LidarParser:
    """Parser and validator for LiDAR point clouds across standard autonomous driving formats."""

    BYTES_PER_POINT = 16  # 4 x 4-byte float32 values (x, y, z, intensity)

    @classmethod
    def compute_content_hash(cls, raw_bytes: bytes) -> str:
        """Compute SHA-256 hash of raw bytes for traceability and cache-free tracking."""
        return hashlib.sha256(raw_bytes).hexdigest()

    @classmethod
    def parse_kitti_bin(cls, raw_bytes: bytes) -> np.ndarray:
        """Parse raw binary data from SemanticKITTI .bin file into float32 array (N, 4).
        
        Args:
            raw_bytes: Raw binary bytes of the .bin file.
            
        Returns:
            np.ndarray of shape (N, 4) with columns [x, y, z, intensity].
            
        Raises:
            LidarParserError: If byte length is invalid or empty.
        """
        byte_len = len(raw_bytes)
        if byte_len == 0:
            raise LidarParserError("Empty LiDAR file received: 0 bytes.")

        if byte_len % cls.BYTES_PER_POINT != 0:
            remainder = byte_len % cls.BYTES_PER_POINT
            raise LidarParserError(
                f"Malformed SemanticKITTI .bin file: total size {byte_len} bytes is not a "
                f"multiple of {cls.BYTES_PER_POINT} bytes (has {remainder} leftover bytes). "
                f"Expected 4 x 32-bit floats [x, y, z, intensity] per point."
            )

        try:
            points = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 4)
            return np.copy(points)
        except Exception as e:
            raise LidarParserError(f"Failed to decode binary buffer into float32 array: {str(e)}")

    @classmethod
    def parse_xyz(cls, raw_bytes: bytes) -> np.ndarray:
        """Parse ASCII XYZ / XYZI point cloud text file."""
        if not raw_bytes:
            raise LidarParserError("Empty XYZ file received: 0 bytes.")

        try:
            text = raw_bytes.decode("utf-8", errors="replace").strip()
            lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(("#", "//"))]
            if not lines:
                raise LidarParserError("XYZ file contains no numeric point data.")

            points_list = []
            for line in lines:
                tokens = re.split(r'[,\s]+', line)
                if len(tokens) >= 3:
                    try:
                        x = float(tokens[0])
                        y = float(tokens[1])
                        z = float(tokens[2])
                        i = float(tokens[3]) if len(tokens) > 3 else 0.0
                        points_list.append([x, y, z, i])
                    except ValueError:
                        continue

            if not points_list:
                raise LidarParserError("Failed to parse valid 3D coordinates from XYZ file.")

            return np.array(points_list, dtype=np.float32)
        except Exception as e:
            if isinstance(e, LidarParserError):
                raise e
            raise LidarParserError(f"Failed to parse XYZ format: {str(e)}")

    @classmethod
    def parse_pcd(cls, raw_bytes: bytes) -> np.ndarray:
        """Parse Point Cloud Library (.pcd) format in ASCII or Binary."""
        if not raw_bytes:
            raise LidarParserError("Empty PCD file received: 0 bytes.")

        try:
            # Find the header end
            header_end = raw_bytes.find(b"\nDATA ")
            if header_end == -1:
                header_end = raw_bytes.find(b"\r\nDATA ")
            if header_end == -1:
                raise LidarParserError("Invalid PCD file: 'DATA' line not found in header.")

            # Read full DATA line
            data_line_end = raw_bytes.find(b"\n", header_end + 6)
            if data_line_end == -1:
                data_line_end = len(raw_bytes)

            header_bytes = raw_bytes[:data_line_end].decode("ascii", errors="replace")
            header_lines = [l.strip() for l in header_bytes.splitlines() if l.strip()]

            fields = ["x", "y", "z"]
            type_map = {}
            count_map = {}
            data_mode = "ascii"
            num_points = 0

            for line in header_lines:
                parts = line.split()
                if not parts:
                    continue
                tag = parts[0].upper()
                if tag == "FIELDS":
                    fields = [p.lower() for p in parts[1:]]
                elif tag == "DATA":
                    data_mode = parts[1].lower()
                elif tag == "POINTS":
                    num_points = int(parts[1])

            body_bytes = raw_bytes[data_line_end + 1:]

            if "x" not in fields or "y" not in fields or "z" not in fields:
                raise LidarParserError("PCD header must specify at least 'x', 'y', and 'z' fields.")

            x_idx = fields.index("x")
            y_idx = fields.index("y")
            z_idx = fields.index("z")
            i_idx = fields.index("intensity") if "intensity" in fields else (fields.index("i") if "i" in fields else -1)

            if data_mode == "ascii":
                text = body_bytes.decode("utf-8", errors="replace")
                pts_list = []
                for line in text.splitlines():
                    tokens = line.strip().split()
                    if len(tokens) >= len(fields):
                        try:
                            x = float(tokens[x_idx])
                            y = float(tokens[y_idx])
                            z = float(tokens[z_idx])
                            i = float(tokens[i_idx]) if i_idx >= 0 else 0.0
                            pts_list.append([x, y, z, i])
                        except ValueError:
                            continue
                if not pts_list:
                    raise LidarParserError("No valid points decoded from ASCII PCD body.")
                return np.array(pts_list, dtype=np.float32)

            elif data_mode == "binary":
                # Assuming standard float32 for each field (most common LiDAR PCD)
                num_fields = len(fields)
                pts_per_elem = 4 * num_fields
                if len(body_bytes) < pts_per_elem:
                    raise LidarParserError(f"PCD binary payload too short ({len(body_bytes)} bytes).")

                arr = np.frombuffer(body_bytes, dtype=np.float32)
                usable = (len(arr) // num_fields) * num_fields
                arr = arr[:usable].reshape(-1, num_fields)

                x = arr[:, x_idx]
                y = arr[:, y_idx]
                z = arr[:, z_idx]
                i = arr[:, i_idx] if i_idx >= 0 else np.zeros(len(arr), dtype=np.float32)
                return np.column_stack([x, y, z, i]).astype(np.float32)
            else:
                raise LidarParserError(f"Unsupported PCD data mode: {data_mode}")

        except Exception as e:
            if isinstance(e, LidarParserError):
                raise e
            raise LidarParserError(f"Failed to parse PCD file: {str(e)}")

    @classmethod
    def parse_ply(cls, raw_bytes: bytes) -> np.ndarray:
        """Parse Stanford PLY format (ASCII or binary_little_endian) point clouds."""
        if not raw_bytes:
            raise LidarParserError("Empty PLY file received: 0 bytes.")

        try:
            end_header = raw_bytes.find(b"end_header")
            if end_header == -1:
                raise LidarParserError("Invalid PLY file: 'end_header' token not found.")

            # Advance past end_header line
            body_start = raw_bytes.find(b"\n", end_header)
            if body_start == -1:
                body_start = end_header + 10
            else:
                body_start += 1

            header_str = raw_bytes[:end_header].decode("ascii", errors="replace")
            header_lines = [l.strip() for l in header_str.splitlines() if l.strip()]

            is_binary = False
            vertex_count = 0
            properties = []

            for line in header_lines:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "format":
                    if "binary_little_endian" in parts[1]:
                        is_binary = True
                    elif "ascii" not in parts[1]:
                        raise LidarParserError(f"Unsupported PLY format: {parts[1]}")
                elif parts[0] == "element" and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                elif parts[0] == "property":
                    prop_type = parts[1]
                    prop_name = parts[2].lower()
                    properties.append((prop_name, prop_type))

            prop_names = [p[0] for p in properties]
            if "x" not in prop_names or "y" not in prop_names or "z" not in prop_names:
                raise LidarParserError("PLY vertex element missing required 'x', 'y', 'z' properties.")

            x_idx = prop_names.index("x")
            y_idx = prop_names.index("y")
            z_idx = prop_names.index("z")
            i_idx = -1
            for cand in ("intensity", "scalar", "confidence", "value", "i"):
                if cand in prop_names:
                    i_idx = prop_names.index(cand)
                    break

            body = raw_bytes[body_start:]

            if is_binary:
                # Type mapping for binary little endian
                dtype_list = []
                type_lookup = {
                    "float": "<f4", "float32": "<f4",
                    "double": "<f8", "float64": "<f8",
                    "int": "<i4", "int32": "<i4",
                    "uint": "<u4", "uint32": "<u4",
                    "short": "<i2", "int16": "<i2",
                    "ushort": "<u2", "uint16": "<u2",
                    "uchar": "<u1", "uint8": "<u1",
                    "char": "<i1", "int8": "<i1",
                }
                for name, p_type in properties:
                    np_t = type_lookup.get(p_type, "<f4")
                    dtype_list.append((name, np_t))

                struct_dt = np.dtype(dtype_list)
                element_size = struct_dt.itemsize
                total_elems = len(body) // element_size
                if total_elems == 0:
                    raise LidarParserError("PLY binary data contains 0 vertex records.")

                parsed = np.frombuffer(body[:total_elems * element_size], dtype=struct_dt)
                x = parsed["x"].astype(np.float32)
                y = parsed["y"].astype(np.float32)
                z = parsed["z"].astype(np.float32)
                if i_idx >= 0:
                    i = parsed[prop_names[i_idx]].astype(np.float32)
                else:
                    i = np.zeros(len(x), dtype=np.float32)
                return np.column_stack([x, y, z, i]).astype(np.float32)

            else:
                # ASCII mode
                text = body.decode("utf-8", errors="replace")
                pts_list = []
                for line in text.splitlines():
                    toks = line.strip().split()
                    if len(toks) >= len(properties):
                        try:
                            x = float(toks[x_idx])
                            y = float(toks[y_idx])
                            z = float(toks[z_idx])
                            i = float(toks[i_idx]) if i_idx >= 0 else 0.0
                            pts_list.append([x, y, z, i])
                        except ValueError:
                            continue
                if not pts_list:
                    raise LidarParserError("No valid points decoded from ASCII PLY body.")
                return np.array(pts_list, dtype=np.float32)

        except Exception as e:
            if isinstance(e, LidarParserError):
                raise e
            raise LidarParserError(f"Failed to parse PLY file: {str(e)}")

    @classmethod
    def parse_point_cloud(cls, raw_bytes: bytes, filename: str = "") -> np.ndarray:
        """Universal parser dispatching by file extension or content magic bytes.

        Supports .bin, .pcd, .xyz, .ply.
        """
        if not raw_bytes or len(raw_bytes) == 0:
            raise LidarParserError("Uploaded file is empty (0 bytes).")

        ext = Path(filename).suffix.lower() if filename else ""

        # Check explicit extensions
        if ext == ".xyz" or ext == ".txt":
            return cls.parse_xyz(raw_bytes)
        elif ext == ".pcd":
            return cls.parse_pcd(raw_bytes)
        elif ext == ".ply":
            return cls.parse_ply(raw_bytes)
        elif ext == ".bin":
            return cls.parse_kitti_bin(raw_bytes)

        # Sniff by magic header
        lead = raw_bytes[:100].strip()
        if lead.startswith(b"ply\n") or lead.startswith(b"ply\r\n"):
            return cls.parse_ply(raw_bytes)
        if b"# .PCD" in lead or lead.startswith(b"VERSION "):
            return cls.parse_pcd(raw_bytes)

        # Try SemanticKITTI binary as default binary format if divisible by 16
        if len(raw_bytes) % cls.BYTES_PER_POINT == 0 and len(raw_bytes) >= 16:
            try:
                return cls.parse_kitti_bin(raw_bytes)
            except Exception:
                pass

        # Try ASCII XYZ fallback
        try:
            return cls.parse_xyz(raw_bytes)
        except Exception:
            pass

        raise LidarParserError(
            f"Unsupported LiDAR file format '{filename or 'unknown'}'. "
            "Supported formats: .bin (SemanticKITTI), .pcd (PCL), .xyz (ASCII), .ply (Stanford 3D)."
        )

    @classmethod
    def parse_json_points(cls, payload: JsonPointCloudPayload) -> np.ndarray:
        """Parse JSON point cloud list into float32 array (N, 4)."""
        if not payload.points:
            raise LidarParserError("Empty point array in JSON payload.")

        n_pts = len(payload.points)
        arr = np.zeros((n_pts, 4), dtype=np.float32)
        for i, p in enumerate(payload.points):
            arr[i, 0] = p.x
            arr[i, 1] = p.y
            arr[i, 2] = p.z
            arr[i, 3] = p.intensity if p.intensity is not None else 0.0

        return arr

    @classmethod
    def compute_bounds(cls, points: np.ndarray) -> BoundingBox3D:
        """Compute axis-aligned 3D bounding box for points array (N, >=3)."""
        if points.shape[0] == 0:
            return BoundingBox3D(
                min_x=0.0, max_x=0.0,
                min_y=0.0, max_y=0.0,
                min_z=0.0, max_z=0.0,
            )

        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        if not np.any(valid_mask):
            return BoundingBox3D(
                min_x=0.0, max_x=0.0,
                min_y=0.0, max_y=0.0,
                min_z=0.0, max_z=0.0,
            )

        valid_pts = points[valid_mask, :3]
        min_vals = np.min(valid_pts, axis=0)
        max_vals = np.max(valid_pts, axis=0)

        return BoundingBox3D(
            min_x=float(min_vals[0]),
            max_x=float(max_vals[0]),
            min_y=float(min_vals[1]),
            max_y=float(max_vals[1]),
            min_z=float(min_vals[2]),
            max_z=float(max_vals[2]),
        )

    @classmethod
    def downsample_for_preview(cls, points: np.ndarray, max_points: int = 5000) -> List[Point3D]:
        """Uniformly downsample points for lightweight frontend JSON transfer."""
        n_pts = points.shape[0]
        if n_pts == 0:
            return []

        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        pts = points[valid_mask]
        n_valid = pts.shape[0]

        if n_valid <= max_points:
            selected_indices = np.arange(n_valid)
        else:
            step = n_valid / max_points
            selected_indices = (np.arange(max_points) * step).astype(np.int64)

        result: List[Point3D] = []
        for idx in selected_indices:
            result.append(
                Point3D.model_construct(
                    x=float(pts[idx, 0]),
                    y=float(pts[idx, 1]),
                    z=float(pts[idx, 2]),
                    intensity=float(pts[idx, 3]) if pts.shape[1] > 3 else 0.0,
                )
            )
        return result


def serialize_point_cloud(points: np.ndarray, fmt: str = "bin", binary: bool = True) -> bytes:
    """Serialize (N, 3) or (N, 4) numpy point cloud into bytes of specified format.
    
    Supported formats: 'bin', 'pcd', 'xyz', 'ply'.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("Points must have shape (N, 3) or (N, 4)")
    if pts.shape[1] == 3:
        intensity = np.zeros((pts.shape[0], 1), dtype=np.float32)
        pts = np.hstack([pts, intensity])

    fmt = fmt.lower().lstrip(".")
    if fmt == "bin":
        return pts.astype(np.float32).tobytes()
    elif fmt == "pcd":
        if binary:
            header = (
                "# .PCD v0.7 - Point Cloud Data\n"
                "VERSION 0.7\n"
                "FIELDS x y z intensity\n"
                "SIZE 4 4 4 4\n"
                "TYPE F F F F\n"
                "COUNT 1 1 1 1\n"
                f"WIDTH {len(pts)}\n"
                "HEIGHT 1\n"
                "VIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(pts)}\n"
                "DATA binary\n"
            ).encode("ascii")
            return header + pts.astype(np.float32).tobytes()
        else:
            lines = [
                "# .PCD v0.7 - Point Cloud Data",
                "VERSION 0.7",
                "FIELDS x y z intensity",
                "SIZE 4 4 4 4",
                "TYPE F F F F",
                "COUNT 1 1 1 1",
                f"WIDTH {len(pts)}",
                "HEIGHT 1",
                "VIEWPOINT 0 0 0 1 0 0 0",
                f"POINTS {len(pts)}",
                "DATA ascii",
            ]
            for p in pts:
                lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]:.4f}")
            return "\n".join(lines).encode("ascii")
    elif fmt == "xyz" or fmt == "txt":
        lines = [f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]:.4f}" for p in pts]
        return "\n".join(lines).encode("ascii")
    elif fmt == "ply":
        if binary:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float intensity\n"
                "end_header\n"
            ).encode("ascii")
            return header + pts.astype(np.float32).tobytes()
        else:
            lines = [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(pts)}",
                "property float x",
                "property float y",
                "property float z",
                "property float intensity",
                "end_header",
            ]
            for p in pts:
                lines.append(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]:.4f}")
            return "\n".join(lines).encode("ascii")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

