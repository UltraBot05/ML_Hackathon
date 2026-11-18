"""
PathSnap Hackathon — Phase 2: Data Loader
Discovers, validates, and loads trip data from the nested dataset directory structure.
Supports both relative and absolute paths without hardcoding trip IDs.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataLoader")


@dataclass
class GPSObservation:
    index: int
    lon: float
    lat: float
    timestamp: float


@dataclass
class RoadNode:
    node_id: int
    lon: float
    lat: float


@dataclass
class RoadArc:
    arc_id: int
    from_node: int
    to_node: int


@dataclass
class TripData:
    trip_id: str
    partition: str
    track: List[GPSObservation]
    nodes: List[RoadNode]
    arcs: List[RoadArc]
    route: List[int]

    @property
    def num_points(self) -> int:
        return len(self.track)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)

    @property
    def num_route_arcs(self) -> int:
        return len(self.route)


class DataLoader:
    """Discovers and parses PathSnap trip data."""

    EXTENSIONS = [".track", ".nodes", ".arcs", ".route"]

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.dataset_root}")

    def discover_trip_dirs(self, partition: str = "train") -> List[Tuple[str, Path]]:
        """Discovers trip subdirectories in Dataset/{partition}/."""
        part_dir = self.dataset_root / partition
        if not part_dir.exists():
            logger.warning(f"Partition directory does not exist: {part_dir}")
            return []

        trip_dirs = []
        for p in sorted(part_dir.iterdir()):
            if p.is_dir():
                trip_dirs.append((p.name, p))
        return trip_dirs

    def validate_trip_files(self, trip_dir: Path, trip_id: str) -> Dict[str, bool]:
        """Verifies presence and readability of all four trip files."""
        status = {}
        for ext in self.EXTENSIONS:
            file_path = trip_dir / f"{trip_id}{ext}"
            exists = file_path.exists() and file_path.stat().st_size > 0
            status[ext] = exists
        return status

    def load_track(self, file_path: Path) -> List[GPSObservation]:
        observations = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        timestamp = float(parts[2])
                        observations.append(GPSObservation(index=idx, lon=lon, lat=lat, timestamp=timestamp))
                    except ValueError:
                        logger.warning(f"Malformed row in {file_path.name} at line {idx+1}")
        return observations

    def load_nodes(self, file_path: Path) -> List[RoadNode]:
        nodes = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        nodes.append(RoadNode(node_id=idx, lon=lon, lat=lat))
                    except ValueError:
                        logger.warning(f"Malformed node row in {file_path.name} at line {idx+1}")
        return nodes

    def load_arcs(self, file_path: Path) -> List[RoadArc]:
        arcs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        from_node = int(float(parts[0]))
                        to_node = int(float(parts[1]))
                        arcs.append(RoadArc(arc_id=idx, from_node=from_node, to_node=to_node))
                    except ValueError:
                        logger.warning(f"Malformed arc row in {file_path.name} at line {idx+1}")
        return arcs

    def load_route(self, file_path: Path) -> List[int]:
        route = []
        if not file_path.exists():
            return route
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if parts:
                    try:
                        arc_id = int(float(parts[0]))
                        route.append(arc_id)
                    except ValueError:
                        logger.warning(f"Malformed route row in {file_path.name} at line {idx+1}")
        return route

    def load_trip(self, trip_id: str, partition: str = "train") -> Optional[TripData]:
        """Loads a single trip by ID from the given partition."""
        trip_dir = self.dataset_root / partition / trip_id
        if not trip_dir.exists():
            logger.error(f"Trip directory does not exist: {trip_dir}")
            return None

        file_status = self.validate_trip_files(trip_dir, trip_id)
        if not all(file_status.values()):
            missing = [ext for ext, present in file_status.items() if not present]
            logger.error(f"Trip {trip_id} missing or empty files: {missing}")
            return None

        track = self.load_track(trip_dir / f"{trip_id}.track")
        nodes = self.load_nodes(trip_dir / f"{trip_id}.nodes")
        arcs = self.load_arcs(trip_dir / f"{trip_id}.arcs")
        route = self.load_route(trip_dir / f"{trip_id}.route")

        return TripData(trip_id=trip_id, partition=partition, track=track, nodes=nodes, arcs=arcs, route=route)

    def load_all_trips(self, partition: str = "train", max_trips: Optional[int] = None) -> List[TripData]:
        """Loads all trips in the partition up to optional limit."""
        trip_dirs = self.discover_trip_dirs(partition)
        if max_trips is not None:
            trip_dirs = trip_dirs[:max_trips]

        logger.info(f"Loading {len(trip_dirs)} trips from partition '{partition}'...")
        trips = []
        for trip_id, _ in trip_dirs:
            t = self.load_trip(trip_id, partition)
            if t is not None:
                trips.append(t)
        logger.info(f"Successfully loaded {len(trips)}/{len(trip_dirs)} trips.")
        return trips

