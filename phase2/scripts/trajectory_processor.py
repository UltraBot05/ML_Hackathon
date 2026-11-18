"""
PathSnap Hackathon — Phase 2: Trajectory Processor
Extracts kinematic, temporal, and spatial features for GPS trajectories.
Flags anomalies non-destructively while preserving raw observations.
"""

import math
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import logging

from data_loader import GPSObservation, TripData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrajectoryProcessor")

EARTH_RADIUS = 6_371_000.0  # meters


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two angular coordinates in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS * math.asin(math.sqrt(min(1.0, a)))


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Computes initial compass bearing from point 1 to point 2 in degrees [0, 360).
    0 = North, 90 = East, 180 = South, 270 = West.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


@dataclass
class ProcessedObservation:
    trip_id: str
    obs_index: int
    lon: float
    lat: float
    timestamp: float
    dt: float
    step_distance_m: float
    speed_mps: float
    speed_kmh: float
    acceleration_mps2: float
    bearing_deg: float
    cumulative_distance_m: float
    is_speed_spike: bool
    is_long_gap: bool
    is_stationary: bool
    is_session_break: bool


class TrajectoryProcessor:
    """Processes raw GPS observation streams into enriched kinematic features."""

    def __init__(
        self,
        max_speed_threshold_mps: float = 55.0,
        gap_threshold_seconds: float = 300.0,
        session_split_gap_seconds: float = 900.0,
    ):
        self.max_speed_threshold_mps = max_speed_threshold_mps
        self.gap_threshold_seconds = gap_threshold_seconds
        self.session_split_gap_seconds = session_split_gap_seconds

    def process_trip_track(self, trip_id: str, track: List[GPSObservation]) -> List[ProcessedObservation]:
        """Enriches raw GPS points with step distance, velocity, acceleration, bearing, and anomaly flags."""
        if not track:
            return []

        processed = []
        cum_dist = 0.0
        prev_speed = 0.0

        for i in range(len(track)):
            curr = track[i]

            if i == 0:
                dt = 0.0
                step_dist = 0.0
                speed = 0.0
                accel = 0.0
                bearing = 0.0
            else:
                prev = track[i - 1]
                dt = curr.timestamp - prev.timestamp
                step_dist = haversine_distance(prev.lon, prev.lat, curr.lon, curr.lat)
                speed = (step_dist / dt) if dt > 0 else 0.0
                accel = ((speed - prev_speed) / dt) if dt > 0 else 0.0

                if step_dist > 1.0:
                    bearing = calculate_bearing(prev.lon, prev.lat, curr.lon, curr.lat)
                else:
                    bearing = processed[-1].bearing_deg if processed else 0.0

            cum_dist += step_dist
            prev_speed = speed

            is_speed_spike = speed > self.max_speed_threshold_mps
            is_long_gap = dt > self.gap_threshold_seconds
            is_stationary = (step_dist < 2.0 and dt >= 1.0) if i > 0 else True
            is_session_break = dt > self.session_split_gap_seconds

            obs = ProcessedObservation(
                trip_id=trip_id,
                obs_index=curr.index,
                lon=curr.lon,
                lat=curr.lat,
                timestamp=curr.timestamp,
                dt=round(dt, 2),
                step_distance_m=round(step_dist, 3),
                speed_mps=round(speed, 3),
                speed_kmh=round(speed * 3.6, 2),
                acceleration_mps2=round(accel, 3),
                bearing_deg=round(bearing, 2),
                cumulative_distance_m=round(cum_dist, 2),
                is_speed_spike=is_speed_spike,
                is_long_gap=is_long_gap,
                is_stationary=is_stationary,
                is_session_break=is_session_break,
            )
            processed.append(obs)

        return processed

    def to_dataframe(self, processed_observations: List[ProcessedObservation]) -> pd.DataFrame:
        """Converts observation objects to a pandas DataFrame."""
        return pd.DataFrame([asdict(o) for o in processed_observations])

