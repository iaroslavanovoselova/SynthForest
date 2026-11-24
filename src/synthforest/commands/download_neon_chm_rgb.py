#!/usr/bin/env python3
"""Download the paper's NEON CHMs and matching NEON RGB mosaic tiles.

The CHM inventory is the list published with Tolan et al. (2024):
https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/NEON_training_images.csv

Each CHM URL identifies a year/site/flight and 1 km UTM tile. Matching
DP3.30010.001 RGB mosaics are resolved by year, site, and coordinates. Camera
and LiDAR visit numbers sometimes differ, so the preflight tries alternate
camera visits while retaining the CHM acquisition in the paired output name.
Outputs use the naming convention of the existing NEON height benchmark:

    <year>_<site>_<flight>_NEON_<domain>_<site>_DP3_<x>_<y>_{RGB,CHM}.tif

The five paper test sites are written to test/. By default, six ecologically
diverse training sites (ABBY, BART, OSBS, PUUM, SERC, and SJER) yield about 700
matched training pairs; all other sites are excluded. RGB URLs are checked
before download. A selected CHM without a remotely available RGB counterpart
is recorded in metadata/unmatched_rgb.csv and its CHM is downloaded to
unmatched_rgb/. It is kept out of train/ and test/ because it cannot form a
supervised training pair.

Downloads are resumable: incomplete files remain as ``*.part`` and are resumed
with an HTTP Range request. Existing non-empty final files are skipped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from synthforest.paths import data_path

CHM_PRODUCT_ID = "DP3.30015.001"
RGB_PRODUCT_ID = "DP3.30010.001"
SOURCE_CSV_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/NEON_training_images.csv"
DEFAULT_OUTPUT_ROOT = data_path("NEON_CHM_1m_real_all")
DEFAULT_REFERENCE_ROOT = data_path("CHM_rasters_50cm/NEON_CHM_1m_real_eval_rasters")
TEST_SITES = frozenset({"CUPE", "REDB", "WLOU", "HOPB", "GUAN"})
DEFAULT_TRAIN_SITES = frozenset({"ABBY", "BART", "OSBS", "PUUM", "SERC", "SJER"})
USER_AGENT = "synthforest-neon-pair-downloader/1.0"

ACQUISITION_RE = re.compile(r"^(?P<year>\d{4})_(?P<site>[A-Z0-9]{4})_(?P<flight>\d+)$")
CHM_NAME_RE = re.compile(
    r"^NEON_(?P<domain>D\d{2})_(?P<site>[A-Z0-9]{4})_DP3_"
    r"(?P<easting>\d+)_(?P<northing>\d+)_CHM\.tif$"
)
OUTPUT_STEM_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<site>[A-Z0-9]{4})_(?P<flight>\d+)_"
    r"NEON_(?P<domain>D\d{2})_(?P<site_again>[A-Z0-9]{4})_DP3_"
    r"(?P<easting>\d+)_(?P<northing>\d+)$"
)
CONTENT_RANGE_RE = re.compile(r"^bytes (?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+)$")
UNSATISFIED_RANGE_RE = re.compile(r"^bytes \*/(?P<total>\d+)$")


@dataclass(frozen=True)
class NeonTile:
    year: str
    site: str
    flight: str
    domain: str
    easting: str
    northing: str
    chm_url: str
    rgb_url: str
    split: str
    rgb_flight: str
    rgb_bytes: int | None = None

    @property
    def acquisition(self) -> str:
        return f"{self.year}_{self.site}_{self.flight}"

    @property
    def stem(self) -> str:
        return (
            f"{self.acquisition}_NEON_{self.domain}_{self.site}_DP3_{self.easting}_{self.northing}"
        )

    @property
    def rgb_acquisition(self) -> str:
        return f"{self.year}_{self.site}_{self.rgb_flight}"

    def output_path(self, root: Path, product: str) -> Path:
        return root / self.split / f"{self.stem}_{product}.tif"


@dataclass(frozen=True)
class DownloadSpec:
    tile_stem: str
    product: str
    url: str
    destination: Path


@dataclass(frozen=True)
class DownloadResult:
    spec: DownloadSpec
    status: str
    size: int


@dataclass(frozen=True)
class RemoteCheck:
    status: str
    size: int | None
    detail: str = ""


def parse_chm_url(url: str, test_sites: frozenset[str] = TEST_SITES) -> NeonTile:
    """Parse one published CHM URL and derive its matching RGB URL and output name."""
    parsed = urlsplit(url.strip())
    if parsed.scheme != "https" or parsed.netloc != "storage.googleapis.com":
        raise ValueError(f"Unexpected CHM URL host: {url!r}")

    parts = PurePosixPath(unquote(parsed.path)).parts
    try:
        bucket_index = parts.index("neon-aop-products")
    except ValueError as exc:
        raise ValueError(f"CHM URL is not in the neon-aop-products bucket: {url!r}") from exc
    tail = parts[bucket_index + 1 :]
    if len(tail) != 8:
        raise ValueError(f"Unexpected CHM URL layout: {url!r}")

    year, full_site, domain, acquisition, level, sensor, folder, filename = tail
    if (
        full_site != "FullSite"
        or level != "L3"
        or sensor != "DiscreteLidar"
        or folder != "CanopyHeightModelGtif"
    ):
        raise ValueError(f"URL is not a {CHM_PRODUCT_ID} CHM GeoTIFF: {url!r}")

    acquisition_match = ACQUISITION_RE.fullmatch(acquisition)
    filename_match = CHM_NAME_RE.fullmatch(filename)
    if acquisition_match is None or filename_match is None:
        raise ValueError(f"Could not parse CHM acquisition or filename: {url!r}")

    acquisition_fields = acquisition_match.groupdict()
    filename_fields = filename_match.groupdict()
    site = acquisition_fields["site"]
    if (
        acquisition_fields["year"] != year
        or filename_fields["domain"] != domain
        or filename_fields["site"] != site
    ):
        raise ValueError(f"Inconsistent year/site/domain fields in CHM URL: {url!r}")

    easting = filename_fields["easting"]
    northing = filename_fields["northing"]
    base = url.rsplit("/L3/DiscreteLidar/CanopyHeightModelGtif/", 1)[0]
    rgb_name = f"{acquisition}_{easting}_{northing}_image.tif"
    rgb_url = f"{base}/L3/Camera/Mosaic/{rgb_name}"
    return NeonTile(
        year=year,
        site=site,
        flight=acquisition_fields["flight"],
        domain=domain,
        easting=easting,
        northing=northing,
        chm_url=url.strip(),
        rgb_url=rgb_url,
        split="test" if site in test_sites else "train",
        rgb_flight=acquisition_fields["flight"],
    )


def parse_tiles_csv(
    payload: bytes,
    *,
    test_sites: frozenset[str] = TEST_SITES,
) -> list[NeonTile]:
    """Read the published CSV, validating that it contains unique CHM tiles."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The source CSV is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "url" not in reader.fieldnames:
        raise ValueError("Source CSV must contain a 'url' column")

    tiles: list[NeonTile] = []
    seen_stems: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        url = (row.get("url") or "").strip()
        if not url:
            continue
        try:
            tile = parse_chm_url(url, test_sites)
        except ValueError as exc:
            raise ValueError(f"Invalid source CSV row {line_number}: {exc}") from exc
        if tile.stem in seen_stems:
            raise ValueError(f"Duplicate CHM tile in source CSV: {tile.stem}")
        seen_stems.add(tile.stem)
        tiles.append(tile)

    if not tiles:
        raise ValueError("Source CSV contains no CHM URLs")
    missing_test_sites = test_sites - {tile.site for tile in tiles}
    if missing_test_sites:
        raise ValueError(
            "Source CSV has no tiles for configured test site(s): "
            + ", ".join(sorted(missing_test_sites))
        )
    return sorted(tiles, key=lambda tile: tile.stem)


def select_site_split(
    tiles: Iterable[NeonTile],
    *,
    train_sites: frozenset[str],
    test_sites: frozenset[str],
) -> list[NeonTile]:
    overlap = train_sites & test_sites
    if overlap:
        raise ValueError("Train and test site sets overlap: " + ", ".join(sorted(overlap)))
    tile_list = list(tiles)
    available_sites = {tile.site for tile in tile_list}
    missing_train_sites = train_sites - available_sites
    if missing_train_sites:
        raise ValueError(
            "Source CSV has no tiles for configured train site(s): "
            + ", ".join(sorted(missing_train_sites))
        )
    selected_sites = train_sites | test_sites
    return [tile for tile in tile_list if tile.site in selected_sites]


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
) -> Request:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    return Request(url, headers=request_headers, method=method)


def _retry_delay(attempt: int) -> float:
    return min(30.0, 1.5 * (2**attempt))


def _is_retryable_http(code: int) -> bool:
    return code in {408, 425, 429} or 500 <= code <= 599


def read_source(source: str, *, timeout: float, retries: int) -> bytes:
    parsed = urlsplit(source)
    if parsed.scheme not in {"http", "https"}:
        return Path(source).expanduser().read_bytes()

    for attempt in range(retries + 1):
        try:
            with urlopen(_request(source), timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if not _is_retryable_http(exc.code) or attempt == retries:
                raise RuntimeError(f"Could not fetch source CSV ({exc.code}): {source}") from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt == retries:
                raise RuntimeError(f"Could not fetch source CSV: {source}: {exc}") from exc
        time.sleep(_retry_delay(attempt))
    raise AssertionError("unreachable")


def check_remote_url(url: str, *, timeout: float, retries: int) -> RemoteCheck:
    """Return whether a public object exists and its advertised byte size."""
    for attempt in range(retries + 1):
        try:
            with urlopen(_request(url, method="HEAD"), timeout=timeout) as response:
                size_header = response.headers.get("Content-Length")
                return RemoteCheck(
                    status="available",
                    size=int(size_header) if size_header is not None else None,
                )
        except HTTPError as exc:
            if exc.code in {404, 410}:
                return RemoteCheck(status="missing", size=None, detail=f"HTTP {exc.code}")
            if not _is_retryable_http(exc.code) or attempt == retries:
                return RemoteCheck(status="error", size=None, detail=f"HTTP {exc.code}")
        except (URLError, TimeoutError, socket.timeout) as exc:
            if attempt == retries:
                return RemoteCheck(status="error", size=None, detail=str(exc))
        time.sleep(_retry_delay(attempt))
    raise AssertionError("unreachable")


def rgb_url_for_flight(tile: NeonTile, flight: str) -> str:
    """Build the RGB object URL for a camera visit, retaining year/site/coordinates."""
    source_marker = f"/{tile.acquisition}/L3/DiscreteLidar/CanopyHeightModelGtif/"
    if source_marker not in tile.chm_url:
        raise ValueError(f"Could not locate CHM acquisition in URL: {tile.chm_url}")
    prefix = tile.chm_url.split(source_marker, 1)[0]
    rgb_acquisition = f"{tile.year}_{tile.site}_{flight}"
    rgb_name = f"{rgb_acquisition}_{tile.easting}_{tile.northing}_image.tif"
    return f"{prefix}/{rgb_acquisition}/L3/Camera/Mosaic/{rgb_name}"


def resolve_rgb_tile(
    tile: NeonTile,
    *,
    max_rgb_flight: int,
    timeout: float,
    retries: int,
) -> tuple[NeonTile | None, RemoteCheck]:
    """Resolve an RGB tile, allowing NEON's camera and LiDAR visit numbers to differ."""
    try:
        source_flight = int(tile.flight)
    except ValueError:
        source_flight = 0
    alternative_flights = sorted(
        (flight for flight in range(1, max_rgb_flight + 1) if flight != source_flight),
        key=lambda flight: (abs(flight - source_flight), flight),
    )
    flights = [tile.flight, *(str(flight) for flight in alternative_flights)]
    attempted: list[str] = []
    for flight in flights:
        rgb_url = rgb_url_for_flight(tile, flight)
        attempted.append(flight)
        check = check_remote_url(rgb_url, timeout=timeout, retries=retries)
        if check.status == "available":
            return (
                replace(
                    tile,
                    rgb_url=rgb_url,
                    rgb_flight=flight,
                    rgb_bytes=check.size,
                ),
                check,
            )
        if check.status == "error":
            return None, replace(check, detail=f"camera visit {flight}: {check.detail}")
    return None, RemoteCheck(
        status="missing",
        size=None,
        detail="HTTP 404/410 for camera visit(s) " + ",".join(attempted),
    )


def preflight_rgb_urls(
    tiles: Iterable[NeonTile],
    *,
    workers: int,
    max_rgb_flight: int,
    timeout: float,
    retries: int,
    progress_every: int,
) -> tuple[list[NeonTile], list[tuple[NeonTile, RemoteCheck]]]:
    """Check all derived RGB URLs, returning matched and unmatched tiles."""
    tile_list = list(tiles)
    resolutions: dict[str, tuple[NeonTile | None, RemoteCheck]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_tile = {
            executor.submit(
                resolve_rgb_tile,
                tile,
                max_rgb_flight=max_rgb_flight,
                timeout=timeout,
                retries=retries,
            ): tile
            for tile in tile_list
        }
        for completed, future in enumerate(as_completed(future_to_tile), start=1):
            tile = future_to_tile[future]
            resolutions[tile.stem] = future.result()
            if completed % progress_every == 0 or completed == len(tile_list):
                counts = Counter(resolution[1].status for resolution in resolutions.values())
                print(
                    f"RGB preflight {completed}/{len(tile_list)} "
                    f"(available={counts['available']}, missing={counts['missing']}, "
                    f"errors={counts['error']})",
                    flush=True,
                )

    errors = [
        (tile, resolutions[tile.stem][1])
        for tile in tile_list
        if resolutions[tile.stem][1].status == "error"
    ]
    if errors:
        examples = "; ".join(f"{tile.stem}: {check.detail}" for tile, check in errors[:5])
        raise RuntimeError(
            f"RGB preflight could not determine availability for {len(errors)} tile(s): {examples}"
        )

    matched = [
        resolved for tile in tile_list if (resolved := resolutions[tile.stem][0]) is not None
    ]
    unmatched = [
        (tile, resolutions[tile.stem][1])
        for tile in tile_list
        if resolutions[tile.stem][1].status == "missing"
    ]
    return matched, unmatched


def parse_output_stem(stem: str) -> dict[str, str]:
    match = OUTPUT_STEM_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unexpected NEON benchmark filename stem: {stem!r}")
    fields = match.groupdict()
    if fields["site"] != fields["site_again"]:
        raise ValueError(f"Inconsistent site fields in filename stem: {stem!r}")
    return fields


def audit_reference_dataset(
    reference_root: Path,
    planned_test_tiles: Iterable[NeonTile],
    *,
    test_sites: frozenset[str] = TEST_SITES,
) -> dict[str, object]:
    """Verify the existing benchmark/validation pairs against the planned test fold."""
    products_by_stem: dict[str, dict[str, str]] = defaultdict(dict)
    sites_by_fold: dict[str, Counter[str]] = {}
    for fold in ("benchmark", "validation"):
        fold_root = reference_root / fold
        if not fold_root.is_dir():
            raise FileNotFoundError(f"Reference fold not found: {fold_root}")
        fold_products: dict[str, set[str]] = defaultdict(set)
        for product in ("RGB", "CHM"):
            for path in sorted(fold_root.glob(f"*_{product}.tif")):
                stem = path.name[: -len(f"_{product}.tif")]
                fields = parse_output_stem(stem)
                if product in products_by_stem[stem]:
                    raise ValueError(f"Duplicate reference {product} raster for {stem}")
                products_by_stem[stem][product] = fold
                fold_products[stem].add(product)
                if fields["site"] not in test_sites:
                    raise ValueError(
                        f"Reference {fold} contains non-test site {fields['site']}: {path.name}"
                    )
        unpaired = sorted(stem for stem, products in fold_products.items() if len(products) != 2)
        if unpaired:
            raise ValueError(
                f"Reference {fold} contains {len(unpaired)} unpaired tile(s), "
                f"including {unpaired[0]}"
            )
        sites_by_fold[fold] = Counter(parse_output_stem(stem)["site"] for stem in fold_products)

    split_across_folds = sorted(
        stem for stem, products in products_by_stem.items() if len(set(products.values())) != 1
    )
    if split_across_folds:
        raise ValueError(
            "Reference RGB and CHM are in different folds for "
            f"{len(split_across_folds)} tile(s), including {split_across_folds[0]}"
        )

    planned_stems = {tile.stem for tile in planned_test_tiles}
    reference_stems = set(products_by_stem)
    missing = sorted(planned_stems - reference_stems)
    extra = sorted(reference_stems - planned_stems)
    report: dict[str, object] = {
        "reference_root": str(reference_root),
        "configured_test_sites": sorted(test_sites),
        "planned_test_pairs": len(planned_stems),
        "reference_pairs": len(reference_stems),
        "exact_tile_match": not missing and not extra,
        "missing_from_reference": missing,
        "extra_in_reference": extra,
        "folds": {
            fold: {
                "pairs": sum(counter.values()),
                "sites": dict(sorted(counter.items())),
                "site_set_matches_configured_test_sites": set(counter) == set(test_sites),
            }
            for fold, counter in sorted(sites_by_fold.items())
        },
    }
    if missing or extra:
        raise ValueError(
            "Existing benchmark/validation tiles do not equal the planned test fold: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return report


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"))


def write_manifest(path: Path, tiles: Iterable[NeonTile], output_root: Path) -> None:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "stem",
        "split",
        "site",
        "year",
        "flight",
        "rgb_flight",
        "domain",
        "easting",
        "northing",
        "chm_product_id",
        "rgb_product_id",
        "chm_url",
        "rgb_url",
        "chm_path",
        "rgb_path",
        "rgb_bytes",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for tile in tiles:
        writer.writerow(
            {
                "stem": tile.stem,
                "split": tile.split,
                "site": tile.site,
                "year": tile.year,
                "flight": tile.flight,
                "rgb_flight": tile.rgb_flight,
                "domain": tile.domain,
                "easting": tile.easting,
                "northing": tile.northing,
                "chm_product_id": CHM_PRODUCT_ID,
                "rgb_product_id": RGB_PRODUCT_ID,
                "chm_url": tile.chm_url,
                "rgb_url": tile.rgb_url,
                "chm_path": str(tile.output_path(output_root, "CHM").relative_to(output_root)),
                "rgb_path": str(tile.output_path(output_root, "RGB").relative_to(output_root)),
                "rgb_bytes": "" if tile.rgb_bytes is None else tile.rgb_bytes,
            }
        )
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def write_unmatched_manifest(
    path: Path,
    unmatched: Iterable[tuple[NeonTile, RemoteCheck]],
    output_root: Path,
) -> None:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "stem",
        "site",
        "intended_split",
        "chm_url",
        "expected_rgb_url",
        "chm_path",
        "reason",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for tile, check in unmatched:
        writer.writerow(
            {
                "stem": tile.stem,
                "site": tile.site,
                "intended_split": tile.split,
                "chm_url": tile.chm_url,
                "expected_rgb_url": tile.rgb_url,
                "chm_path": str(unmatched_chm_path(tile, output_root).relative_to(output_root)),
                "reason": check.detail,
            }
        )
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def split_summary(
    matched_tiles: Iterable[NeonTile],
    unmatched: Iterable[tuple[NeonTile, RemoteCheck]],
    *,
    train_sites: frozenset[str] = DEFAULT_TRAIN_SITES,
    test_sites: frozenset[str] = TEST_SITES,
) -> dict[str, object]:
    matched_list = list(matched_tiles)
    unmatched_list = list(unmatched)
    pair_counts = Counter(tile.split for tile in matched_list)
    site_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for tile in matched_list:
        site_counts[tile.split][tile.site] += 1
    return {
        "chm_product_id": CHM_PRODUCT_ID,
        "rgb_product_id": RGB_PRODUCT_ID,
        "configured_train_sites": sorted(train_sites),
        "configured_test_sites": sorted(test_sites),
        "matched_pairs": len(matched_list),
        "unmatched_rgb_tiles": len(unmatched_list),
        "pairs_by_split": dict(sorted(pair_counts.items())),
        "pairs_by_split_and_site": {
            split: dict(sorted(counts.items())) for split, counts in sorted(site_counts.items())
        },
        "advertised_rgb_bytes": sum(tile.rgb_bytes or 0 for tile in matched_list),
        "pairs_using_different_rgb_flight": sum(
            tile.flight != tile.rgb_flight for tile in matched_list
        ),
    }


def _total_from_content_range(value: str | None) -> int | None:
    if not value:
        return None
    match = CONTENT_RANGE_RE.fullmatch(value.strip())
    return int(match.group("total")) if match else None


def _complete_satisfied_part(destination: Path, part_path: Path, exc: HTTPError) -> bool:
    if exc.code != 416 or not part_path.is_file():
        return False
    match = UNSATISFIED_RANGE_RE.fullmatch((exc.headers.get("Content-Range") or "").strip())
    if match is None or part_path.stat().st_size != int(match.group("total")):
        return False
    os.replace(part_path, destination)
    return True


def download_file(
    spec: DownloadSpec,
    *,
    overwrite: bool,
    timeout: float,
    retries: int,
) -> DownloadResult:
    destination = spec.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return DownloadResult(spec=spec, status="skipped", size=destination.stat().st_size)

    part_path = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        resume_at = 0 if overwrite else (part_path.stat().st_size if part_path.is_file() else 0)
        headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
        try:
            with urlopen(_request(spec.url, headers=headers), timeout=timeout) as response:
                status = response.getcode()
                append = resume_at > 0 and status == 206
                mode = "ab" if append else "wb"
                expected_total = _total_from_content_range(response.headers.get("Content-Range"))
                if expected_total is None:
                    length = response.headers.get("Content-Length")
                    expected_total = (
                        (resume_at if append else 0) + int(length) if length is not None else None
                    )
                with part_path.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)

            final_size = part_path.stat().st_size
            if final_size <= 0:
                raise OSError("download produced an empty file")
            if expected_total is not None and final_size != expected_total:
                raise OSError(f"incomplete download: got {final_size} of {expected_total} bytes")
            os.replace(part_path, destination)
            return DownloadResult(spec=spec, status="downloaded", size=final_size)
        except HTTPError as exc:
            if _complete_satisfied_part(destination, part_path, exc):
                return DownloadResult(
                    spec=spec,
                    status="downloaded",
                    size=destination.stat().st_size,
                )
            if not _is_retryable_http(exc.code) or attempt == retries:
                raise RuntimeError(f"HTTP {exc.code} for {spec.url}") from exc
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Download failed for {spec.url}: {exc}") from exc
        time.sleep(_retry_delay(attempt))
    raise AssertionError("unreachable")


def unmatched_chm_path(tile: NeonTile, output_root: Path) -> Path:
    return output_root / "unmatched_rgb" / f"{tile.stem}_CHM.tif"


def build_download_specs(
    tiles: Iterable[NeonTile],
    output_root: Path,
    unmatched: Iterable[tuple[NeonTile, RemoteCheck]] = (),
) -> list[DownloadSpec]:
    specs: list[DownloadSpec] = []
    for tile in tiles:
        specs.extend(
            [
                DownloadSpec(
                    tile_stem=tile.stem,
                    product="CHM",
                    url=tile.chm_url,
                    destination=tile.output_path(output_root, "CHM"),
                ),
                DownloadSpec(
                    tile_stem=tile.stem,
                    product="RGB",
                    url=tile.rgb_url,
                    destination=tile.output_path(output_root, "RGB"),
                ),
            ]
        )
    for tile, _check in unmatched:
        specs.append(
            DownloadSpec(
                tile_stem=tile.stem,
                product="CHM_UNMATCHED_RGB",
                url=tile.chm_url,
                destination=unmatched_chm_path(tile, output_root),
            )
        )
    return specs


def run_downloads(
    specs: Iterable[DownloadSpec],
    *,
    workers: int,
    overwrite: bool,
    timeout: float,
    retries: int,
    progress_every: int,
) -> tuple[list[DownloadResult], list[tuple[DownloadSpec, str]]]:
    spec_list = list(specs)
    results: list[DownloadResult] = []
    failures: list[tuple[DownloadSpec, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_spec: dict[Future[DownloadResult], DownloadSpec] = {
            executor.submit(
                download_file,
                spec,
                overwrite=overwrite,
                timeout=timeout,
                retries=retries,
            ): spec
            for spec in spec_list
        }
        for completed, future in enumerate(as_completed(future_to_spec), start=1):
            spec = future_to_spec[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((spec, str(exc)))
            if completed % progress_every == 0 or completed == len(spec_list):
                statuses = Counter(result.status for result in results)
                print(
                    f"Downloads {completed}/{len(spec_list)} "
                    f"(downloaded={statuses['downloaded']}, "
                    f"skipped={statuses['skipped']}, failed={len(failures)})",
                    flush=True,
                )
    return results, failures


def _parse_sites(value: str) -> frozenset[str]:
    sites = frozenset(site.strip().upper() for site in value.split(",") if site.strip())
    invalid = sorted(site for site in sites if re.fullmatch(r"[A-Z0-9]{4}", site) is None)
    if not sites or invalid:
        raise argparse.ArgumentTypeError(
            "sites must be a comma-separated list of four-character NEON site codes"
        )
    return sites


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the published NEON CHM inventory and matching RGB mosaic tiles."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-csv",
        default=SOURCE_CSV_URL,
        help="Published CSV URL or a local copy of the CSV.",
    )
    parser.add_argument(
        "--test-sites",
        type=_parse_sites,
        default=TEST_SITES,
        help="Comma-separated site-level test split (default: CUPE,REDB,WLOU,HOPB,GUAN).",
    )
    parser.add_argument(
        "--train-sites",
        type=_parse_sites,
        default=DEFAULT_TRAIN_SITES,
        help=(
            "Comma-separated training sites "
            "(default: ABBY,BART,OSBS,PUUM,SERC,SJER; about 700 matched RGBs)."
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Existing root containing benchmark/ and validation/ for the test-tile audit.",
    )
    parser.add_argument("--skip-reference-audit", action="store_true")
    parser.add_argument(
        "--preflight-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="HEAD-check derived RGB URLs and exclude explicitly missing pairs (default: true).",
    )
    parser.add_argument(
        "--strict-rgb-match",
        action="store_true",
        help="Fail instead of recording and excluding CHMs with no matching RGB object.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-rgb-flight",
        type=int,
        default=10,
        help=(
            "Try camera visit numbers through this value when NEON's camera and LiDAR "
            "visit numbers differ (default: 10)."
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write metadata and manifests after preflight, but do not download rasters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and audit the split without writing metadata or downloading rasters.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_rgb_flight < 0:
        parser.error("--max-rgb-flight cannot be negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def _print_summary(summary: Mapping[str, object], reference_report: object | None) -> None:
    print(f"Matched CHM/RGB pairs: {summary['matched_pairs']}")
    print(f"Unmatched RGB tiles: {summary['unmatched_rgb_tiles']}")
    print(f"Pairs by split: {summary['pairs_by_split']}")
    advertised_bytes = int(summary["advertised_rgb_bytes"])
    if advertised_bytes:
        print(f"Advertised RGB size: {advertised_bytes / (1024**3):.1f} GiB")
    if isinstance(reference_report, dict):
        print(
            "Reference audit: "
            f"{reference_report['reference_pairs']} pairs, "
            f"exact_tile_match={reference_report['exact_tile_match']}"
        )


def main() -> None:
    args = _parse_args()
    source_payload = read_source(args.source_csv, timeout=args.timeout, retries=args.retries)
    source_tiles = parse_tiles_csv(source_payload, test_sites=args.test_sites)
    selected_source_tiles = select_site_split(
        source_tiles,
        train_sites=args.train_sites,
        test_sites=args.test_sites,
    )

    unmatched: list[tuple[NeonTile, RemoteCheck]] = []
    if args.preflight_rgb:
        matched_tiles, unmatched = preflight_rgb_urls(
            selected_source_tiles,
            workers=args.workers,
            max_rgb_flight=args.max_rgb_flight,
            timeout=args.timeout,
            retries=args.retries,
            progress_every=args.progress_every,
        )
    else:
        matched_tiles = selected_source_tiles
    if unmatched and args.strict_rgb_match:
        raise SystemExit(
            f"{len(unmatched)} CHM tile(s) have no matching RGB object; "
            "see the preflight output or omit --strict-rgb-match to exclude them."
        )

    reference_report: dict[str, object] | None = None
    if not args.skip_reference_audit:
        reference_report = audit_reference_dataset(
            args.reference_root,
            (tile for tile in matched_tiles if tile.split == "test"),
            test_sites=args.test_sites,
        )

    summary = split_summary(
        matched_tiles,
        unmatched,
        train_sites=args.train_sites,
        test_sites=args.test_sites,
    )
    summary["source_chm_tiles"] = len(source_tiles)
    summary["selected_source_chm_tiles"] = len(selected_source_tiles)
    summary["excluded_source_chm_tiles"] = len(source_tiles) - len(selected_source_tiles)
    _print_summary(summary, reference_report)
    if args.dry_run:
        print("Dry run: no files were written.")
        return

    metadata_root = args.output_root / "metadata"
    _atomic_write_bytes(metadata_root / "NEON_training_images.csv", source_payload)
    write_manifest(metadata_root / "download_manifest.csv", matched_tiles, args.output_root)
    write_unmatched_manifest(
        metadata_root / "unmatched_rgb.csv",
        unmatched,
        args.output_root,
    )
    _atomic_write_json(metadata_root / "split_summary.json", summary)
    if reference_report is not None:
        _atomic_write_json(metadata_root / "reference_test_audit.json", reference_report)
    if args.manifest_only:
        print(f"Manifest-only run complete: {metadata_root}")
        return

    specs = build_download_specs(matched_tiles, args.output_root, unmatched)
    results, failures = run_downloads(
        specs,
        workers=args.workers,
        overwrite=args.overwrite,
        timeout=args.timeout,
        retries=args.retries,
        progress_every=args.progress_every,
    )
    status_counts = Counter(result.status for result in results)
    download_report = {
        "files_expected": len(specs),
        "files_downloaded": status_counts["downloaded"],
        "files_skipped": status_counts["skipped"],
        "files_failed": len(failures),
        "bytes_in_successful_files": sum(result.size for result in results),
        "failures": [
            {
                "stem": spec.tile_stem,
                "product": spec.product,
                "url": spec.url,
                "destination": str(spec.destination),
                "error": error,
            }
            for spec, error in failures
        ],
    }
    _atomic_write_json(metadata_root / "download_report.json", download_report)
    if failures:
        print(
            f"Download finished with {len(failures)} failure(s); see "
            f"{metadata_root / 'download_report.json'}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Download complete: {args.output_root}")


if __name__ == "__main__":
    main()
