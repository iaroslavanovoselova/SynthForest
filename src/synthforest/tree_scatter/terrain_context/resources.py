from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import bpy
from omegaconf import OmegaConf

from synthforest.paths import external_root, project_root

from ..config import TerrainConfig
from ..scene import (
    make_material,
    make_material_from_image,
    make_path_blend_material_from_image,
    make_water_material_from_image,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}
PROJECT_ROOT = project_root()
_EXTERNAL_ASSET_ROOT = external_root("SYNTHFOREST_ASSET_ROOT", required=False)
_SEARCH_ROOTS = tuple(
    path for path in (_EXTERNAL_ASSET_ROOT, PROJECT_ROOT, Path.cwd()) if path is not None
)

_TEXTURE_DIR_CACHE: dict[str, list[Path]] = {}


def _resolve_context_assets_root(cfg: TerrainConfig) -> Path | None:
    explicit = getattr(cfg.context, "context_assets_dir", None)
    if isinstance(explicit, str) and explicit.strip():
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_dir():
            return candidate

    for rel in ("volumes/city_rs/assets", "data/volumes/city_rs/assets"):
        rel_path = Path(rel)
        for base in _SEARCH_ROOTS:
            candidate = (base / rel_path).resolve()
            if candidate.is_dir():
                return candidate
    return None


def _list_images_in_dir(
    dir_path: Path, recursive: bool = False, max_files: int = 2000
) -> list[Path]:
    key = f"{dir_path.resolve()}|rec={int(recursive)}"
    cached = _TEXTURE_DIR_CACHE.get(key)
    if cached is not None:
        return list(cached)

    results: list[Path] = []
    if not dir_path.is_dir():
        _TEXTURE_DIR_CACHE[key] = []
        return []

    if recursive:
        for root, _, files in os.walk(dir_path):
            for filename in files:
                if len(results) >= max_files:
                    break
                p = Path(root) / filename
                if p.suffix.lower() in _IMAGE_EXTS and p.is_file():
                    results.append(p)
            if len(results) >= max_files:
                break
    else:
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                results.append(p)
                if len(results) >= max_files:
                    break

    _TEXTURE_DIR_CACHE[key] = results
    return list(results)


def _pick_texture_from_relative_dirs(
    assets_root: Path,
    rel_dirs: list[str],
    rng: random.Random,
) -> Path | None:
    candidates: list[Path] = []
    for rel in rel_dirs:
        rel_str = str(rel).strip()
        if not rel_str:
            continue
        tex_dir = (assets_root / rel_str).resolve()
        candidates.extend(_list_images_in_dir(tex_dir, recursive=False))
    if not candidates:
        return None
    return rng.choice(candidates)


def _pick_texture_from_dir(
    dir_path: Path,
    rng: random.Random,
    recursive: bool = False,
) -> Path | None:
    candidates = _list_images_in_dir(dir_path, recursive=recursive)
    if not candidates:
        return None
    return rng.choice(candidates)


def _pick_texture_by_keywords(
    candidates: list[Path],
    rng: random.Random,
    *,
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...] = (),
) -> Path | None:
    if not candidates:
        return None
    filtered: list[Path] = []
    for candidate in candidates:
        name = candidate.name.lower()
        if exclude_keywords and any(keyword in name for keyword in exclude_keywords):
            continue
        if include_keywords and not any(keyword in name for keyword in include_keywords):
            continue
        filtered.append(candidate)
    if filtered:
        return rng.choice(filtered)
    return rng.choice(candidates)


def build_context_materials(
    terrain_cfg: TerrainConfig,
    rng: random.Random,
) -> tuple[bpy.types.Material, bpy.types.Material]:
    road_fallback = make_material("Context_Road", (0.26, 0.24, 0.20, 1.0), roughness=0.97)
    water_fallback = make_material("Context_Water", (0.09, 0.19, 0.22, 1.0), roughness=0.24)

    assets_root = _resolve_context_assets_root(terrain_cfg)
    if assets_root is None:
        logger.debug("Context texture assets not found; using fallback materials.")
        return road_fallback, water_fallback

    preferred_trail_dir = (assets_root / "textures_ver2" / "road_txt2img_IF").resolve()
    preferred_river_dir = (assets_root / "textures" / "water_textures").resolve()

    trail_candidates = _list_images_in_dir(preferred_trail_dir, recursive=False)
    road_tex = _pick_texture_by_keywords(
        trail_candidates,
        rng=rng,
        include_keywords=("dirtroad", "trail", "path", "mud", "gravel"),
    )
    logger.debug(
        "Found %d candidate trail textures under %s; picked %s",
        len(trail_candidates),
        preferred_trail_dir,
        road_tex,
    )
    if road_tex is None:
        road_tex = _pick_texture_from_dir(preferred_trail_dir, rng=rng, recursive=False)
    if road_tex is None:
        road_tex = _pick_texture_from_relative_dirs(
            assets_root,
            rel_dirs=[
                "textures_ver2/road_txt2img_IF",
                "textures_ver2/bareland_mix_txt2img",
                "textures/road",
                "textures/road_textures",
            ],
            rng=rng,
        )
    if road_tex is not None:
        try:
            road_mat = make_path_blend_material_from_image(
                name="Context_Road",
                image_path=road_tex,
                roughness=0.97,
            )
            logger.info("Using context trail texture: %s", road_tex)
        except Exception as exc:
            logger.warning("Failed to build textured trail material from %s: %s", road_tex, exc)
            road_mat = road_fallback
    else:
        logger.debug("No road textures found under %s; using fallback road material.", assets_root)
        road_mat = road_fallback

    water_candidates = _list_images_in_dir(preferred_river_dir, recursive=True, max_files=1500)
    water_tex = _pick_texture_by_keywords(
        water_candidates,
        rng=rng,
        include_keywords=("water", "ripple", "river"),
    )
    if water_tex is None:
        water_tex = _pick_texture_from_dir(preferred_river_dir, rng=rng, recursive=True)
    if water_tex is None:
        water_dir = (assets_root / "water").resolve()
        if water_dir.is_dir():
            water_candidates = _list_images_in_dir(water_dir, recursive=True, max_files=1500)
            water_tex = rng.choice(water_candidates) if water_candidates else None
    if water_tex is not None:
        try:
            water_mat = make_water_material_from_image(
                name="Context_Water",
                image_path=water_tex,
                shallow_tint=(0.21, 0.31, 0.29, 1.0),
                deep_tint=(0.06, 0.14, 0.18, 1.0),
                bank_tint=(0.20, 0.25, 0.21, 1.0),
                roughness=0.18,
            )
            logger.info("Using context water texture: %s", water_tex)
        except Exception as exc:
            logger.warning("Failed to build textured water material from %s: %s", water_tex, exc)
            water_mat = water_fallback
    else:
        logger.debug(
            "No water textures found under %s; using fallback water material.",
            preferred_river_dir,
        )
        water_mat = water_fallback

    return road_mat, water_mat


def collect_building_textures(
    terrain_cfg: TerrainConfig,
) -> tuple[list[Path], list[Path]]:
    assets_root = _resolve_context_assets_root(terrain_cfg)
    if assets_root is None:
        return [], []

    wall_dirs = [
        "textures_ver2/pave_mix_txt2img",
        "textures_ver2/sidewalk_txt2img_IF",
        "textures_ver2/container_img2img_sd",
        "textures_ver2/airplaneconcrete_img2img_sd",
    ]
    roof_dirs = [
        "textures_ver2/ruralroof_txt2img_IF",
        "textures_ver2/urbanroof_txt2img_IF",
    ]

    wall_candidates: list[Path] = []
    roof_candidates: list[Path] = []
    for rel in wall_dirs:
        wall_candidates.extend(_list_images_in_dir((assets_root / rel).resolve(), recursive=False))
    for rel in roof_dirs:
        roof_candidates.extend(_list_images_in_dir((assets_root / rel).resolve(), recursive=False))
    return wall_candidates, roof_candidates


def build_building_material_pair(
    rng: random.Random,
    index: int,
    wall_candidates: list[Path],
    roof_candidates: list[Path],
) -> tuple[bpy.types.Material, bpy.types.Material]:
    wall_name = f"Context_Building_Wall_{index:03d}"
    roof_name = f"Context_Building_Roof_{index:03d}"

    wall_tex = rng.choice(wall_candidates) if wall_candidates else None
    roof_tex = rng.choice(roof_candidates) if roof_candidates else None

    if wall_tex is not None:
        try:
            wall_mat = make_material_from_image(
                name=wall_name,
                image_path=wall_tex,
                roughness=0.9,
                texture_tiling=1.0,
            )
        except Exception:
            wall_mat = make_material(
                wall_name,
                (rng.uniform(0.45, 0.68), rng.uniform(0.43, 0.65), rng.uniform(0.40, 0.62), 1.0),
                roughness=0.85,
            )
    else:
        wall_mat = make_material(
            wall_name,
            (rng.uniform(0.45, 0.68), rng.uniform(0.43, 0.65), rng.uniform(0.40, 0.62), 1.0),
            roughness=0.85,
        )

    if roof_tex is not None:
        try:
            roof_mat = make_material_from_image(
                name=roof_name,
                image_path=roof_tex,
                roughness=0.75,
                texture_tiling=1.0,
            )
        except Exception:
            roof_mat = make_material(
                roof_name,
                (rng.uniform(0.20, 0.45), rng.uniform(0.12, 0.30), rng.uniform(0.10, 0.24), 1.0),
                roughness=0.75,
            )
    else:
        roof_mat = make_material(
            roof_name,
            (rng.uniform(0.20, 0.45), rng.uniform(0.12, 0.30), rng.uniform(0.10, 0.24), 1.0),
            roughness=0.75,
        )
    logger.debug(
        "Building materials | wall_tex=%s | roof_tex=%s",
        wall_tex,
        roof_tex,
    )

    return wall_mat, roof_mat


def _resolve_project_path(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_profile_overrides(cfg: TerrainConfig, profile_name: str) -> dict[str, object]:
    catalog_path = _resolve_project_path(str(getattr(cfg.context, "profile_catalog_path", "")))
    if not catalog_path.is_file():
        return {}

    profile_name = str(profile_name).strip()
    if profile_name.lower() in {"none", "null"}:
        profile_name = ""
    if not profile_name:
        return {}

    try:
        loaded = OmegaConf.to_container(OmegaConf.load(catalog_path), resolve=True)
    except Exception:
        logger.warning("Failed to load context profile catalog: %s", catalog_path)
        return {}

    if not isinstance(loaded, dict):
        return {}

    profiles_root = loaded.get("context_profiles", loaded)
    if not isinstance(profiles_root, dict):
        return {}

    profile_data = profiles_root.get(profile_name)
    if not isinstance(profile_data, dict):
        logger.warning(
            "Context profile '%s' not found in %s",
            profile_name,
            catalog_path,
        )
        return {}

    return profile_data


def context_value(cfg: TerrainConfig, profile_overrides: dict[str, object], key: str) -> object:
    if key in profile_overrides:
        return profile_overrides[key]
    return getattr(cfg.context, key)
