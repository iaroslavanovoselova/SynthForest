from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class ProductFold:
    dataset_name: str
    location: str
    product_name: str
    tile_dir: Path
    truth_coco_path: Path
    aoi_gpkg_path: Optional[Path]
    truth_gpkg_path: Optional[Path]


@dataclass(frozen=True)
class PreprocessedDatasetSpec:
    dataset_name: str
    ground_resolution: Optional[float]
    scale_factor: Optional[float]
    tile_level_eval_max_dets: int
    locations: list[str]
    products_per_location: dict[str, list[str]]
    hf_dataset_name: Optional[str] = None
    raster_level_eval_inputs: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)

    def iter_fold(
        self,
        root_output_path: str | Path,
        fold: str,
        include_raster_inputs: bool,
    ) -> Iterator[ProductFold]:
        root = Path(root_output_path)
        fold_products = self.raster_level_eval_inputs.get(fold)
        for location in self.locations:
            for product_name in self.products_per_location[location]:
                if fold_products is not None and product_name not in fold_products:
                    continue
                product_root = root / location / product_name
                tile_dir = product_root / "tiles" / fold
                truth_coco_path = self.resolve_truth_coco_path(product_root, product_name, fold)

                aoi_gpkg_path = None
                truth_gpkg_path = None
                if include_raster_inputs:
                    raster_inputs = self.raster_level_eval_inputs.get(fold, {}).get(
                        product_name, {}
                    )
                    if "ground_truth_gpkg" in raster_inputs and "aoi_gpkg" in raster_inputs:
                        truth_gpkg_path = product_root / raster_inputs["ground_truth_gpkg"]
                        aoi_gpkg_path = product_root / raster_inputs["aoi_gpkg"]

                yield ProductFold(
                    dataset_name=self.dataset_name,
                    location=location,
                    product_name=product_name,
                    tile_dir=tile_dir,
                    truth_coco_path=truth_coco_path,
                    aoi_gpkg_path=aoi_gpkg_path,
                    truth_gpkg_path=truth_gpkg_path,
                )

    def resolve_truth_coco_path(self, product_root: Path, product_name: str, fold: str) -> Path:
        candidates: list[Path] = []
        if self.ground_resolution is not None:
            candidates.append(
                product_root
                / _coco_name(product_name, fold, ground_resolution=self.ground_resolution)
            )
        if self.scale_factor is not None:
            candidates.append(
                product_root / _coco_name(product_name, fold, scale_factor=self.scale_factor)
            )
        candidates.append(product_root / f"{product_name}_coco_{fold}.json")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def verify_fold(
        self,
        root_output_path: str | Path,
        fold: str,
        include_raster_inputs: bool,
    ) -> None:
        missing: list[str] = []
        for product in self.iter_fold(
            root_output_path=root_output_path,
            fold=fold,
            include_raster_inputs=include_raster_inputs,
        ):
            if not product.tile_dir.exists():
                missing.append(f"tiles: {product.tile_dir}")
            if not product.truth_coco_path.exists():
                missing.append(f"truth COCO: {product.truth_coco_path}")
            if include_raster_inputs:
                if product.truth_gpkg_path is None or product.aoi_gpkg_path is None:
                    missing.append(
                        f"raster GPKG metadata for {product.location}/{product.product_name}"
                    )
                else:
                    if not product.truth_gpkg_path.exists():
                        missing.append(f"truth GPKG: {product.truth_gpkg_path}")
                    if not product.aoi_gpkg_path.exists():
                        missing.append(f"AOI GPKG: {product.aoi_gpkg_path}")
        if missing:
            details = "\n  - ".join(missing[:25])
            extra = "" if len(missing) <= 25 else f"\n  - ... and {len(missing) - 25} more"
            raise FileNotFoundError(
                f"Dataset {self.dataset_name!r} is missing required {fold!r} benchmark files "
                f"under {root_output_path}:\n  - {details}{extra}"
            )


def get_dataset_spec(dataset_name: str) -> PreprocessedDatasetSpec:
    try:
        return DATASET_REGISTRY[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(
            f"Unsupported benchmark dataset {dataset_name!r}. Supported: {supported}"
        ) from exc


def _coco_name(
    product_name: str,
    fold: str,
    ground_resolution: Optional[float] = None,
    scale_factor: Optional[float] = None,
) -> str:
    if ground_resolution is not None:
        return f"{product_name}_coco_gr{_format_number(ground_resolution)}_{fold}.json"
    if scale_factor is not None:
        return f"{product_name}_coco_sf{_format_number(scale_factor)}_{fold}.json"
    return f"{product_name}_coco_{fold}.json"


def _format_number(value: float) -> str:
    text = f"{float(value):.12g}"
    if "." not in text:
        text = f"{text}.0"
    return text.replace(".", "p")


DATASET_REGISTRY: dict[str, PreprocessedDatasetSpec] = {
    "SelvaBox": PreprocessedDatasetSpec(
        dataset_name="SelvaBox",
        ground_resolution=0.045,
        scale_factor=None,
        tile_level_eval_max_dets=400,
        locations=["brazil_zf2", "ecuador_tiputini", "panama_aguasalud"],
        products_per_location={
            "brazil_zf2": [
                "20240130_zf2quad_m3m_rgb",
                "20240130_zf2tower_m3m_rgb",
                "20240130_zf2transectew_m3m_rgb",
                "20240131_zf2campirana_m3m_rgb",
            ],
            "ecuador_tiputini": [
                "20170810_transectotoni_mavicpro_rgb",
                "20230525_tbslake_m3e_rgb",
                "20230911_sanitower_mini2_rgb",
                "20231018_inundated_m3e_rgb",
                "20231018_pantano_m3e_rgb",
                "20231018_terrafirme_m3e_rgb",
            ],
            "panama_aguasalud": [
                "20231207_asnortheast_amsunclouds_m3m_rgb",
                "20231207_asnorthnorth_pmclouds_m3m_rgb",
                "20231208_asforestnorthe2_m3m_rgb",
                "20231208_asforestsouth2_m3m_rgb",
            ],
        },
        raster_level_eval_inputs={
            "valid": {
                "20240130_zf2quad_m3m_rgb": {
                    "ground_truth_gpkg": "20240130_zf2quad_m3m_labels_boxes.gpkg",
                    "aoi_gpkg": "20240130_zf2quad_m3m_labels_boxes_aoi_valid.gpkg",
                },
                "20231018_inundated_m3e_rgb": {
                    "ground_truth_gpkg": "20231018_inundated_m3e_labels_boxes.gpkg",
                    "aoi_gpkg": "20231018_inundated_m3e_labels_boxes_aoi_valid.gpkg",
                },
                "20231207_asnortheast_amsunclouds_m3m_rgb": {
                    "ground_truth_gpkg": "20231207_asnortheast_amsunclouds_m3m_labels_boxes.gpkg",
                    "aoi_gpkg": "20231207_asnortheast_amsunclouds_m3m_labels_boxes_aoi_valid.gpkg",
                },
                "20231208_asforestnorthe2_m3m_rgb": {
                    "ground_truth_gpkg": "20231208_asforestnorthe2_m3m_labels_boxes.gpkg",
                    "aoi_gpkg": "20231208_asforestnorthe2_m3m_labels_boxes_aoi_valid.gpkg",
                },
            },
            "test": {
                "20240130_zf2tower_m3m_rgb": {
                    "ground_truth_gpkg": "20240130_zf2tower_m3m_labels_boxes.gpkg",
                    "aoi_gpkg": "20240130_zf2tower_m3m_labels_boxes_aoi_test.gpkg",
                },
                "20230525_tbslake_m3e_rgb": {
                    "ground_truth_gpkg": "20230525_tbslake_m3e_labels_boxes.gpkg",
                    "aoi_gpkg": "20230525_tbslake_m3e_labels_boxes_aoi_test.gpkg",
                },
                "20231018_inundated_m3e_rgb": {
                    "ground_truth_gpkg": "20231018_inundated_m3e_labels_boxes.gpkg",
                    "aoi_gpkg": "20231018_inundated_m3e_labels_boxes_aoi_test.gpkg",
                },
                "20231207_asnortheast_amsunclouds_m3m_rgb": {
                    "ground_truth_gpkg": "20231207_asnortheast_amsunclouds_m3m_labels_boxes.gpkg",
                    "aoi_gpkg": "20231207_asnortheast_amsunclouds_m3m_labels_boxes_aoi_test.gpkg",
                },
                "20231208_asforestnorthe2_m3m_rgb": {
                    "ground_truth_gpkg": "20231208_asforestnorthe2_m3m_labels_boxes.gpkg",
                    "aoi_gpkg": "20231208_asforestnorthe2_m3m_labels_boxes_aoi_test.gpkg",
                },
            },
        },
    ),
    "BCI50ha": PreprocessedDatasetSpec(
        dataset_name="BCI50ha",
        ground_resolution=0.045,
        scale_factor=None,
        tile_level_eval_max_dets=100,
        locations=["panama_bci50ha"],
        products_per_location={
            "panama_bci50ha": [
                "bci_50ha_2020_08_01_crownmap_raw",
                "bci_50ha_2022_09_29_crownmap_raw",
            ]
        },
        raster_level_eval_inputs={
            "test": {
                "bci_50ha_2020_08_01_crownmap_raw": {
                    "ground_truth_gpkg": "BCI_50ha_2020_08_01_crownmap_improved.gpkg",
                    "aoi_gpkg": "20200801_aoi_test_without_holes.gpkg",
                },
                "bci_50ha_2022_09_29_crownmap_raw": {
                    "ground_truth_gpkg": "BCI_50ha_2022_09_29_crownmap_improved.gpkg",
                    "aoi_gpkg": "20220929_aoi_test_without_holes.gpkg",
                },
            }
        },
    ),
    "QuebecTrees": PreprocessedDatasetSpec(
        dataset_name="QuebecTrees",
        ground_resolution=0.03,
        scale_factor=None,
        tile_level_eval_max_dets=400,
        locations=["quebec_trees"],
        products_per_location={
            "quebec_trees": [
                "20210902_sblz1_p4rtk_rgb",
                "20210902_sblz2_p4rtk_rgb",
                "20210902_sblz3_p4rtk_rgb",
            ]
        },
        raster_level_eval_inputs={
            "valid": {
                "20210902_sblz1_p4rtk_rgb": {
                    "ground_truth_gpkg": "20210902_sblz1_p4rtk_labels_masks.gpkg",
                    "aoi_gpkg": "20210902_sblz1_p4rtk_labels_masks_aoi_valid.gpkg",
                }
            },
            "test": {
                "20210902_sblz1_p4rtk_rgb": {
                    "ground_truth_gpkg": "20210902_sblz1_p4rtk_labels_masks.gpkg",
                    "aoi_gpkg": "20210902_sblz1_p4rtk_labels_masks_aoi_test.gpkg",
                },
                "20210902_sblz3_p4rtk_rgb": {
                    "ground_truth_gpkg": "20210902_sblz3_p4rtk_labels_masks.gpkg",
                    "aoi_gpkg": "20210902_sblz3_p4rtk_labels_masks_aoi_test.gpkg",
                },
            },
        },
    ),
    "NeonTreeEvaluation": PreprocessedDatasetSpec(
        dataset_name="NeonTreeEvaluation",
        ground_resolution=0.1,
        scale_factor=1.0,
        tile_level_eval_max_dets=100,
        locations=["unitedstates_neon"],
        products_per_location={
            "unitedstates_neon": [
                "2018_bart_4_322000_4882000_image_crop",
                "2018_harv_5_733000_4698000_image_crop",
                "2018_jerc_4_742000_3451000_image_crop",
                "2018_mlbs_3_541000_4140000_image_crop",
                "2018_mlbs_3_541000_4140000_image_crop2",
                "2018_niwo_2_450000_4426000_image_crop",
                "2018_osbs_4_405000_3286000_image",
                "2018_sjer_3_258000_4106000_image",
                "2018_sjer_3_259000_4110000_image",
                "2018_teak_3_315000_4094000_image_crop",
                "2019_dela_5_423000_3601000_image_crop",
                "2019_leno_5_383000_3523000_image_crop",
                "2019_onaq_2_367000_4449000_image_crop",
                "2019_osbs_5_405000_3287000_image_crop",
                "2019_osbs_5_405000_3287000_image_crop2",
                "2019_sjer_4_251000_4103000_image",
                "2019_tool_3_403000_7617000_image",
                "NeonTreeEvaluation_Test",
            ]
        },
        raster_level_eval_inputs={
            "valid": {
                "2018_bart_4_322000_4882000_image_crop": {},
                "2018_harv_5_733000_4698000_image_crop": {},
                "2018_jerc_4_742000_3451000_image_crop": {},
                "2018_mlbs_3_541000_4140000_image_crop": {},
                "2018_mlbs_3_541000_4140000_image_crop2": {},
                "2018_niwo_2_450000_4426000_image_crop": {},
                "2018_osbs_4_405000_3286000_image": {},
                "2018_sjer_3_258000_4106000_image": {},
                "2018_sjer_3_259000_4110000_image": {},
                "2018_teak_3_315000_4094000_image_crop": {},
                "2019_dela_5_423000_3601000_image_crop": {},
                "2019_leno_5_383000_3523000_image_crop": {},
                "2019_onaq_2_367000_4449000_image_crop": {},
                "2019_osbs_5_405000_3287000_image_crop": {},
                "2019_osbs_5_405000_3287000_image_crop2": {},
            },
            "test": {"NeonTreeEvaluation_Test": {}},
        },
    ),
    "OAM-TCD": PreprocessedDatasetSpec(
        dataset_name="OAM-TCD",
        ground_resolution=0.1,
        scale_factor=1.0,
        tile_level_eval_max_dets=400,
        locations=["global_oamtcd"],
        products_per_location={"global_oamtcd": ["oam_tcd"]},
        raster_level_eval_inputs={"valid": {"oam_tcd": {}}, "test": {"oam_tcd": {}}},
    ),
    "Detectree2": PreprocessedDatasetSpec(
        dataset_name="Detectree2",
        ground_resolution=0.1,
        scale_factor=1.0,
        tile_level_eval_max_dets=100,
        locations=["malaysia_detectree2"],
        products_per_location={
            "malaysia_detectree2": [
                "dan_2014_rgb_project_to_chm",
                "sep_ma14_21_orthomosaic_20141023_reprojected_full_res",
            ]
        },
        raster_level_eval_inputs={
            "valid": {
                "dan_2014_rgb_project_to_chm": {
                    "ground_truth_gpkg": "Dan_2014_RGB_project_to_CHM_labels.gpkg",
                    "aoi_gpkg": "detectree2_aoi_valid.gpkg",
                },
                "sep_ma14_21_orthomosaic_20141023_reprojected_full_res": {
                    "ground_truth_gpkg": "Sep_MA14_21_orthomosaic_20141023_reprojected_full_res_labels.gpkg",
                    "aoi_gpkg": "detectree2_aoi_valid.gpkg",
                },
            },
            "test": {
                "dan_2014_rgb_project_to_chm": {
                    "ground_truth_gpkg": "Dan_2014_RGB_project_to_CHM_labels.gpkg",
                    "aoi_gpkg": "detectree2_aoi_test.gpkg",
                },
                "sep_ma14_21_orthomosaic_20141023_reprojected_full_res": {
                    "ground_truth_gpkg": "Sep_MA14_21_orthomosaic_20141023_reprojected_full_res_labels.gpkg",
                    "aoi_gpkg": "detectree2_aoi_test.gpkg",
                },
            },
        },
    ),
    "BAMForest": PreprocessedDatasetSpec(
        dataset_name="BAMForest",
        hf_dataset_name="BAMForests",
        ground_resolution=0.017,
        scale_factor=1.0,
        tile_level_eval_max_dets=400,
        locations=["bamforests"],
        products_per_location={
            "bamforests": [
                "BAMForest_train2023",
                "BAMForest_val2023",
                "BAMForest_TestSet1_2023",
                "BAMForest_TestSet2_2023",
            ]
        },
        raster_level_eval_inputs={
            "train": {"BAMForest_train2023": {}},
            "valid": {"BAMForest_val2023": {}},
            "test": {"BAMForest_TestSet1_2023": {}, "BAMForest_TestSet2_2023": {}},
        },
    ),
    "PanamaBCNM": PreprocessedDatasetSpec(
        dataset_name="PanamaBCNM",
        ground_resolution=0.045,
        scale_factor=None,
        tile_level_eval_max_dets=400,
        locations=["panama_bcnm"],
        products_per_location={
            "panama_bcnm": [
                "20241125_bcigiganteplot_m3e_rgb_cog",
                "2024_07_16_orthowhole_bci_resfull_cropped_25haplot_cog",
                "2024_07_16_orthowhole_bci_resfull_cropped_ava6haplot_cog",
                "2024_07_16_orthowhole_bci_resfull_cropped_drayton6haplot_cog",
                "2024_07_16_orthowhole_bci_resfull_cropped_pearson6haplot_cog",
                "2024_07_16_orthowhole_bci_resfull_cropped_zetek6haplot_cog",
                "bci_50ha_2024_11_12_cropped_cog",
            ]
        },
        raster_level_eval_inputs={
            "test": {
                "20241125_bcigiganteplot_m3e_rgb_cog": {
                    "ground_truth_gpkg": "BCNM_gigante_crownmap_2025.gpkg",
                    "aoi_gpkg": "20241125_bcigiganteplot_m3e_rgb_cog_aoi_gr0p045_test.gpkg",
                },
                "2024_07_16_orthowhole_bci_resfull_cropped_25haplot_cog": {
                    "ground_truth_gpkg": "BCI_25ha_crownmap_2025.gpkg",
                    "aoi_gpkg": "2024_07_16_orthowhole_bci_resfull_cropped_25haplot_cog_aoi_gr0p045_test.gpkg",
                },
                "2024_07_16_orthowhole_bci_resfull_cropped_ava6haplot_cog": {
                    "ground_truth_gpkg": "BCI_ava_crownmap_2025.gpkg",
                    "aoi_gpkg": "2024_07_16_orthowhole_bci_resfull_cropped_ava6haplot_cog_aoi_gr0p045_test.gpkg",
                },
                "2024_07_16_orthowhole_bci_resfull_cropped_drayton6haplot_cog": {
                    "ground_truth_gpkg": "BCI_drayton_crownmap_2025.gpkg",
                    "aoi_gpkg": "2024_07_16_orthowhole_bci_resfull_cropped_drayton6haplot_cog_aoi_gr0p045_test.gpkg",
                },
                "2024_07_16_orthowhole_bci_resfull_cropped_pearson6haplot_cog": {
                    "ground_truth_gpkg": "BCI_pearson_crownmap_2025.gpkg",
                    "aoi_gpkg": "2024_07_16_orthowhole_bci_resfull_cropped_pearson6haplot_cog_aoi_gr0p045_test.gpkg",
                },
                "2024_07_16_orthowhole_bci_resfull_cropped_zetek6haplot_cog": {
                    "ground_truth_gpkg": "BCI_zetek_crownmap_2025.gpkg",
                    "aoi_gpkg": "2024_07_16_orthowhole_bci_resfull_cropped_zetek6haplot_cog_aoi_gr0p045_test.gpkg",
                },
                "bci_50ha_2024_11_12_cropped_cog": {
                    "ground_truth_gpkg": "BCI_50ha_crownmap_2025.gpkg",
                    "aoi_gpkg": "bci_50ha_2024_11_12_cropped_cog_aoi_gr0p045_test.gpkg",
                },
            }
        },
    ),
    "SelvaMask": PreprocessedDatasetSpec(
        dataset_name="SelvaMask",
        ground_resolution=0.045,
        scale_factor=None,
        tile_level_eval_max_dets=400,
        locations=["selvamask"],
        products_per_location={
            "selvamask": [
                "20240131_zf2block4_ms_m3m_rgb",
                "20240613_tbsnewsite2_m3e_rgb",
                "20241122_bcifairchildn_m3m_rgb",
            ]
        },
        raster_level_eval_inputs={
            "valid": {
                "20240131_zf2block4_ms_m3m_rgb": {
                    "ground_truth_gpkg": "20240131_zf2block4_ms_m3m_labels_masks.gpkg",
                    "aoi_gpkg": "20240131_zf2block4_ms_m3m_rgb_aoi_gr0p045_valid.gpkg",
                },
                "20240613_tbsnewsite2_m3e_rgb": {
                    "ground_truth_gpkg": "20240613_tbsnewsite2_m3e_labels_masks.gpkg",
                    "aoi_gpkg": "20240613_tbsnewsite2_m3e_rgb_aoi_gr0p045_valid.gpkg",
                },
                "20241122_bcifairchildn_m3m_rgb": {
                    "ground_truth_gpkg": "20241122_bcifairchildn_m3m_rgb_labels_masks.gpkg",
                    "aoi_gpkg": "20241122_bcifairchildn_m3m_rgb_aoi_gr0p045_valid.gpkg",
                },
            },
            "test": {
                "20240131_zf2block4_ms_m3m_rgb": {
                    "ground_truth_gpkg": "20240131_zf2block4_ms_m3m_labels_masks.gpkg",
                    "aoi_gpkg": "20240131_zf2block4_ms_m3m_rgb_aoi_gr0p045_test.gpkg",
                },
                "20240613_tbsnewsite2_m3e_rgb": {
                    "ground_truth_gpkg": "20240613_tbsnewsite2_m3e_labels_masks.gpkg",
                    "aoi_gpkg": "20240613_tbsnewsite2_m3e_rgb_aoi_gr0p045_test.gpkg",
                },
                "20241122_bcifairchildn_m3m_rgb": {
                    "ground_truth_gpkg": "20241122_bcifairchildn_m3m_rgb_labels_masks.gpkg",
                    "aoi_gpkg": "20241122_bcifairchildn_m3m_rgb_aoi_gr0p045_test.gpkg",
                },
            },
        },
    ),
}
