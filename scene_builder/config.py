import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build articulated-part dataset from SAPIEN PartModels with SAPIEN ray tracing."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--instance",
        type=str,
        help="Single instance name, e.g. bottle_3398",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Build all discovered instances.",
    )
    parser.add_argument(
        "--start-instance",
        type=str,
        default=None,
        help="When used with --all, start building from this instance name (inclusive), e.g. bottle_3398.",
    )
    parser.add_argument(
        "--end-instance",
        type=str,
        default=None,
        help="When used with --all, stop building at this instance name (inclusive), e.g. bottle_3398.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/Models",
        help="Root containing PartModels folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/inspire/hdd/project/robot-dna/jiangyixuan-CZXS25230137/yuquan/dataset_train",
        help="Dataset output root.",
    )
    parser.add_argument("--views", type=int, default=50, help="Views per instance.")
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Rendered image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Rendered image height.",
    )
    parser.add_argument("--fov-deg", type=float, default=35.0)
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.25,
        help="Camera distance multiplier after fitting the object bounding sphere; larger values make objects smaller in the image.",
    )
    parser.add_argument(
        "--target-max-extent",
        type=float,
        default=0.45,
        help="Target maximum object extent in scene units before scale clipping; used to normalize inconsistent source assets without over-enlarging small objects.",
    )
    parser.add_argument(
        "--min-object-scale",
        type=float,
        default=0.25,
        help="Lower bound for URDF loader scale after bbox normalization.",
    )
    parser.add_argument(
        "--max-object-scale",
        type=float,
        default=1.5,
        help="Upper bound for URDF loader scale; keeps small objects from becoming unrealistically large in the image.",
    )
    parser.add_argument(
        "--min-object-coverage",
        type=float,
        default=0.05,
        help="Minimum fraction of image covered by the exported movable-part union mask.",
    )
    parser.add_argument(
        "--max-object-coverage",
        type=float,
        default=0.45,
        help="Maximum fraction of image covered by the exported movable-part union mask; prevents close-up frames that fill the image.",
    )
    parser.add_argument(
        "--min-part-mask-pixels",
        type=int,
        default=30,
        help="Minimum visible mask pixels required for each movable joint/link part in an accepted view.",
    )
    parser.add_argument(
        "--min-part-mask-coverage",
        type=float,
        default=0.00015,
        help="Additional per-image minimum coverage for each movable joint/link part.",
    )
    parser.add_argument(
        "--require-all-part-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every exported movable joint/link part to be visible in accepted frames. "
            "Default is on; use --no-require-all-part-visible to require only one visible movable part."
        ),
    )
    parser.add_argument(
        "--view-candidate-multiplier",
        type=int,
        default=16,
        help="How many candidate camera directions to test per requested source view.",
    )
    parser.add_argument(
        "--part-occlusion-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use part-only auxiliary renders to reject views where a movable part is mostly hidden "
            "by other object geometry. Default is off for regular builds."
        ),
    )
    parser.add_argument(
        "--min-part-visible-ratio",
        type=float,
        default=0.30,
        help="Minimum full-scene visible pixels divided by part-only projected pixels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--exclude-part-keywords",
        type=str,
        default="base_body,frame,body",
        help=(
            "Comma-separated case-insensitive keywords for part/link names to exclude "
            "from exported models, masks, and cam_params. Empty string disables this filter."
        ),
    )
    parser.add_argument(
        "--render-ground",
        action="store_true",
        help="Render ground plane. Default is off to avoid object-ground intersection/occlusion.",
    )
    rt_group = parser.add_mutually_exclusive_group()
    rt_group.add_argument(
        "--rt",
        dest="rt",
        action="store_true",
        default=True,
        help=(
            "Use ray-tracing renderer settings. This is the default and matches the "
            "original rendering path that exposes SAPIEN camera buffers reliably."
        ),
    )
    rt_group.add_argument(
        "--no-rt",
        dest="rt",
        action="store_false",
        help="Disable ray-tracing renderer settings.",
    )
    rt_camera_group = parser.add_mutually_exclusive_group()
    rt_camera_group.add_argument(
        "--rt-camera-shader",
        dest="rt_camera_shader",
        action="store_true",
        default=False,
        help=(
            "Use the ray-tracing camera shader for saved RGB. Off by default because "
            "dataset export needs Position/Segmentation buffers for depth and masks."
        ),
    )
    rt_camera_group.add_argument(
        "--no-rt-camera-shader",
        dest="rt_camera_shader",
        action="store_false",
        help=(
            "Keep the default camera shader while retaining other rt settings. "
            "Use this if your local SAPIEN build cannot export Position/Segmentation under rt."
        ),
    )
    parser.add_argument(
        "--rt-spp",
        type=int,
        default=64,
        help="Ray-tracing samples per pixel. 32 is faster; 64/128 is cleaner.",
    )
    parser.add_argument(
        "--rt-path-depth",
        type=int,
        default=8,
        help="Ray-tracing path depth if supported by the installed SAPIEN version.",
    )
    parser.add_argument(
        "--rt-denoiser",
        type=str,
        default="optix",
        choices=["optix", "oidn", "none"],
        help="Ray-tracing denoiser. Use 'none' to disable denoising.",
    )
    parser.add_argument(
        "--background-mode",
        type=str,
        default="composite",
        choices=["plain", "composite", "scene"],
        help=(
            "plain keeps the original renderer background; composite replaces non-object "
            "pixels in RGB with deterministic non-gray studio/tabletop backgrounds while "
            "leaving depth, masks, segmentation, and poses unchanged; scene adds simple "
            "SAPIEN ground/wall geometry and randomized lighting."
        ),
    )
    parser.add_argument(
        "--background-variants",
        type=int,
        default=4,
        help=(
            "Number of RGB background variants per camera view. Total exported frames "
            "become views * background_variants when background_mode=composite. "
            "Plain and scene modes render one frame per view."
        ),
    )
    parser.add_argument(
        "--background-seed",
        type=int,
        default=2026,
        help="Seed for deterministic per-instance background generation.",
    )
    joint_group = parser.add_mutually_exclusive_group()
    joint_group.add_argument(
        "--joint-motion",
        dest="joint_motion",
        action="store_true",
        default=True,
        help="Apply small deterministic joint motion per source view. Enabled by default.",
    )
    joint_group.add_argument(
        "--no-joint-motion",
        dest="joint_motion",
        action="store_false",
        help="Keep all articulation joints at their loaded default positions.",
    )
    parser.add_argument(
        "--joint-motion-fraction",
        type=float,
        default=0.01,
        help="Maximum fraction of each limited joint range to use for motion.",
    )
    parser.add_argument(
        "--joint-motion-max-delta",
        type=float,
        default=3.0,
        help="Maximum absolute qpos delta for a moved joint; keeps poses modest.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes. Use 1 for single-process; <=0 means auto.",
    )
    parser.add_argument(
        "--max-tasks-per-worker",
        type=int,
        default=20,
        help=(
            "Restart each worker process after this many instances to limit long-run "
            "SAPIEN/renderer memory growth. Use <=0 to disable worker recycling."
        ),
    )
    parser.add_argument(
        "--instance-timeout",
        type=int,
        default=1800,
        help="Maximum seconds allowed for one instance in single-worker mode. Use <=0 to disable.",
    )
    return parser.parse_args()
