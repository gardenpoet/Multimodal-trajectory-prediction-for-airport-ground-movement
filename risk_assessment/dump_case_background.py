"""
One-off utility: crop the airport's real background image (assets/{airport}/
bkg_map.png -- the actual diagram/imagery amelia_scenes/visualization's
load_assets() uses, referenced but never previously exercised in this
folder) to the small local area around one case, for a real (not schematic)
map background in the case-study trajectory plot.

limits.json's espg_4326 north/east/south/west give the WHOLE image's
geographic extent; bkg_map.png is assumed equirectangular (linear lat/lon
per pixel row/col) across that extent, matching how load_assets() uses the
same two files together (imshow(bkg, extent=[west,east,south,north])).
Crops to the requested bounds (clamped to the image's own extent) and
writes both the cropped PNG and its ACTUAL (possibly clamped) geographic
bounds, since the crop request may extend past the image edge.

Usage:
    python -m risk_assessment.dump_case_background \\
        --assets_dir /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/assets \\
        --airport kbos \\
        --north 42.375110 --south 42.357192 --east -71.008830 --west -71.021086 \\
        --output_png /gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_103_47_bg.png \\
        --output_json /gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_103_47_bg.json
"""
import argparse
import json
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets_dir", required=True)
    ap.add_argument("--airport", required=True)
    ap.add_argument("--north", type=float, required=True)
    ap.add_argument("--south", type=float, required=True)
    ap.add_argument("--east", type=float, required=True)
    ap.add_argument("--west", type=float, required=True)
    ap.add_argument("--output_png", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    limits_path = os.path.join(args.assets_dir, args.airport, "limits.json")
    with open(limits_path) as f:
        limits = json.load(f)
    espg = limits["espg_4326"]
    img_north, img_south = espg["north"], espg["south"]
    img_east, img_west = espg["east"], espg["west"]

    img_path = os.path.join(args.assets_dir, args.airport, "bkg_map.png")
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")
    h, w = img.shape[:2]

    # Clamp the requested bounds to the image's own extent.
    north = min(args.north, img_north)
    south = max(args.south, img_south)
    east = min(args.east, img_east)
    west = max(args.west, img_west)

    def lat_to_row(lat):
        # row 0 is the TOP of the image, which corresponds to img_north.
        t = (img_north - lat) / (img_north - img_south)
        return int(round(t * h))

    def lon_to_col(lon):
        t = (lon - img_west) / (img_east - img_west)
        return int(round(t * w))

    r0, r1 = sorted([lat_to_row(north), lat_to_row(south)])
    c0, c1 = sorted([lon_to_col(west), lon_to_col(east)])
    r0, r1 = max(0, r0), min(h, r1)
    c0, c1 = max(0, c0), min(w, c1)
    if r1 <= r0 or c1 <= c0:
        raise RuntimeError("Requested crop does not overlap the image extent.")

    crop = img[r0:r1, c0:c1]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output_png, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    # Actual geographic bounds of the saved crop (row/col back to lat/lon).
    actual_north = img_north - (r0 / h) * (img_north - img_south)
    actual_south = img_north - (r1 / h) * (img_north - img_south)
    actual_west = img_west + (c0 / w) * (img_east - img_west)
    actual_east = img_west + (c1 / w) * (img_east - img_west)

    with open(args.output_json, "w") as f:
        json.dump({
            "north": actual_north, "south": actual_south,
            "east": actual_east, "west": actual_west,
            "width_px": c1 - c0, "height_px": r1 - r0,
        }, f)
    print(f"[dump_case_background] wrote {args.output_png} "
          f"({c1-c0}x{r1-r0}px) and {args.output_json}")


if __name__ == "__main__":
    main()
