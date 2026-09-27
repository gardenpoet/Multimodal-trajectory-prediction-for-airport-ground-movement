"""
One-off utility: dump an airport's taxiway/runway/hold-line network (from
semantic_graph.pkl, the same reference network OffRoadEvaluator uses for the
on/off-road check) as a flat list of lat/lon edges, for a case-study map
overlay background -- so a trajectory plot shows the actual movement-area
geometry (where the taxiways/runways/intersections are) instead of floating
in blank space.

Static per-airport data, independent of any specific case -- run once per
airport, not per case. No model/checkpoint/Hydra config needed, just the
assets directory.

Usage:
    python -m risk_assessment.dump_airport_network \\
        --assets_dir /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/assets \\
        --airport kbos \\
        --output_json /gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_network.json

Output JSON: {"edges": [{"node_type": int, "lat1":, "lon1":, "lat2":, "lon2":}, ...]}
node_type: 1=hold_line, 3=runway, 4=taxiway (per OffRoadEvaluator's own docstring).
"""
import argparse
import json
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets_dir", required=True)
    ap.add_argument("--airport", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    pkl_path = os.path.join(args.assets_dir, args.airport, "semantic_graph.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    G = data["graph_networkx"]

    edges = []
    for u, v, edata in G.edges(data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        if "x" not in u_data or "y" not in u_data:
            continue
        node_type = max(u_data.get("node_type", 4), v_data.get("node_type", 4))
        edges.append({
            "node_type": node_type,
            "lat1": u_data["y"], "lon1": u_data["x"],
            "lat2": v_data["y"], "lon2": v_data["x"],
        })

    with open(args.output_json, "w") as f:
        json.dump({"edges": edges}, f, separators=(",", ":"))
    print(f"[dump_airport_network] wrote {len(edges)} edges to {args.output_json}")


if __name__ == "__main__":
    main()
