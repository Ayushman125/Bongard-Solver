import json
import networkx as nx
from pathlib import Path

DSL_OPS = {
    "left_of": "(move {0} left_of {1})",
    "above":  "(move {0} above {1})",
    "overlap":"(group {0} {1})"
}

def synthesize_programs(graph_dir: str, out_prog_dir: str):
    graph_dir = Path(graph_dir)
    out_dir   = Path(out_prog_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from networkx.readwrite import json_graph
    for gp in graph_dir.glob("*.json"):
        G = json_graph.node_link_graph(json.load(gp.open()))
        steps = []
        for u, v, d in G.edges(data=True):
            rel = d.get("relation")
            if rel in DSL_OPS:
                steps.append(DSL_OPS[rel].format(u, v))

        prog_txt = "\n".join(steps) if steps else "; no relations found"
        (out_dir / f"{gp.stem}.lisp").write_text(prog_txt)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--graph_dir",   default="graphs")
    p.add_argument("--out_prog_dir",default="programs")
    args = p.parse_args()
    synthesize_programs(args.graph_dir, args.out_prog_dir)
