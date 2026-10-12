"""Fixed, reproducible letter strokes and a long-distance A* arena."""
import json
from pathlib import Path
import numpy as np
from track_astar_v6 import ROOT, SCENE


def rounded(points, radius=0.65):
    points = np.asarray(points, float)
    result = [points[0]]
    for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
        incoming, outgoing = b-a, c-b
        cut = min(radius, .35*np.linalg.norm(incoming), .35*np.linalg.norm(outgoing))
        p, q = b-cut*incoming/np.linalg.norm(incoming), b+cut*outgoing/np.linalg.norm(outgoing)
        result.append(p)
        for t in np.linspace(0, 1, 12)[1:]:
            result.append((1-t)**2*p+2*t*(1-t)*b+t*t*q)
    result.append(points[-1])
    return np.asarray(result)


def make():
    output = ROOT / "record/v6/navigation_demos_20261002"
    output.mkdir(parents=True, exist_ok=True)
    # Centre-line capitals; transit curves join disconnected strokes physically.
    letters = {
        "F": [[(3,5),(0,5),(0,0)], [(0,2.6),(2.5,2.6)]],
        "D": [[(0,0),(0,5),(1.8,5),(3,4),(3,1),(1.8,0),(0,0)]],
        "U": [[(0,5),(0,1),(0.8,0),(2.2,0),(3,1),(3,5)]],
        "B": [[(0,0),(0,5)], [(0,5),(2.2,5),(3,4.3),(3,3.4),(2.2,2.6),(0,2.6)],
              [(0,2.6),(2.2,2.6),(3,1.9),(3,.7),(2.2,0),(0,0)]],
        "S": [[(3,5),(0.8,5),(0,4.2),(0,3.3),(.8,2.5),(2.2,2.5),(3,1.7),(3,.8),(2.2,0),(0,0)]],
        "R": [[(0,0),(0,5)], [(0,5),(2.2,5),(3,4.2),(3,3.3),(2.2,2.5),(0,2.5)], [(1.3,2.5),(3,0)]],
        "L": [[(0,5),(0,0),(3,0)]],
    }
    paths, labels, stroke_names = [], [], []
    offset = np.array([-3.5218487172304253, -5.])
    previous = None
    stroke_id = 0
    for index, letter in enumerate("FDUBSRL"):
        x = index*5.5 + (1.5 if index >= 3 else 0.)
        for stroke in letters[letter]:
            current = rounded(np.asarray(stroke)+offset+[x, 0])
            if previous is not None:
                a, d = previous[-1], current[0]
                u = previous[-1]-previous[-2]
                v = current[1]-current[0]
                u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
                handle = max(1.5, .7*np.linalg.norm(d-a))
                b, c = a+handle*u, d-handle*v
                for t in np.linspace(0, 1, max(40, int(np.linalg.norm(d-a)*15)))[1:]:
                    point = (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d
                    if np.linalg.norm(point-paths[-1]) > 1e-8:
                        paths.append(point); labels.append(-1)
            for point in current:
                if not paths or np.linalg.norm(point-paths[-1]) > 1e-8:
                    paths.append(point); labels.append(stroke_id)
            stroke_names.append(letter)
            stroke_id += 1
            previous = current
    path = np.asarray(paths)
    scene = dict(SCENE, bounds_m=[-10, 40, -12, 8], obstacles=[], goal_m=path[-1].tolist())
    payload = {"name":"FDU BSRL ordered letter strokes", "scene":scene, "path_m":path.tolist(),
               "point_stroke_id":labels, "stroke_names":stroke_names,
               "note":"Letter strokes are coloured; continuous physical transfers are grey. No teleportation."}
    (output / "letters_route.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    long_scene = dict(SCENE, bounds_m=[-28, 2, -8, 8], goal_m=[-25., 2.], obstacles=[
        {"center_m":[-5.,0.],"half_size_m":[.6,1.5]},
        {"center_m":[-10.,-2.],"half_size_m":[.7,2.2]},
        {"center_m":[-15.,2.],"half_size_m":[.7,2.2]},
        {"center_m":[-20.,-1.5],"half_size_m":[.7,2.]}])
    (output / "long_route.json").write_text(json.dumps({"name":"25 m arena with four obstacles", "scene":long_scene},indent=2),encoding="utf-8")
    assert np.all(np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-8)
    assert len(stroke_names)==12 and np.allclose(path[0],[-.5218487172304253,0.])
    print(f"Letters: {len(path)} points, {np.linalg.norm(np.diff(path,axis=0),axis=1).sum():.2f} m, {stroke_names}")


if __name__ == "__main__":
    make()
