import numpy as np
from skillfusion.catalog import Catalog, CatalogError
from skillfusion.scene import place, rects_overlap, suggest_relations, normalize_relation

cat = Catalog.load()

def test_aliases():
    assert cat.robot("Franka Panda") == "franka"
    assert cat.robot("the trossen arm") == "wxai"
    assert cat.env("ware house") == "warehouse"
    assert cat.anchor("warehouse", "the rack") == "shelf"
    assert cat.region("point a") == "A" and cat.region("B") == "B"
    assert cat.skill("pick_place") == "pick_place"

def test_unknown_rejected():
    for fn in (lambda: cat.robot("ur10"), lambda: cat.env("moon"), lambda: cat.skill("dance")):
        try:
            fn()
        except CatalogError:
            continue
        raise AssertionError("should reject")

def test_skill_indices_unique():
    idx = [s["index"] for s in cat.raw["skills"].values()]
    assert len(idx) == len(set(idx)) and max(idx) < cat.max_skills

def test_placement_clear_of_anchor_all_envs():
    for env, spec in cat.raw["environments"].items():
        for anchor in [k for k, v in spec["objects"].items() if v["anchor"]]:
            for rel in ("beside", "in_front_of", "behind", "left_of"):
                p = place(cat, env, anchor, rel)
                a = spec["objects"][anchor]
                fp = cat.workcell["footprint"]
                from skillfusion.scene import rot
                fc = np.array(p.base_xy) + rot(p.yaw_deg) @ np.array([(fp["x_min"]+fp["x_max"])/2, (fp["y_min"]+fp["y_max"])/2])
                fh = ((fp["x_max"]-fp["x_min"])/2, (fp["y_max"]-fp["y_min"])/2)
                assert not rects_overlap(fc, fh, p.yaw_deg, a["pose"][:2], (a["size"][0]/2, a["size"][1]/2), a["pose"][2])

def test_robot_faces_anchor_when_in_front():
    p = place(cat, "warehouse", "shelf", "in_front_of")
    # anchor centre must be in front (+x) of the robot in the base frame
    assert p.objects_in_base_frame["shelf"]["xy"][0] > 0
    assert abs(p.objects_in_base_frame["shelf"]["xy"][1]) < 1e-6

def test_beside_is_parallel():
    p = place(cat, "warehouse", "shelf", "beside")
    assert abs(p.yaw_deg - 0.0) < 1e-6
    assert p.objects_in_base_frame["shelf"]["xy"][1] > 0   # shelf on the robot's left when robot is on its right

def test_relation_aliases():
    assert normalize_relation("next to") == "beside"
    assert normalize_relation("in front of") == "in_front_of"

def test_suggestions_nonempty():
    assert suggest_relations(cat, "kitchen", "counter")
