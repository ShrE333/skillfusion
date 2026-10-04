import json, httpx
from skillfusion.catalog import Catalog, CatalogError
from skillfusion.agent.llm import NemotronClient, extract_text, parse_json
from skillfusion.agent.planner import Planner, rule_based, validate_action
from skillfusion.agent import codegen

cat = Catalog.load()

def mock_client(content=None, reasoning=None, status=200):
    def handler(req):
        msg = {"role": "assistant", "content": content}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        return httpx.Response(status, json={"choices": [{"message": msg}]})
    return NemotronClient(api_key="k", transport=httpx.MockTransport(handler))

def test_reasoning_fallback_and_think_strip():
    assert extract_text({"choices": [{"message": {"content": "", "reasoning_content": "hi"}}]}) == "hi"
    assert extract_text({"choices": [{"message": {"content": "<think>x</think>ok"}}]}) == "ok"

def test_parse_json_variants():
    assert parse_json('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1,2]} done') == {"a": [1, 2]}

def test_spawn_rules():
    a = validate_action(cat, rule_based(cat, "spawn the franka in the warehouse environment and place it beside the shelf"))
    assert (a["robot"], a["env"], a["anchor"], a["relation"]) == ("franka", "warehouse", "shelf", "beside")

def test_task_rules():
    a = validate_action(cat, rule_based(cat, "make the franka pick and place the blocks in front of it from point a to b"))
    assert a["skill"] == "pick_place" and a["from"] == "A" and a["to"] == "B"
    assert set(a["objects"]) == {"red_block", "blue_block", "green_block"}

def test_llm_hallucination_falls_back_to_rules():
    llm = mock_client(json.dumps({"action": "spawn", "robot": "ur10", "env": "warehouse"}))
    a = Planner(cat, llm).parse("spawn the franka in the kitchen next to the fridge")
    assert a["source"] == "rules" and a["env"] == "kitchen" and a["anchor"] == "fridge"

def test_llm_valid_action_used():
    llm = mock_client(json.dumps({"action": "spawn", "robot": "Franka Panda", "env": "lab", "anchor": "desk", "relation": "next to"}))
    a = Planner(cat, llm).parse("put a panda by the desk in the lab")
    assert a["source"] == "nemotron" and a["anchor"] == "workbench" and a["relation"] == "beside"

def test_llm_down_uses_rules():
    a = Planner(cat, mock_client(status=500)).parse("split it into steps")
    assert a["action"] == "decompose"

def test_decompose_rules_only_ready_catalog_skills():
    steps = Planner(cat).decompose({"objects": ["red_block", "blue_block"], "from": "A", "to": "B"})
    assert [s["skill"] for s in steps] == ["reach", "pick_place"] * 2 + ["reach"]
    assert all(cat.raw["skills"][s["skill"]]["status"] == "ready" for s in steps)

def test_decompose_llm_rejects_stretch_skill():
    bad = mock_client(json.dumps({"steps": [{"skill": "throw", "object": "box", "goal": "B"}]}))
    steps = Planner(cat, bad).decompose({"objects": ["red_block"], "to": "B"})
    assert all(s["skill"] != "throw" for s in steps)

def test_code_gate():
    ok = "for o in scene.props():\n    robot.pick_place(o, 'B')\nrobot.reach('home')\n"
    codegen.check_code(ok)
    for bad in ("import os", "open('x')", "robot.__class__", "eval('1')", "import subprocess",
                "x = getattr(robot,'a')", "try:\n  pass\nexcept:\n  pass"):
        try:
            codegen.check_code(bad)
        except (codegen.UnsafeCode, SyntaxError):
            continue
        raise AssertionError(bad)

def test_run_code_calls_api():
    calls = []
    class R:
        def pick_place(self, o, g): calls.append((o, g))
        def reach(self, g): calls.append(("reach", g))
    class S:
        def props(self): return ["red_block", "blue_block"]
    codegen.run_code("for o in scene.props():\n    robot.pick_place(o,'B')\nrobot.reach('home')", {"robot": R(), "scene": S()})
    assert calls == [("red_block", "B"), ("blue_block", "B"), ("reach", "home")]

def test_docindex():
    d = codegen.DocIndex(); d.add_text("a", "camera rgb output tiled camera newton warp"); d.add_text("b", "quaternion xyzw order")
    assert d.search("quaternion order")[0][0] == "b"
