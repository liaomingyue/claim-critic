from ragtruth.own_fill import CID, make_split


def test_split_is_half_per_task_and_stable():
    tasks = {str(i): ("QA" if i % 2 else "Summary") for i in range(100)}
    a, b = make_split(tasks), make_split(dict(reversed(list(tasks.items()))))
    assert a == b
    assert not set(a["fit"]) & set(a["final"]) and len(a["fit"]) == len(a["final"]) == 50
    assert sum(tasks[k] == "QA" for k in a["fit"]) == 25


def test_label_id_parts():
    m = CID.match("s100-a11332-6-S12")
    assert (m.group(1), m.group(2), m.group(3)) == ("a", "11332-6", "12")
