import json


EVALUATION_PATH = "src/data/real_evaluation_queries.json"


def load_queries():
    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_queries(queries):
    with open(
        EVALUATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            queries,
            file,
            indent=2,
            ensure_ascii=False,
        )


def parse_selection(selected, ids):
    if not selected.strip():
        return []

    indexes = [
        int(value.strip()) - 1
        for value
        in selected.split(",")
    ]

    for index in indexes:
        if index < 0 or index >= len(ids):
            raise ValueError(
                f"{index + 1} is out of range (1-{len(ids)})"
            )

    return [ids[index] for index in indexes]
