import json
import sqlite3
import tqdm
import os

SRC = "data/conceptnet_build/conceptnet.db"
DST = "data/conceptnet_lite.json"

con = sqlite3.connect(SRC)
cur = con.cursor()

# The query from the project report was updated to use the correct table and column names,
# and to extract the weight from the JSON 'etc' column.
qry = """
SELECT  '/c/' || start_lang.name || '/' || start_label.text,
        rel.name,
        '/c/' || end_lang.name || '/' || end_label.text,
        json_extract(edge.etc, '$.weight')
FROM edge
JOIN relation   AS rel           ON edge.relation_id = rel.id
JOIN concept    AS start_concept ON edge.start_id    = start_concept.id
JOIN label      AS start_label   ON start_concept.label_id = start_label.id
JOIN language   AS start_lang    ON start_label.language_id = start_lang.id
JOIN concept    AS end_concept   ON edge.end_id      = end_concept.id
JOIN label      AS end_label     ON end_concept.label_id = end_label.id
JOIN language   AS end_lang      ON end_label.language_id = end_lang.id;
"""

with open(DST, "w", encoding="utf-8") as f:
    # The project report's query returns tuples, so we'll zip with keys.
    keys = ("head", "predicate", "tail", "weight")
    for row in tqdm.tqdm(cur.execute(qry), total=34074917):
        rec = dict(zip(keys, row))
        json.dump(rec, f, ensure_ascii=False)
        f.write("\n")

con.close()

print(f"Successfully exported {DST}")

