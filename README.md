# 🚀 Basic Workflow

This project follows a simple four-step workflow.

------------------------------------------------------------------------

## 1️⃣ Update Raw Data

Edit or append paper entries in:

    papers.jsonl

Each paper must include at least:

``` json
{
  "title": "...",
  "authors": "...",
  "year": 2025,
  "l1": "A",
  "l2": "A1"
}
```

If a paper does not fit any existing category:

``` json
{
  "l1": "Others",
  "l2": "Others"
}
```

(See governance rules below for how to create new categories.)

------------------------------------------------------------------------

## 2️⃣ Push Changes

After editing `papers.jsonl`, commit and push:

``` bash
git add papers.jsonl
git commit -m "Update papers data"
git push
```

------------------------------------------------------------------------

## 3️⃣ Build Tree Data

Generate the hierarchical tree structure:

``` bash
python build_tree.py
```

This creates:

    tree.json

`tree.json` is the structured dataset used for visualization.

------------------------------------------------------------------------

## 4️⃣ Run Streamlit App

Launch the interactive research tree:

``` bash
streamlit run app.py
```

The application will open automatically in your browser.

------------------------------------------------------------------------

# 🔁 Category Governance Rule (Important)

If a paper does **not clearly fit** into any existing L1/L2 category:

    L1: Others
    L2: Others

If multiple papers accumulate in `Others`, you must:

1.  Propose a new L2 (or L1 if necessary)\
2.  Update the category list in `README.md`\
3.  Reclassify affected papers\
4.  Rebuild the tree

This system is intentionally **extensible**.\
The category structure is not fixed and must evolve with new research
themes.
