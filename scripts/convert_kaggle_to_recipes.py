#!/usr/bin/env python3
"""
Convert a Kaggle recipe + reviews CSV dataset into many JSON files
matching the existing pipeline's `data/recipe_*.json` format.

Usage:
  uv run python scripts/convert_kaggle_to_recipes.py \
    --input-dir data-external \
    --recipes-file RAW_recipes.csv \
    --reviews-file RAW_interactions.csv \
    --output-dir data/generated \
    --sample-size 100 \
    --min-mod-reviews 1
"""

import argparse
import json
import os
import random
from datetime import datetime
import re

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("Please install pandas (pip install pandas) before running this script")

from slugify import slugify

MOD_KEYWORDS = re.compile(
    r"\b(add(ed|ing)?|omit(ted|ting)?|skip(ped)?|remove(d)?|replace(d)?|substitut(e|ed|ing)?|increase(d)?|decrease(d)?|reduce(d)?|half|halve(d)?|more|less|extra|chill|refrigerat|bake|cook|preheat)\b",
    flags=re.I,
)


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_modification(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(MOD_KEYWORDS.search(text))


def normalize_ingredients(raw):
    if pd.isna(raw):
        return []
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    if (s.startswith("[") and s.endswith("]")) or ("','" in s) or ('", "' in s):
        try:
            inner = re.sub(r'^\[|\]$', '', s)
            parts = re.split(r"'\s*,\s*'|\"\s*,\s*\"", inner)
            parts = [p.strip().strip('"\'' ) for p in parts if p.strip()]
            return parts
        except Exception:
            pass
    if "\n" in s:
        return [line.strip() for line in s.splitlines() if line.strip()]
    if "," in s and len(s) < 1000:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
    return [s] if s else []


def normalize_instructions(raw):
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if "\n" in s:
        return [line.strip() for line in s.splitlines() if line.strip()]
    if re.search(r"\d\.", s):
        parts = re.split(r"\d+\.\s*", s)
        parts = [p.strip() for p in parts if p.strip()]
        return parts
    if ". " in s and len(s) > 200:
        parts = [p.strip() for p in s.split(". ") if p.strip()]
        return parts
    return [s] if s else []


def main(args):
    in_dir = args.input_dir
    recipes_file = os.path.join(in_dir, args.recipes_file)
    reviews_file = os.path.join(in_dir, args.reviews_file)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Reading recipes CSV:", recipes_file)
    recipes_df = pd.read_csv(recipes_file)
    print("Reading reviews CSV:", reviews_file)
    reviews_df = pd.read_csv(reviews_file)

    recipes_df.columns = [c.strip() for c in recipes_df.columns]
    reviews_df.columns = [c.strip() for c in reviews_df.columns]

    id_col = find_column(recipes_df, ["id", "recipe_id", "recipeId", "recipeID", "recipe_id"])
    title_col = find_column(recipes_df, ["title", "name", "recipe_name", "headline"])
    ingredients_col = find_column(recipes_df, ["ingredients", "ingredients_text", "ingredients_list"])
    instr_col = find_column(recipes_df, ["instructions", "directions", "steps", "method"])
    desc_col = find_column(recipes_df, ["description", "summary", "desc"])
    rating_col = find_column(recipes_df, ["rating", "rating_value", "average_rating", "ratingScore"])
    rating_count_col = find_column(recipes_df, ["rating_count", "reviews_count", "num_reviews", "review_count"])
    servings_col = find_column(recipes_df, ["servings", "yield", "servings_count"])
    author_col = find_column(recipes_df, ["author", "author_name", "contributor"])
    categories_col = find_column(recipes_df, ["tags", "categories", "cuisine", "course"])

    print("Detected columns:", {
        "id": id_col, "title": title_col, "ingredients": ingredients_col,
        "instructions": instr_col, "desc": desc_col, "rating": rating_col,
        "rating_count": rating_count_col
    })

    review_recipe_col = find_column(reviews_df, ["recipe_id", "recipeId", "id", "recipe"])
    review_text_col = find_column(reviews_df, ["review", "review_text", "text", "comments", "content"])
    review_rating_col = find_column(reviews_df, ["rating", "stars", "rating_value"])

    if not id_col or not title_col:
        raise SystemExit("Could not detect recipe id/title columns. Please open your CSV and re-run with correct column names.")

    if not review_recipe_col or not review_text_col:
        raise SystemExit("Could not detect review linking or review text column. Please check your reviews CSV.")

    reviews_df = reviews_df[[review_recipe_col, review_text_col] + ([review_rating_col] if review_rating_col else [])]
    reviews_df = reviews_df.rename(columns={review_recipe_col: "recipe_id", review_text_col: "text"})
    if review_rating_col:
        reviews_df = reviews_df.rename(columns={review_rating_col: "rating"})
    else:
        reviews_df["rating"] = None

    if "rating" in reviews_df.columns:
      grouped = reviews_df.groupby("recipe_id")[["text", "rating"]]
    else:
      grouped = reviews_df.groupby("recipe_id")[["text"]]

    reviews_map = {}
    for rid, group in reviews_df.groupby("recipe_id"):
        arr = []
        for _, row in group.iterrows():
            txt = row["text"]
            r = row.get("rating", None) if "rating" in row else None
            has_mod = detect_modification(txt)
            arr.append({"text": str(txt), "rating": r, "has_modification": bool(has_mod)})
        reviews_map[str(rid)] = arr

    rows = recipes_df.to_dict(orient="records")
    print("Total recipes available:", len(rows))
    selected = []
    for r in rows:
        rid = str(r.get(id_col))
        revs = reviews_map.get(rid, [])
        mod_count = sum(1 for rv in revs if rv.get("has_modification"))
        if args.min_mod_reviews <= mod_count:
            selected.append((r, revs))

    print(f"Found {len(selected)} recipes that have >= {args.min_mod_reviews} mod-flagged reviews.")

    if len(selected) == 0:
        print("No recipes meet the min_mod_reviews threshold; lowering threshold to include any recipe with reviews.")
        selected = [(r, reviews_map.get(str(r.get(id_col)), [])) for r in rows]

    sample_size = args.sample_size or len(selected)
    sample_size = min(sample_size, len(selected))
    chosen = random.sample(selected, sample_size) if sample_size < len(selected) else selected[:sample_size]

    print(f"Generating {len(chosen)} recipe JSON files to {out_dir}")

    now_iso = datetime.utcnow().isoformat()
    for row, revs in chosen:
        rid = str(row.get(id_col))
        title = str(row.get(title_col)) if title_col else f"recipe-{rid}"
        slug = slugify(title)[:60]
        filename = f"recipe_{rid}_{slug}.json"
        ingredients = normalize_ingredients(row.get(ingredients_col)) if ingredients_col else []
        instructions = normalize_instructions(row.get(instr_col)) if instr_col else []
        desc = row.get(desc_col) if desc_col else ""
        rating_val = str(row.get(rating_col)) if rating_col and row.get(rating_col) is not None else None
        rating_count = str(row.get(rating_count_col)) if rating_count_col and row.get(rating_count_col) is not None else None
        author = row.get(author_col) if author_col else None
        categories = []
        if categories_col and row.get(categories_col):
            categories = normalize_ingredients(row.get(categories_col))

        out_obj = {
            "url": row.get("url", ""),
            "scraped_at": now_iso,
            "recipe_id": rid,
            "title": title,
            "description": desc if desc else None,
            "rating": {"value": rating_val, "count": rating_count},
            "preptime": None,
            "cooktime": None,
            "totaltime": None,
            "servings": str(row.get(servings_col)) if servings_col else None,
            "ingredients": ingredients,
            "instructions": instructions,
            "nutrition": None,
            "author": author,
            "categories": categories,
            "featured_tweaks": [],
            "reviews": revs
        }

        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(out_obj, fw, ensure_ascii=False, indent=2)

    print("Done. Generated files:", len(os.listdir(out_dir)))
    print("Tip: move or symlink the generated JSON files into `data/` if you want the pipeline to pick them up directly.")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Kaggle recipes + reviews into JSON recipe files")
    parser.add_argument("--input-dir", required=True, help="Directory where RAW_recipes.csv and RAW_interactions.csv are stored")
    parser.add_argument("--recipes-file", required=True, help="Recipes CSV filename (e.g. RAW_recipes.csv)")
    parser.add_argument("--reviews-file", required=True, help="Reviews CSV filename (e.g. RAW_interactions.csv)")
    parser.add_argument("--output-dir", required=True, help="Where to write JSON recipe files")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of recipes to randomly sample")
    parser.add_argument("--min-mod-reviews", type=int, default=0, help="Minimum modification-flagged reviews required")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
