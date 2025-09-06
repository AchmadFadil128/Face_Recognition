#!/usr/bin/env python3
"""
Build embeddings from folder structure under `database/` and save to a pickle
that is compatible with face-insight.py.

Usage:
  python build_embeddings.py --ref database --embeddings face_embeddings.pkl

Requirements:
  pip install insightface onnxruntime opencv-python numpy
"""

import os
import cv2
import pickle
import argparse
import numpy as np
from insightface.app import FaceAnalysis


def save_embeddings(embeddings_file, known_embeddings):
    try:
        with open(embeddings_file, 'wb') as f:
            pickle.dump(known_embeddings, f)
        print(f"[i] Saved embeddings to {embeddings_file}")
    except Exception as e:
        print(f"[!] Failed to save embeddings: {str(e)}")


def index_references(ref_dir, app, embeddings_file, min_face_size=20):
    """
    Scan ref_dir for subfolders; for each <ref_dir>/<person>/*.jpg create embedding(s).
    Output format: dict[str, list[np.ndarray (512,)]] compatible with face-insight.py
    """
    print("[i] Indexing reference images from:", ref_dir)

    known_embeddings = {}

    if not os.path.isdir(ref_dir):
        print(f"[!] Reference folder not found: {ref_dir}")
        return known_embeddings

    people = sorted(os.listdir(ref_dir))
    for person in people:
        person_dir = os.path.join(ref_dir, person)
        if not os.path.isdir(person_dir):
            continue

        # collect candidate images (jpg/jpeg/png)
        image_files = [fn for fn in os.listdir(person_dir) if fn.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_files:
            print(f"  - skip {person}: no image files")
            continue

        person_embs = []
        for fn in image_files:
            path = os.path.join(person_dir, fn)
            img = cv2.imread(path)
            if img is None:
                print(f"  - skip {person}: cannot read {fn}")
                continue

            faces = app.get(img)
            if len(faces) == 0:
                print(f"  - skip {person}/{fn}: no face detected")
                continue

            # Choose largest face (in case group photo)
            areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
            idx = int(np.argmax(areas))
            face = faces[idx]

            # filter tiny faces (noise)
            x1, y1, x2, y2 = face.bbox.astype(int)
            bw, bh = (x2 - x1), (y2 - y1)
            if bw < min_face_size or bh < min_face_size:
                print(f"  - skip {person}/{fn}: tiny face {bw}x{bh}")
                continue

            emb = face.normed_embedding
            if emb is None:
                print(f"  - skip {person}/{fn}: no embedding")
                continue

            person_embs.append(emb)
            print(f"  - added {person}: {fn}")

        if person_embs:
            known_embeddings[person] = person_embs
            print(f"[i] {person}: {len(person_embs)} embeddings")
        else:
            print(f"[i] {person}: no valid embeddings")

    save_embeddings(embeddings_file, known_embeddings)
    return known_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, default="database", help="reference images folder")
    parser.add_argument("--embeddings", type=str, default="face_embeddings.pkl", help="pickle embeddings file path")
    parser.add_argument("--det_w", type=int, default=800, help="detector width (insightface det_size)")
    parser.add_argument("--det_h", type=int, default=800, help="detector height (insightface det_size)")
    parser.add_argument("--min_face", type=int, default=20, help="ignore faces smaller than this (px)")
    args = parser.parse_args()

    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(args.det_w, args.det_h))
    print(f"[i] insightface ready. det_size=({args.det_w},{args.det_h})")

    index_references(args.ref, app, args.embeddings, min_face_size=args.min_face)


if __name__ == "__main__":
    main()


