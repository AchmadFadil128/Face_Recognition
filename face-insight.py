#!/usr/bin/env python3
"""
face_recognition_realtime.py

Requirements:
    pip install insightface onnxruntime opencv-python numpy

Usage:
    python face_recognition_realtime.py --ref database --embeddings face_embeddings.pkl
"""

import os
import cv2
import pickle
import argparse
import requests
import time
import numpy as np
from insightface.app import FaceAnalysis

# ---------------------------
# Attendance POST config
# ---------------------------
ATTENDANCE_ENDPOINT = "http://localhost:3000/api/attendance/mark"

# Manual name → studentNumber mapping
NAME_TO_STUDENT = {
    "Achmad": "SW001",
    "Adel": "SW002",
    "Aziz": "SW003",
    "Ibrahim": "SW004",
}

# Per-run dedupe to avoid multiple posts for same student
_SENT_STUDENT_NUMBERS = set()

def send_attendance(label):
    """Send attendance for recognized label once per run."""
    student_number = NAME_TO_STUDENT.get(label)
    if not student_number:
        return
    if student_number in _SENT_STUDENT_NUMBERS:
        return
    payload = {"studentNumber": student_number, "status": "PRESENT"}
    try:
        requests.post(ATTENDANCE_ENDPOINT, json=payload)
        _SENT_STUDENT_NUMBERS.add(student_number)
    except Exception:
        # per requirement: no error handling; ignore failures for now
        pass

# ---------------------------
# Utility: Pickle storage for embeddings
# ---------------------------
def load_embeddings(embeddings_file):
    """Load embeddings from pickle file"""
    if os.path.exists(embeddings_file):
        try:
            print(f"[i] Loading existing embeddings from {embeddings_file}")
            with open(embeddings_file, 'rb') as f:
                known_embeddings = pickle.load(f)
            print(f"[i] Loaded {len(known_embeddings)} persons with embeddings")
            return known_embeddings
        except Exception as e:
            print(f"[!] Failed to load embeddings: {str(e)}")
            print("[i] Will create new embeddings...")
            return {}
    else:
        print(f"[i] No existing embeddings file found: {embeddings_file}")
        print("[i] Will create new embeddings...")
        return {}

def save_embeddings(embeddings_file, known_embeddings):
    """Save embeddings to pickle file"""
    try:
        with open(embeddings_file, 'wb') as f:
            pickle.dump(known_embeddings, f)
        print(f"[i] Saved embeddings to {embeddings_file}")
    except Exception as e:
        print(f"[!] Failed to save embeddings: {str(e)}")

def embeddings_to_arrays(known_embeddings):
    """Convert pickle embeddings format to arrays for matching"""
    names = []
    embs = []
    paths = []
    
    for person_name, embeddings_list in known_embeddings.items():
        for i, embedding in enumerate(embeddings_list):
            names.append(person_name)
            embs.append(embedding)
            # Create a placeholder path since pickle doesn't store paths
            paths.append(f"{person_name}_{i}")
    
    if len(embs) > 0:
        embs_arr = np.vstack(embs)  # shape: (N, D)
    else:
        embs_arr = np.zeros((0, 512), dtype=np.float32)  # placeholder
    
    return names, embs_arr, paths

# ---------------------------
# Index reference images into pickle
# ---------------------------
def index_references(ref_dir, app, embeddings_file, min_face_size=20):
    """
    Scan ref_dir for subfolders; for each <ref_dir>/<person>/<person>.jpg create embedding.
    """
    print("[i] Indexing reference images from:", ref_dir)
    
    # Load existing embeddings
    known_embeddings = load_embeddings(embeddings_file)
    
    for person in os.listdir(ref_dir):
        person_dir = os.path.join(ref_dir, person)
        if not os.path.isdir(person_dir):
            continue
            
        # prefer file named <person>.* inside folder
        candidate = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(person_dir, f"{person}{ext}")
            if os.path.isfile(p):
                candidate = p
                break
        if candidate is None:
            # fallback: take first image file in folder
            for fn in os.listdir(person_dir):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    candidate = os.path.join(person_dir, fn)
                    break
        if candidate is None:
            print(f"  - skip {person}: no image found")
            continue

        img = cv2.imread(candidate)
        if img is None:
            print(f"  - skip {person}: cannot read {candidate}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"  - skip {person}: no face detected in {candidate}")
            continue

        # Choose largest face (in case group photo)
        areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
        idx = int(np.argmax(areas))
        face = faces[idx]
        emb = face.normed_embedding  # normalized embedding (ArcFace)
        if emb is None:
            print(f"  - skip {person}: no embedding")
            continue

        # Initialize person's embedding list if not exists
        if person not in known_embeddings:
            known_embeddings[person] = []
        
        # Add embedding to person's list
        known_embeddings[person].append(emb)
        print(f"  - saved {person}: {candidate}")

    # Save updated embeddings
    save_embeddings(embeddings_file, known_embeddings)
    return known_embeddings

# ---------------------------
# Matching utility
# ---------------------------
def cosine_similarity_batch(query, gallery):
    """
    query: (D,) numpy
    gallery: (N, D) numpy
    returns: (N,) similarities in [-1,1] (higher is better)
    """
    if gallery.shape[0] == 0:
        return np.array([])
    q = query.astype(np.float32)
    g = gallery.astype(np.float32)
    qnorm = np.linalg.norm(q) + 1e-10
    gnorms = np.linalg.norm(g, axis=1) + 1e-10
    sims = (g @ q) / (gnorms * qnorm)
    return sims

# ---------------------------
# Main realtime loop
# ---------------------------
def main(args):
    # Initialize face app (RetinaFace detector + ArcFace embeddings)
    # Use CPU only: providers argument ensures onnxruntime CPU provider is used.
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    # det_size controls detector input size; smaller is faster but might miss tiny faces.
    # For crowded FullHD, det_size=(800,800) is a good starting point for CPU.
    app.prepare(ctx_id=0, det_size=(800, 800))
    print("[i] insightface ready. Detector det_size=(800,800)")

    # index references (and store embeddings)
    if args.index:
        known_embeddings = index_references(args.ref, app, args.embeddings)
    else:
        known_embeddings = load_embeddings(args.embeddings)
    
    # Convert to arrays for matching
    names, gallery, paths = embeddings_to_arrays(known_embeddings)
    if gallery.shape[0] > 0:
        gallery_norms = np.linalg.norm(gallery, axis=1)
    else:
        gallery_norms = np.array([])

    print(f"[i] loaded {len(names)} enrolled identities")

    # open camera
    cap = cv2.VideoCapture(0)
    # attempt to set to 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("[!] cannot open camera")
        return

    frame_idx = 0
    last_results = []  # store (bbox, name, score)
    t0 = time.time()
    fps_smooth = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] end of stream or cannot read frame")
            break

        h, w = frame.shape[:2]
        frame_idx += 1

        do_process = (frame_idx % args.frame_skip == 0)

        if do_process:
            t1 = time.time()
            # detect faces (app.get handles BGR images)
            faces = app.get(frame)
            results = []
            for f in faces:
                # bbox as ints
                box = f.bbox.astype(int)  # [x1,y1,x2,y2]
                # optional: filter tiny boxes (noise)
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                if bw < args.min_face or bh < args.min_face:
                    continue

                emb = f.normed_embedding
                if emb is None:
                    continue

                # match to gallery
                if gallery.shape[0] > 0:
                    sims = cosine_similarity_batch(emb, gallery)  # higher = better (1.0)
                    best_idx = int(np.argmax(sims))
                    best_score = float(sims[best_idx])
                    if best_score >= args.threshold:
                        label = names[best_idx]
                    else:
                        label = "Unknown"
                else:
                    best_score = 0.0
                    label = "Unknown"

                results.append((box, label, best_score))
                # send attendance for recognized label (not Unknown)
                if label != "Unknown":
                    send_attendance(label)

            last_results = results
            t2 = time.time()
            proc_time = t2 - t1
            fps = 1.0 / proc_time if proc_time > 0 else 0.0
            fps_smooth = fps if fps_smooth is None else (fps_smooth * 0.85 + fps * 0.15)

        # draw last_results
        display = frame.copy()
        for (box, label, score) in last_results:
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} ({score:.3f})" if label != "Unknown" else f"{label}"
            cv2.putText(display, text, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # overlay status
        status = f"Enrolled: {len(names)}  FPS: {fps_smooth:.1f}" if fps_smooth is not None else f"Enrolled: {len(names)}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        cv2.imshow("FaceRecognition (press q to quit, i to re-index refs)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('i'):
            # re-index on demand
            known_embeddings = index_references(args.ref, app, args.embeddings)
            names, gallery, paths = embeddings_to_arrays(known_embeddings)
            if gallery.shape[0] > 0:
                gallery_norms = np.linalg.norm(gallery, axis=1)
            print(f"[i] reloaded {len(names)} identities")

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# CLI & config
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, default="0", help="camera index or video path (default 0)")
    parser.add_argument("--ref", type=str, default="database", help="reference images folder")
    parser.add_argument("--embeddings", type=str, default="face_embeddings.pkl", help="pickle embeddings file path")
    parser.add_argument("--index", action="store_true", help="index reference images at start")
    parser.add_argument("--frame_skip", type=int, default=2, help="process 1 every N frames (speed tuning)")
    parser.add_argument("--threshold", type=float, default=0.45, help="cosine similarity threshold for match (0..1)")
    parser.add_argument("--min_face", type=int, default=30, help="ignore tiny faces smaller than this (px)")
    args = parser.parse_args()
    # if camera is int-like, convert
    try:
        args.camera = int(args.camera)
    except Exception:
        pass

    main(args)
