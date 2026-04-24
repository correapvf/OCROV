import os
import json
import cv2
import pandas as pd
from tqdm.auto import tqdm
from paddleocr import PaddleOCR


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# =========================
# LIMPEZA DE TEXTO
# =========================

def clean_text(text):

    if not text:
        return ""

    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "D": "0",
        "B": "8"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.strip()


# =========================
# ESCOLHER MELHOR OCR
# =========================

def best_ocr_of_frames(results):

    best_text = ""
    best_conf = 0

    for result in results:

        if not result:
            continue

        texts = result['rec_texts']
        confidences = result['rec_scores']

        if len(confidences) == 0:
            continue

        combined_text = " ".join(texts)
        mean_conf = sum(confidences) / len(confidences)

        if mean_conf > best_conf:
            best_conf = mean_conf
            best_text = combined_text

    return clean_text(best_text)


# =========================
# PROCESSAR IMAGENS
# =========================

def process_images(image_folder, bboxes, output_csv, ocr):

    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    image_files.sort()

    rows = []

    for img_name in tqdm(image_files):

        img_path = os.path.join(image_folder, img_name)

        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Erro lendo imagem: {img_name}")
            continue

        row = {
            "image": img_name
        }

        roi_frames = []
        roi_order = []

        for var, (x1, y1, x2, y2) in bboxes.items():

            roi = frame[y1:y2, x1:x2]

            roi_frames.append(roi)
            roi_order.append(var)

        result_flat = ocr.predict(roi_frames)

        for var, result in zip(roi_order, result_flat):

            text = best_ocr_of_frames([result])
            row[var] = text

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"CSV salvo: {output_csv}")


# =========================
# MAIN
# =========================

def parse_args(image_folder, json_path, final_output):

    if json_path is None:
        json_files = [f for f in os.listdir(image_folder) if f.endswith("json")]
        json_path = os.path.join(image_folder, json_files[-1])

    with open(json_path, "r") as f:
        config = json.load(f)

    if image_folder is None:
        image_folder = config["image_folder"]

    bboxes = config["bboxes_pixels"]

    if final_output is None:
        final_output = os.path.splitext(os.path.basename(json_path))[0] + ".csv"

    main(image_folder, bboxes, final_output)


def main(image_folder, bboxes, final_output):

    print("Inicializando PaddleOCR...")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en",
        enable_mkldnn=False
    )

    print()

    process_images(image_folder, bboxes, final_output, ocr)

    print("Processo concluído com sucesso.")


# =========================
# CLI
# =========================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="ROV OCR Image Processor"
    )

    parser.add_argument("path",
                        help="Folder containing images")

    parser.add_argument("--json",
                        help="JSON com bounding boxes")

    parser.add_argument("--output",
                        help="CSV final")

    args = parser.parse_args()

    parse_args(args.path, args.json, args.output)