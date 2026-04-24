import os
import json
import pandas as pd
from tqdm.auto import tqdm
from paddleocr import PaddleOCR
# from decord import VideoReader
import cv2


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


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

    text = text.strip()

    return text


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

        # for line in result:
        #     texts.append(line['rec_texts'])
        #     confidences.append(line['rec_scores'])

        combined_text = " ".join(texts)
        mean_conf = sum(confidences) / len(confidences)

        if mean_conf > best_conf:
            best_conf = mean_conf
            best_text = combined_text

    best_text = clean_text(best_text)

    return best_text


# =========================
# PROCESSAR VÍDEO
# =========================

def process_video(video_path, bboxes, output_csv, ocr):

    print(f"Processando: {video_path}")

    # vr = VideoReader(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    # fps = round(vr.get_avg_fps())
    # vr.seek(0)

    if fps == 0:
        print("Erro lendo FPS.")
        return

    # total_frames = len(vr)-3
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-3

    rows = []
    frame_index = 0

    pbar = tqdm(total=total_frames//fps)
    while frame_index < total_frames:

        roi_frames_dict = {}

        # get one frame, skip, get third frame
        # frame = vr.ne xt().asnumpy()
        success, frame = cap.read()
        if not success:
            break
        for var, (x1, y1, x2, y2) in bboxes.items():
            roi = frame[y1:y2, x1:x2]
            # roi = cv2.copyMakeBorder(roi, top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            if var not in roi_frames_dict:
                roi_frames_dict[var] = []
            roi_frames_dict[var].append(roi)

        # vr.skip_frames(1)
        success = cap.grab() # skip one frame
            
        # frame = vr.next().asnumpy()
        success, frame = cap.read()
        if not success:
            break
        for var, (x1, y1, x2, y2) in bboxes.items():
            roi = frame[y1:y2, x1:x2]
            # roi = cv2.copyMakeBorder(roi, top=5, bottom=5, left=5, right=5, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if var not in roi_frames_dict:
                roi_frames_dict[var] = []
            roi_frames_dict[var].append(roi)

        timestamp_sec = frame_index / fps
        row = {
            "video": os.path.basename(video_path),
            "timestamp_sec": timestamp_sec
        }

        roi_frames_list = [item for sublist in roi_frames_dict.values() for item in sublist]

        result_flat = ocr.predict(roi_frames_list)

        result_list = [result_flat[i:i+2] for i in range(0, len(result_flat), 2)]
        result_dict = dict(zip(roi_frames_dict.keys(), result_list))

        for var, results in result_dict.items():

            text = best_ocr_of_frames(results)
            row[var] = text

        rows.append(row)

        # vr.skip_frames(fps - 3)
        for _ in range(fps - 3):
            cap.grab()
        frame_index += fps
        pbar.update()

    pbar.close()
    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"CSV salvo: {output_csv}")


# =========================
# MAIN
# =========================

def parse_args(video_folder, json_path, final_output):
    if json_path is None:
        json_path = [f for f in os.listdir(video_folder) if f.endswith("json")]
        json_path = os.path.join(video_folder, json_path[-1])

    with open(json_path, "r") as f:
        config = json.load(f)

    if video_folder is None:
        video_folder = config["video_folder"]

    bboxes = config["bboxes_pixels"]

    if final_output is None:
        final_output = os.path.splitext(os.path.basename(json_path))[0] + '.csv'
    
    main(video_folder, bboxes, final_output)

def main(video_folder, bboxes, final_output):

    video_files = [
        f for f in os.listdir(video_folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    print("Inicializando PaddleOCR...")
    ocr = PaddleOCR(use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en", #"pt"
        enable_mkldnn=False # prevent crash in the OCR, bug to be fixed
    )

    print()

    temp_csvs = []

    for video in tqdm(video_files):

        video_path = os.path.join(video_folder, video)
        temp_csv = os.path.join(
            video_folder,
            f"{os.path.splitext(video)[0]}_ocr_temp.csv"
        )

        # Pular se já existir
        if os.path.exists(temp_csv):
            print(f"Pulando (já processado): {video}")
            temp_csvs.append(temp_csv)
            continue

        process_video(video_path, bboxes, temp_csv, ocr)
        temp_csvs.append(temp_csv)

    # =====================
    # Juntar CSVs
    # =====================

    print("\nUnindo CSVs finais...")

    dfs = [pd.read_csv(csv) for csv in temp_csvs]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(final_output, index=False, encoding='utf-8-sig')

    print(f"CSV final salvo: {final_output}")

    # =====================
    # Deletar temporários
    # =====================

    print("Removendo temporários...")
    for csv in temp_csvs:
        os.remove(csv)

    print("Processo concluído com sucesso.")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="ROV OCR Resumable Processor"
    )

    parser.add_argument("path",
                        help="Folder containg videos")

    parser.add_argument("--json",
                        help="JSON com bounding boxes")

    parser.add_argument("--output",
                        help="CSV final consolidado")

    args = parser.parse_args()

    parse_args(args.path, args.json, args.output)
    