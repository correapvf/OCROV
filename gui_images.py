import os
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class OverlayBBoxApp:

    def __init__(self, root):
        self.root = root
        self.root.title("ROV Overlay OCR (Images)")

        self.image_folder = None
        self.image_files = []
        self.current_image_index = 0

        self.original_frame = None
        self.tk_image = None

        self.scale_x = 1
        self.scale_y = 1

        self.bboxes = {}
        self.bbox_canvas_ids = {}

        self.current_label = None

        self.start_x = None
        self.start_y = None
        self.temp_rect_id = None

        self.variable_colors = {
            "latitude": "#e74c3c",
            "longitude": "#3498db",
            "profundidade": "#2ecc71",
            "data": "#f1c40f",
            "hora": "#9b59b6",
            "heading": "#1abc9c",
            "altitude": "#e67e22",
            "dive": "#34495e"
        }

        self.setup_ui()

        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

    # ================= UI =================

    def setup_ui(self):

        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5)

        tk.Button(top_frame, text="Selecionar Pasta",
                  command=self.select_folder).pack(side=tk.LEFT)

        tk.Button(top_frame, text="Carregar JSON",
                  command=self.load_json).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame,
                  text="Salvar BBoxes",
                  command=self.save_bboxes).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame,
                  text="Executar OCR",
                  command=self.ocr).pack(side=tk.LEFT)

        tk.Button(top_frame,
                  text="Dicas",
                  command=self.tips).pack(side=tk.LEFT, padx=5)

        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=5, padx=5)

        tk.Button(nav_frame, text="Primeiro",
                  command=self.first_image).pack(side=tk.LEFT)

        tk.Button(nav_frame, text="<< Anterior",
                  command=self.prev_image).pack(side=tk.LEFT, padx=5)

        tk.Button(nav_frame, text="Próximo >>",
                  command=self.next_image).pack(side=tk.LEFT, padx=5)

        tk.Button(nav_frame, text="Último",
                  command=self.last_image).pack(side=tk.LEFT)
        
        self.image_label = tk.Label(nav_frame,
                                    text="Nenhuma imagem carregada")
        self.image_label.pack(side=tk.LEFT, padx=10)

        var_frame = tk.Frame(self.root)
        var_frame.pack(pady=5)

        self.variable_buttons = {}

        for var, color in self.variable_colors.items():
            btn = tk.Button(var_frame,
                            text=var,
                            bg=color,
                            fg="white",
                            command=lambda v=var: self.set_current_label(v))
            btn.pack(side=tk.LEFT, padx=3)
            self.variable_buttons[var] = btn

        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    # ================= Imagens =================

    def select_folder(self):

        folder = filedialog.askdirectory()
        if not folder:
            return

        self.image_folder = folder

        self.image_files = sorted(
            [f for f in os.listdir(folder)
             if f.lower().endswith(IMAGE_EXTENSIONS)]
        )

        if not self.image_files:
            messagebox.showerror("Erro", "Nenhuma imagem encontrada.")
            return

        self.current_image_index = 0
        self.load_image()

        json_files = [f for f in os.listdir(folder) if f.endswith(".json")]

        if json_files:
            json_file = json_files[-1]
            self.load_json_core(os.path.join(folder, json_file))

            messagebox.showinfo("Sucesso",
                                f"JSON {json_file} carregado.")

    def load_image(self):

        image_path = os.path.join(
            self.image_folder,
            self.image_files[self.current_image_index]
        )

        frame = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]
        # frame = cv2.imread(image_path)

        if frame is None:
            messagebox.showerror("Erro", "Erro ao ler imagem.")
            return

        self.original_frame = frame

        self.image_label.config(
            text=f"{self.current_image_index + 1}/{len(self.image_files)} - {self.image_files[self.current_image_index]}"
        )

        self.prepare_display_frame()

    def next_image(self):

        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()

    def prev_image(self):

        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def first_image(self):

        self.current_image_index = 0
        self.load_image()

    def last_image(self):

        self.current_image_index = len(self.image_files) - 1
        self.load_image()

    # ================= Frame =================

    def prepare_display_frame(self):

        h, w = self.original_frame.shape[:2]

        scale = min(MAX_DISPLAY_WIDTH / w,
                    MAX_DISPLAY_HEIGHT / h,
                    1)

        new_w = int(w * scale)
        new_h = int(h * scale)

        self.scale_x = w / new_w
        self.scale_y = h / new_h

        resized = cv2.resize(self.original_frame,
                             (new_w, new_h))

        self.display_frame(resized)

    def display_frame(self, frame):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.tk_image = ImageTk.PhotoImage(image)

        self.canvas.config(width=image.width,
                           height=image.height)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0,
                                 anchor=tk.NW,
                                 image=self.tk_image)

        self.redraw_bboxes()

    # ================= BBoxes =================

    def sanitize_bbox(self, x1, y1, x2, y2):

        h, w = self.original_frame.shape[:2]

        # garantir ordem correta
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # limitar dentro da imagem
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))

        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        return [x1, y1, x2, y2]

    def set_current_label(self, label):
        self.current_label = label

        for var, btn in self.variable_buttons.items():
            btn.config(relief=tk.SUNKEN if var == label else tk.RAISED)

    def on_mouse_down(self, event):

        if not self.current_label:
            messagebox.showwarning("Aviso",
                                   "Selecione uma variável.")
            return

        self.start_x = event.x
        self.start_y = event.y

        color = self.variable_colors[self.current_label]

        self.temp_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x, self.start_y,
            outline=color, width=2
        )

    def on_mouse_drag(self, event):

        if self.temp_rect_id:
            self.canvas.coords(self.temp_rect_id,
                               self.start_x,
                               self.start_y,
                               event.x,
                               event.y)

    def on_mouse_up(self, event):

        if not self.temp_rect_id:
            return

        x1 = int(self.start_x * self.scale_x)
        y1 = int(self.start_y * self.scale_y)
        x2 = int(event.x * self.scale_x)
        y2 = int(event.y * self.scale_y)

        if self.current_label in self.bbox_canvas_ids:
            self.canvas.delete(self.bbox_canvas_ids[self.current_label])

        self.bboxes[self.current_label] = self.sanitize_bbox(x1, y1, x2, y2)
        self.bbox_canvas_ids[self.current_label] = self.temp_rect_id
        self.temp_rect_id = None

    def redraw_bboxes(self):

        for label, (x1, y1, x2, y2) in self.bboxes.items():

            color = self.variable_colors[label]

            x1d = x1 / self.scale_x
            y1d = y1 / self.scale_y
            x2d = x2 / self.scale_x
            y2d = y2 / self.scale_y

            rect_id = self.canvas.create_rectangle(
                x1d, y1d, x2d, y2d,
                outline=color, width=2
            )

            self.bbox_canvas_ids[label] = rect_id

    # ================= JSON =================

    def save_bboxes(self):

        if not self.bboxes:
            messagebox.showwarning("Aviso",
                                   "Nenhuma bbox definida.")
            return

        h, w = self.original_frame.shape[:2]

        data = {
            "image_folder": self.image_folder,
            "resolution": {
                "width": w,
                "height": h
            },
            "bboxes_pixels": self.bboxes
        }

        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=os.path.basename(self.image_folder)
        )

        if not save_path:
            return

        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

        messagebox.showinfo("Sucesso",
                            "Bounding boxes salvas.")

    def load_json_core(self, path):

        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.bboxes = data.get("bboxes_pixels", {})
        self.bbox_canvas_ids = {}

        self.prepare_display_frame()

    def load_json(self):

        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")]
        )

        self.load_json_core(path)

    # ================= OCR =================

    def ocr(self):

        final_output = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=os.path.basename(self.image_folder)
        )

        from OCROV_images import main

        main(self.image_folder, self.bboxes, final_output)

    def tips(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("Dicas")

        tk.Label(
            new_window,
            text="Selecionar bounding boxes com um pouco de espaço entre o texto e as bordas.\n"
                 "Não há problema se a bounding boxes se sobreporem parcialmente"
        ).pack(padx=20, pady=20)


if __name__ == "__main__":

    root = tk.Tk()

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    MAX_DISPLAY_WIDTH = max(1100, root.winfo_screenwidth() * 0.8)
    MAX_DISPLAY_HEIGHT = max(750, root.winfo_screenheight() * 0.7)

    app = OverlayBBoxApp(root)

    root.mainloop()