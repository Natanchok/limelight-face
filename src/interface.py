import base64
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont

from protocol import request

# --- Configuration & Paths ---
BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
COMBINED_SCRIPT = BASE_DIR / "combined.py"

HOST = os.environ.get("VERITAG_HOST", "127.0.0.1")
PORT = int(os.environ.get("VERITAG_PORT", "5000"))

if sys.platform == "win32":
    VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
else:
    VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

ALLOWED_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}

# --- Palette ---
BG = "#0d0f14"
CARD = "#161a22"
BORDER = "#242a36"
TEXT = "#f3f5f8"
MUTED = "#7a8497"
ACCENT = "#e63946"
ACCENT_HOVER = "#f04d5a"
ACCENT_PRESS = "#c92f3d"
GHOST = "#222834"
GHOST_HOVER = "#2d3442"
GHOST_PRESS = "#1b202a"
SUCCESS = "#3ddc84"


# --- Server registration ---
def register_user(user_id, name, source_photo):
    """Send a registration request to the server (which owns the database)."""
    source = Path(source_photo)
    if user_id <= 0:
        return False, "ID must be a positive integer."
    if not name:
        return False, "Name is required."
    if not source.is_file():
        return False, "Please choose a photo."
    if source.suffix.lower() not in ALLOWED_PHOTO_SUFFIXES:
        return False, "Please choose a JPG or PNG photo."

    try:
        photo_b64 = base64.b64encode(source.read_bytes()).decode("ascii")
        response = request(HOST, PORT, {
            "action": "register",
            "id": user_id,
            "name": name,
            "filename": source.name,
            "photo": photo_b64,
        })
    except Exception as exc:
        return False, f"Could not reach server: {exc}"

    return bool(response.get("ok")), response.get("message", "Registration failed.")


# --- Helpers ---
def pick_font(*choices, fallback="TkDefaultFont"):
    available = set(tkfont.families())
    for name in choices:
        if name in available:
            return name
    return fallback


def round_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, width=170, height=44,
                 fill=ACCENT, hover=ACCENT_HOVER, press=ACCENT_PRESS,
                 fg="#ffffff", bg=CARD, font=None, radius=12):
        super().__init__(parent, width=width, height=height,
                         bg=bg, highlightthickness=0, bd=0)
        self.command = command
        self.fill = fill
        self.hover = hover
        self.press = press
        self._inside = False

        self.configure(cursor="hand2")
        self._rect = round_rect(self, 1, 1, width - 1, height - 1, radius, fill=fill, outline="")
        self._label = self.create_text(width / 2, height / 2, text=text,
                                       fill=fg, font=font)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _paint(self, color):
        self.itemconfigure(self._rect, fill=color)

    def _on_enter(self, _):
        self._inside = True
        self._paint(self.hover)

    def _on_leave(self, _):
        self._inside = False
        self._paint(self.fill)

    def _on_press(self, _):
        self._paint(self.press)

    def _on_release(self, event):
        if self._inside and self.command:
            self._paint(self.hover)
            self.command()


def launch_script(status_var):
    python_exe = str(VENV_PYTHON) if VENV_PYTHON.is_file() else sys.executable
    try:
        process = subprocess.Popen(
            [python_exe, str(COMBINED_SCRIPT)],
            cwd=str(COMBINED_SCRIPT.parent),
        )
        status_var.set("Running combined.py")
        print(f"Launched {COMBINED_SCRIPT.name} successfully.")
        return process
    except Exception as e:
        status_var.set("Failed to launch combined.py")
        print(f"Error launching script: {e}")
        return None


def main():
    root = tk.Tk()
    root.title("VeriTag Face")
    root.configure(bg=BG)

    # Arch / tiling WM fix: float instead of tiling.
    try:
        root.attributes("-type", "dialog")
    except Exception:
        pass

    def toggle_fullscreen(_=None):
        root.attributes("-fullscreen", not root.attributes("-fullscreen"))

    root.attributes("-fullscreen", True)

    ui_font = pick_font("Inter", "Segoe UI", "Helvetica Neue", "Helvetica", "DejaVu Sans")
    mono_font = pick_font("JetBrains Mono", "SF Mono", "Consolas", "DejaVu Sans Mono")

    # Backdrop canvas so the card can have rounded corners and scale to any screen.
    canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    content = tk.Frame(canvas, bg=CARD)

    def make_screen():
        frame = tk.Frame(content, bg=CARD)
        frame.place(relwidth=1, relheight=1)
        return frame

    def show(frame):
        frame.tkraise()

    def build_header(parent, title_text, on_back=None):
        header = tk.Frame(parent, bg=CARD)
        header.pack(fill="x")

        accent = tk.Canvas(header, width=4, height=32, bg=CARD, highlightthickness=0)
        accent.create_rectangle(0, 0, 4, 32, fill=ACCENT, outline="")
        accent.pack(side="left", padx=(0, 14))

        tk.Label(header, text=title_text, font=(ui_font, 26, "bold"),
                 bg=CARD, fg=TEXT).pack(side="left")

        if on_back is not None:
            back = RoundedButton(
                header, "Back", on_back,
                width=96, height=36, radius=10,
                fill=GHOST, hover=GHOST_HOVER, press=GHOST_PRESS,
                fg=TEXT, font=(ui_font, 10, "bold"),
            )
            back.pack(side="right")
        return header

    # ---------------- Welcome screen ----------------
    welcome = make_screen()
    build_header(welcome, "VeriTag Face")
    tk.Label(
        welcome,
        text="Welcome. Secure webcam access control for your space.",
        font=(ui_font, 11), bg=CARD, fg=MUTED,
    ).pack(anchor="w", pady=(12, 0))

    welcome_actions = tk.Frame(welcome, bg=CARD)
    welcome_actions.pack(side="bottom")
    RoundedButton(
        welcome_actions, "Start", lambda: show(menu),
        width=220, height=52, font=(ui_font, 12, "bold"),
    ).pack(side="left", padx=(0, 14))
    RoundedButton(
        welcome_actions, "Close", root.destroy,
        width=120, height=52,
        fill=GHOST, hover=GHOST_HOVER, press=GHOST_PRESS,
        fg=TEXT, font=(ui_font, 12, "bold"),
    ).pack(side="left")

    # ---------------- Login / registration choice ----------------
    menu = make_screen()
    build_header(menu, "Access Control", on_back=lambda: show(welcome))
    tk.Label(
        menu,
        text="Sign in to launch the recognition camera, or register a new face.",
        font=(ui_font, 11), bg=CARD, fg=MUTED,
    ).pack(anchor="w", pady=(12, 0))

    menu_status_var = tk.StringVar(value="")
    menu_status = tk.Label(menu, textvariable=menu_status_var,
                           font=(mono_font, 10), bg=CARD, fg=MUTED)
    menu_status.pack(anchor="w", pady=(18, 0))

    def watch_combined(process):
        """Wait for combined.py to exit, then close or restore the launcher."""
        if process.poll() is None:
            root.after(150, watch_combined, process)
            return

        if process.returncode == 0:
            # Successful login: quit the whole app so no stray launcher window
            # is left behind (Hyprland used to float it into the top-left).
            root.destroy()
            return

        # Cancelled or failed: bring the launcher back in fullscreen.
        root.deiconify()
        root.attributes("-fullscreen", True)
        menu_status.configure(fg=MUTED)
        menu_status_var.set("Recognition cancelled.")

    def handle_login():
        try:
            response = request(HOST, PORT, {"action": "get_users"}, timeout=5)
        except Exception as exc:
            menu_status.configure(fg=ACCENT)
            menu_status_var.set(f"Server unavailable: {exc}")
            return

        if not response.get("ok") or not response.get("users"):
            menu_status.configure(fg=ACCENT)
            menu_status_var.set("No registered users. Register a face first.")
            return

        process = launch_script(menu_status_var)
        if process is None:
            menu_status.configure(fg=ACCENT)
            return

        # Hide the launcher while the camera/success window is on screen.
        menu_status.configure(fg=SUCCESS)
        root.withdraw()
        watch_combined(process)

    menu_actions = tk.Frame(menu, bg=CARD)
    menu_actions.pack(side="bottom")
    RoundedButton(
        menu_actions, "Login", handle_login,
        width=190, height=52, font=(ui_font, 12, "bold"),
    ).pack(side="left", padx=(0, 14))
    RoundedButton(
        menu_actions, "Register Account", lambda: show(register),
        width=190, height=52,
        fill=GHOST, hover=GHOST_HOVER, press=GHOST_PRESS,
        fg=TEXT, font=(ui_font, 12, "bold"),
    ).pack(side="left")

    # ---------------- Registration screen ----------------
    register = make_screen()
    build_header(register, "Register Account", on_back=lambda: show(menu))
    tk.Label(
        register,
        text="Enter a name, an integer ID, and upload a photo of the face.",
        font=(ui_font, 11), bg=CARD, fg=MUTED,
    ).pack(anchor="w", pady=(12, 0))

    form = tk.Frame(register, bg=CARD)
    form.pack(anchor="w", fill="x", pady=(20, 0))

    name_row = tk.Frame(form, bg=CARD)
    name_row.pack(anchor="w", fill="x", pady=(0, 12))
    tk.Label(name_row, text="Name", font=(ui_font, 11, "bold"),
             bg=CARD, fg=TEXT, width=10, anchor="w").pack(side="left")

    name_entry = tk.Entry(
        name_row, font=(mono_font, 12), bg=GHOST, fg=TEXT,
        insertbackground=TEXT, relief="flat", highlightthickness=1,
        highlightbackground=BORDER, highlightcolor=ACCENT,
    )
    name_entry.pack(side="left", ipady=6, ipadx=8)

    id_row = tk.Frame(form, bg=CARD)
    id_row.pack(anchor="w", fill="x", pady=(0, 12))
    tk.Label(id_row, text="ID", font=(ui_font, 11, "bold"),
             bg=CARD, fg=TEXT, width=10, anchor="w").pack(side="left")

    id_entry = tk.Entry(
        id_row, font=(mono_font, 12), bg=GHOST, fg=TEXT,
        insertbackground=TEXT, relief="flat", highlightthickness=1,
        highlightbackground=BORDER, highlightcolor=ACCENT,
    )
    id_entry.pack(side="left", ipady=6, ipadx=8)

    photo_row = tk.Frame(form, bg=CARD)
    photo_row.pack(anchor="w", fill="x")

    tk.Label(photo_row, text="Photo", font=(ui_font, 11, "bold"),
             bg=CARD, fg=TEXT, width=10, anchor="w").pack(side="left")

    selected_photo = {"path": None}

    photo_label = tk.Label(photo_row, text="No file selected",
                           font=(mono_font, 9), bg=CARD, fg=MUTED)

    def choose_photo():
        path = filedialog.askopenfilename(
            title="Select a face photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png")],
        )
        if path:
            selected_photo["path"] = Path(path)
            photo_label.configure(text=Path(path).name, fg=TEXT)

    RoundedButton(
        photo_row, "Upload Photo", choose_photo,
        width=150, height=40, radius=10,
        fill=GHOST, hover=GHOST_HOVER, press=GHOST_PRESS,
        fg=TEXT, font=(ui_font, 10, "bold"),
    ).pack(side="left")
    photo_label.pack(side="left", padx=(12, 0))

    register_status_var = tk.StringVar(value="")
    register_status = tk.Label(register, textvariable=register_status_var,
                               font=(mono_font, 10), bg=CARD, fg=MUTED)
    register_status.pack(anchor="w", pady=(16, 0))

    def handle_register():
        name = name_entry.get().strip()
        raw_id = id_entry.get().strip()

        if not name:
            register_status.configure(fg=ACCENT)
            register_status_var.set("Enter a name.")
            return
        if not raw_id.isdigit() or int(raw_id) <= 0:
            register_status.configure(fg=ACCENT)
            register_status_var.set("Enter a positive integer ID.")
            return
        if selected_photo["path"] is None:
            register_status.configure(fg=ACCENT)
            register_status_var.set("Please upload a photo first.")
            return

        ok, message = register_user(int(raw_id), name, selected_photo["path"])
        register_status.configure(fg=SUCCESS if ok else ACCENT)
        register_status_var.set(message)

        if ok:
            name_entry.delete(0, tk.END)
            id_entry.delete(0, tk.END)
            selected_photo["path"] = None
            photo_label.configure(text="No file selected", fg=MUTED)

    register_actions = tk.Frame(register, bg=CARD)
    register_actions.pack(side="bottom")
    RoundedButton(
        register_actions, "Register", handle_register,
        width=200, height=52, font=(ui_font, 12, "bold"),
    ).pack(side="left", padx=(0, 14))
    RoundedButton(
        register_actions, "Back", lambda: show(menu),
        width=120, height=52,
        fill=GHOST, hover=GHOST_HOVER, press=GHOST_PRESS,
        fg=TEXT, font=(ui_font, 12, "bold"),
    ).pack(side="left")

    # ---------------- Responsive card layout ----------------
    def layout(_=None):
        width = root.winfo_width()
        height = root.winfo_height()
        card_w = min(760, max(420, width - 120))
        card_h = min(520, max(340, height - 120))
        x1 = (width - card_w) // 2
        y1 = (height - card_h) // 2

        canvas.delete("card")
        round_rect(canvas, x1, y1, x1 + card_w, y1 + card_h, 26,
                   fill=CARD, outline=BORDER, width=1, tags="card")
        canvas.tag_lower("card")
        content.place(x=x1 + 48, y=y1 + 48, width=card_w - 96, height=card_h - 96)

    canvas.bind("<Configure>", layout)

    show(welcome)

    root.bind("<Escape>", lambda _: root.attributes("-fullscreen", False))
    root.bind("<F11>", toggle_fullscreen)
    root.bind("<Control-q>", lambda _: root.destroy())

    # Subtle fade-in when the window manager supports alpha.
    try:
        root.attributes("-alpha", 0.0)

        def fade(step=0):
            root.attributes("-alpha", min(1.0, step / 10))
            if step < 10:
                root.after(16, fade, step + 1)

        fade()
    except Exception:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()
