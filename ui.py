import tkinter as tk
from tkinter import ttk


def build_app_ui(root, app):

    root.title("AI Offside Detection System")
    # Full screen
    try:
        root.state("zoomed")
    except:
        root.attributes("-zoomed", True)
    root.configure(bg="#111827")

    # -----------------------------
    # STYLE
    # -----------------------------
    style = ttk.Style()

    style.theme_use("clam")

    style.configure(
        "Dark.TFrame",
        background="#111827"
    )

    style.configure(
        "Toolbar.TFrame",
        background="#1F2937"
    )

    style.configure(
        "Dark.TLabel",
        background="#111827",
        foreground="white",
        font=("Segoe UI", 11)
    )

    style.configure(
        "Title.TLabel",
        background="#111827",
        foreground="white",
        font=("Segoe UI", 18, "bold")
    )

    style.configure(
        "Action.TButton",
        font=("Segoe UI", 10, "bold"),
        padding=10,
        background="#2563EB",
        foreground="white",
        borderwidth=0
    )

    style.map(
        "Action.TButton",
        background=[
            ("active", "#3B82F6")
        ]
    )

    style.configure(
        "Nav.TButton",
        font=("Segoe UI", 10, "bold"),
        padding=12,
        background="#374151",
        foreground="white",
        borderwidth=0
    )

    style.map(
        "Nav.TButton",
        background=[
            ("active", "#4B5563")
        ]
    )

    style.configure(
        "Dark.TEntry",
        fieldbackground="#374151",
        background="#374151",
        foreground="white",
        borderwidth=1,
        relief="solid"
    )

    style.map(
        "Dark.TEntry",
        foreground=[("focus", "white")],
        fieldbackground=[("focus", "#4B5563")]
    )

    # -----------------------------
    # MAIN CONTAINER
    # -----------------------------
    main = ttk.Frame(root, style="Dark.TFrame")
    main.pack(fill="both", expand=True)

    # -----------------------------
    # HEADER
    # -----------------------------
    header = ttk.Frame(main, style="Dark.TFrame")
    header.pack(fill="x", padx=20, pady=(15, 5))

    ttk.Label(
        header,
        text="AI Offside Detection",
        style="Title.TLabel"
    ).pack(side="left")

    # -----------------------------
    # TOOLBAR
    # -----------------------------
    toolbar = ttk.Frame(main, style="Toolbar.TFrame")
    toolbar.pack(fill="x", padx=20, pady=10)
###############################
    app.upload_btn = ttk.Button(
        toolbar,
        text="Upload Image",
        command=app.load_image,
        style="Action.TButton"
    )

    app.upload_btn.pack(side="left", padx=5, pady=10)

    app.undo_btn = ttk.Button(
        toolbar,
        text="Undo",
        command=app.undo_box,
        style="Action.TButton"
    )

    app.clear_btn = ttk.Button(
        toolbar,
        text="Clear", 
        command=app.clear_boxes,
        style="Action.TButton"
    )

    app.attack_direction_var = tk.StringVar()
    app.attack_dir_label = ttk.Label(
        toolbar,
        text="Attack Direction:",
        style="Dark.TLabel"
    )
    # attack_dir_label.pack(side="left", padx=(15, 5), pady=12)  #! dikra: moved to conditional in show_step

    app.attack_direction_entry = ttk.Entry(
        toolbar,
        textvariable=app.attack_direction_var,
        width=8,
        font=("Segoe UI", 10),
        style="Dark.TEntry"
    )
    # app.attack_direction_entry.pack(side="left", padx=(0, 15), pady=10)  #! dikra: moved to conditional in show_step

    app.attack_direction_var.set("")
################################
    # -----------------------------
    # STEP INFO
    # -----------------------------
    app.step_label = ttk.Label(
        main,
        text="Step 0 - Upload image",
        style="Dark.TLabel"
    )

    app.step_label.pack(
        anchor="w",
        padx=25,
        pady=(0, 10)
    )

    # -----------------------------
    # CANVAS FRAME
    # -----------------------------
    canvas_frame = tk.Frame(
        main,
        bg="#0B1220",
        highlightthickness=2,
        highlightbackground="#374151"
    )

    canvas_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=10
    )

    canvas = tk.Canvas(
        canvas_frame,
        bg="#0B1220",
        bd=0,
        highlightthickness=0
    )

    canvas.pack(fill="both", expand=True)

    # -----------------------------
    # BOTTOM NAVIGATION
    # -----------------------------
    bottom = ttk.Frame(main, style="Dark.TFrame")
    bottom.pack(fill="x", padx=20, pady=15)

    ttk.Button(
        bottom,
        text="◀ Previous",
        command=app.prev_step,
        style="Nav.TButton"
    ).pack(side="left")

    ttk.Button(
        bottom,
        text="Next ▶",
        command=app.next_step,
        style="Nav.TButton"
    ).pack(side="right")

    # -----------------------------
    # EVENTS
    # -----------------------------
    canvas.bind("<Configure>", lambda e: app.show_step())
    canvas.bind("<ButtonPress-1>", lambda e: app.on_mouse_down(e))
    canvas.bind("<B1-Motion>", lambda e: app.on_mouse_drag(e))
    canvas.bind("<ButtonRelease-1>", lambda e: app.on_mouse_up(e))

    root.bind("<Delete>", lambda e: app.delete_selected_box())

    return canvas
