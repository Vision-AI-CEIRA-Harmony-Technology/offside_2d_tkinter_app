from email import header
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
    root.option_add("*TCombobox*Listbox.background", "#374151")
    root.option_add("*TCombobox*Listbox.foreground", "white")
    root.option_add("*TCombobox*Listbox.selectBackground", "#4B5563")
    root.option_add("*TCombobox*Listbox.selectForeground", "white")

    # -----------------------------
    # STYLE
    # -----------------------------
    style = ttk.Style()
    style.configure(
        "SidebarToggle.TButton",
        background="#111827",
        foreground="white",
        borderwidth=0,
        relief="flat",
        padding=0,
        font=("Segoe UI", 14, "bold")
    )

    style.map(
        "SidebarToggle.TButton",
        background=[("active", "#1F2937")]
    )
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

    style.configure(
        "Dark.TCombobox",
        fieldbackground="#374151",
        background="#374151",
        foreground="white",
        arrowcolor="white",
        borderwidth=1
    )

    style.map(
        "Dark.TCombobox",
        fieldbackground=[("readonly", "#374151")],
        foreground=[("readonly", "white")],
        selectbackground=[("readonly", "#4B5563")],
        selectforeground=[("readonly", "white")]
    )

    # -----------------------------
    # MAIN CONTAINER
    # -----------------------------
    main = ttk.Frame(root, style="Dark.TFrame")
    main.pack(fill="both", expand=True)

    # -----------------------------
    # CONTENT LAYOUT
    # -----------------------------
    content = ttk.Frame(main, style="Dark.TFrame")
    content.pack(fill="both", expand=True)
    app.content_container = content
    app.sidebar_handle = tk.Label(
        main,
        text=">>",
        bg="#111827",
        fg="white",
        font=("Segoe UI", 14, "bold"),
        cursor="hand2"
    )
    app.sidebar_handle.place(
        x=10,
        y=45
    )

    app.sidebar_handle.bind(
        "<Button-1>",
        lambda e: app.toggle_sidebar()
    )
    app.sidebar_handle.lift()
    # Sidebar
    app.sidebar = ttk.Frame(
        content,
        style="Toolbar.TFrame",
        width=250
    )

    app.sidebar.pack(
        side="left",
        fill="y",
        padx=(25, 5),
        pady=(35, 10)
    )
    app.sidebar.pack_forget()
    app.sidebar.pack_propagate(False)



    # Main workspace
    workspace = ttk.Frame(
        content,
        style="Dark.TFrame"
    )
    app.workspace = workspace
    workspace.pack(
        side="left",
        fill="both",
        expand=True
    )

    # ==================================================
    # MATCH INFO SIDEBAR
    # ==================================================

    ttk.Label(
        app.sidebar,
        text="MATCH DATA",
        style="Title.TLabel"
    ).pack(pady=(15, 20))

    app.left_team_label = ttk.Label(
        app.sidebar,
        text="Left Team : Not Set",
        style="Dark.TLabel"
    )
    app.left_team_label.pack(anchor="w", padx=10, pady=5)

    app.left_team_color = tk.Canvas(
        app.sidebar,
        width=40,
        height=20,
        highlightthickness=1
    )
    app.left_team_color.pack(anchor="w", padx=10)

    app.right_team_label = ttk.Label(
        app.sidebar,
        text="Right Team : Not Set",
        style="Dark.TLabel"
    )
    app.right_team_label.pack(anchor="w", padx=10, pady=(15, 5))

    app.right_team_color = tk.Canvas(
        app.sidebar,
        width=40,
        height=20,
        highlightthickness=1
    )
    app.right_team_color.pack(anchor="w", padx=10)

    app.half_label = ttk.Label(
        app.sidebar,
        text="Half : 1st",
        style="Dark.TLabel"
    )
    app.half_label.pack(anchor="w", padx=10, pady=(20, 5))

    app.attacking_team_label = ttk.Label(
        app.sidebar,
        text="Attacking Team : ?",
        style="Dark.TLabel"
    )
    app.attacking_team_label.pack(anchor="w", padx=10, pady=5)

    app.defending_team_label = ttk.Label(
        app.sidebar,
        text="Defending Team : ?",
        style="Dark.TLabel"
    )
    app.defending_team_label.pack(anchor="w", padx=10, pady=5)

    # -----------------------------
    # HEADER
    # -----------------------------
    header = ttk.Frame(workspace, style="Dark.TFrame")
    header.pack(fill="x", padx=20, pady=(15, 5))

    ttk.Label(
        header,
        text="AI Offside Detection",
        style="Title.TLabel"
    ).pack(side="left")

    # -----------------------------
    # TOOLBAR
    # -----------------------------
    toolbar = ttk.Frame(workspace, style="Toolbar.TFrame")
    toolbar.pack(fill="x", padx=20, pady=10)

    
###############################
    app.upload_btn = ttk.Button(
        toolbar,
        text="Upload Image",
        command=app.load_image,
        style="Action.TButton"
    )

    app.upload_video_btn = ttk.Button(
        toolbar,
        text="Upload Video",
        command=app.load_video,
        style="Action.TButton"
    )

    app.upload_video_btn.pack(side="left", padx=5, pady=10)

    app.upload_btn.pack(side="left", padx=5, pady=10)
    app.play_btn = ttk.Button(
        toolbar,
        text="Play/Pause",
        command=app.toggle_play_video,
        style="Action.TButton"
    )

    app.prev_frame_btn = ttk.Button(
        toolbar,
        text="◀ Frame",
        command=app.prev_frame,
        style="Action.TButton"
    )

    app.next_frame_btn = ttk.Button(
        toolbar,
        text="Frame >>",
        command=app.next_frame,
        style="Action.TButton"
    )

    app.choose_frame_btn = ttk.Button(
        toolbar,
        text="Choose Frame",
        command=app.choose_current_frame,
        style="Action.TButton"
    )

    app.back_to_video_btn = ttk.Button(
        toolbar,
        text="Back to Video",
        command=app.back_to_video,
        style="Action.TButton"
    )

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

    # app.add_keypoint_btn = ttk.Button(
    #     toolbar,
    #     text="Add keypoint",
    #     command=app.toggle_add_keypoint_mode,
    #     style="Action.TButton"
    # )

    # app.delete_keypoint_btn = ttk.Button(
    #     toolbar,
    #     text="Delete keypoint",
    #     command=app.delete_selected_keypoint,
    #     style="Action.TButton"
    # )

    app.attack_direction_var = tk.StringVar(value="Right")
    app.attack_dir_label = ttk.Label(
        toolbar,
        text="Attack Direction:",
        style="Dark.TLabel"
    )
    app.attack_direction_entry = ttk.Combobox(
        toolbar,
        textvariable=app.attack_direction_var,
        values=["Right", "Left"],
        width=8,
        state="readonly"
    )
    app.mitemp_var = tk.StringVar(value="1")

    app.mitemp_label = ttk.Label(
        toolbar,
        text="Mi-temps:",
        style="Dark.TLabel"
    )

    app.mitemp_entry = ttk.Combobox(
        toolbar,
        textvariable=app.mitemp_var,
        values=["1", "2"],
        state="readonly",
        width=5,
        style="Dark.TCombobox"
    )

    app.mitemp_entry.bind(
        "<<ComboboxSelected>>",
        lambda e: app.update_sidebar()
    )

    # attack_dir_label.pack(side="left", padx=(15, 5), pady=12)  #! dikra: moved to conditional in show_step

    # app.attack_direction_entry = ttk.Entry(
    #     toolbar,
    #     textvariable=app.attack_direction_var,
    #     width=8,
    #     font=("Segoe UI", 10),
    #     style="Dark.TEntry"
    # )
    # app.attack_direction_entry.pack(side="left", padx=(0, 15), pady=10)  #! dikra: moved to conditional in show_step

    # app.attack_direction_var.set("")

################################
    # -----------------------------
    # CANVAS FRAME
    # -----------------------------
    canvas_frame = tk.Frame(
        workspace,
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
    bottom = ttk.Frame(workspace, style="Dark.TFrame")
    bottom.pack(fill="x", padx=20, pady=15)

    # Configure grid so center column expands
    bottom.columnconfigure(0, weight=1)
    bottom.columnconfigure(1, weight=2)
    bottom.columnconfigure(2, weight=1)

    prev_btn = ttk.Button(
        bottom,
        text="◀ Previous",
        command=app.prev_step,
        style="Nav.TButton"
    )
    prev_btn.grid(row=0, column=0, sticky="w")

    app.prev_btn = prev_btn
    

    app.step_label = ttk.Label(
        bottom,
        text="Step 0 - Upload image",
        style="Dark.TLabel",
        anchor="center"
    )
    app.step_label.grid(row=0, column=1)

    #khadija code: init load state label
    app.status_label = ttk.Label(
        bottom,
        text="",
        style="Dark.TLabel",
        anchor="center"
    )
    app.status_label.grid(row=1, column=1)


    next_btn = ttk.Button(
        bottom,
        text="Next ▶",
        command=app.next_step,
        style="Nav.TButton"
    )
    next_btn.grid(row=0, column=2, sticky="e")

    app.next_btn = next_btn
    # -----------------------------
    # EVENTS
    # -----------------------------
    canvas.bind("<Configure>", lambda e: app.show_step())
    canvas.bind("<ButtonPress-1>", lambda e: app.on_mouse_down(e))
    canvas.bind("<B1-Motion>", lambda e: app.on_mouse_drag(e))
    canvas.bind("<ButtonRelease-1>", lambda e: app.on_mouse_up(e))
    canvas.bind("<Button-3>", lambda e: app.on_right_click(e))
    canvas.bind("<MouseWheel>", lambda e: app.on_mouse_scroll(e))
    canvas.bind("<Button-4>", lambda e: app.on_mouse_scroll(e))
    canvas.bind("<Button-5>", lambda e: app.on_mouse_scroll(e))

    def handle_delete(event):

        if app.current_step == 1:
            app.delete_selected_box()

        elif app.current_step == 2:
            app.delete_selected_roi()

        elif app.current_step == 6:
            app.delete_selected_keypoint()

    root.bind("<Delete>", handle_delete)

    return canvas
