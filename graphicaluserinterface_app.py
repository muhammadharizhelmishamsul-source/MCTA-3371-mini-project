"""
gui_app.py  –  Fixed version
Tkinter GUI for Intelligent Robot Navigation (Fuzzy + GA)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np

from environment      import make_env, N_ACTIONS
import numpy as np  # ensure available in sim loop
from fuzzy_controller import FuzzyController, DEFAULT_PARAMS
from ga_optimizer     import GeneticOptimizer, evaluate_individual

# ── colours ──────────────────────────────────────────────────────────────────
BG            = "#1A1F2E"
PANEL         = "#242938"
BORDER        = "#2E3450"
ACCENT        = "#4F8EF7"
ACCENT2       = "#F7934F"
SUCCESS       = "#4FC87A"
DANGER        = "#F74F4F"
TEXT          = "#E8EAF0"
TEXT_DIM      = "#7A7F99"
CELL_FREE     = "#1E2435"
CELL_OBS      = "#4A5275"
CELL_START    = "#1A4A2E"
CELL_GOAL     = "#4A3A1A"
CELL_PATH     = "#1A2F4A"
ROBOT_FILL    = "#F74F4F"
SENSOR_COL    = "#F7934F"

CELL = 38   # pixels per grid cell


class RobotNavApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Robot Navigation — Fuzzy + GA")
        self.configure(bg=BG)
        self.resizable(True, True)

        # app state
        self.env        = None
        self.controller = None
        self.ga_params  = None
        self.running    = False
        self.sim_thread = None
        self.ga_thread  = None

        # tk vars
        self.map_id       = tk.IntVar(value=1)
        self.ctrl_mode    = tk.StringVar(value="fuzzy")
        self.delay_ms     = tk.IntVar(value=300)
        self.show_sensors = tk.BooleanVar(value=True)
        self.show_path    = tk.BooleanVar(value=True)

        self._build_ui()

        # Wait for window to appear before loading env so canvas has real size
        self.after(100, self._load_env)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=10, pady=6)
        tk.Label(bar, text="Robot Navigation  |  Fuzzy + GA Controller",
                 bg=BG, fg=ACCENT, font=("Helvetica", 14, "bold")).pack(side="left")

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # left controls
        left = tk.Frame(main, bg=PANEL, width=240,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)
        self._build_controls(left)

        # centre canvas with scrollbars
        cf = tk.Frame(main, bg=BG)
        cf.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(cf, bg=CELL_FREE, highlightthickness=0)
        hbar = tk.Scrollbar(cf, orient="horizontal", command=self.canvas.xview)
        vbar = tk.Scrollbar(cf, orient="vertical",   command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        hbar.pack(side="bottom", fill="x")
        vbar.pack(side="right",  fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # right panel
        right = tk.Frame(main, bg=PANEL, width=250,
                         highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)
        self._build_right(right)

        self.status_var = tk.StringVar(value="Ready — press Run")
        tk.Label(self, textvariable=self.status_var, bg=PANEL, fg=TEXT_DIM,
                 anchor="w", font=("Helvetica", 9)).pack(fill="x", padx=10, pady=(0, 4))

    def _sec(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold")).pack(anchor="w", padx=10, pady=(10, 2))

    def _build_controls(self, p):
        self._sec(p, "MAP")
        for lbl, val in [("Simple map  (10x10)", 1), ("Complex map (15x15)", 2)]:
            tk.Radiobutton(p, text=lbl, variable=self.map_id, value=val,
                           command=self._on_map_change,
                           bg=PANEL, fg=TEXT, selectcolor=ACCENT,
                           activebackground=PANEL,
                           font=("Helvetica", 10)).pack(anchor="w", padx=14)

        self._sec(p, "CONTROLLER")
        for lbl, val in [("Fuzzy Logic only", "fuzzy"),
                         ("Fuzzy + GA  (hybrid)", "hybrid")]:
            tk.Radiobutton(p, text=lbl, variable=self.ctrl_mode, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=ACCENT2,
                           activebackground=PANEL,
                           font=("Helvetica", 10)).pack(anchor="w", padx=14)

        self._sec(p, "DISPLAY")
        for lbl, var in [("Show sensor rays", self.show_sensors),
                         ("Show path history", self.show_path)]:
            tk.Checkbutton(p, text=lbl, variable=var,
                           command=lambda: self.after(0, self._draw_grid),
                           bg=PANEL, fg=TEXT, selectcolor=PANEL,
                           activebackground=PANEL,
                           font=("Helvetica", 10)).pack(anchor="w", padx=14)

        self._sec(p, "SPEED")
        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x", padx=10)
        tk.Label(row, text="Fast", bg=PANEL, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")
        tk.Scale(row, from_=20, to=800, orient="horizontal",
                 variable=self.delay_ms, bg=PANEL, fg=TEXT,
                 troughcolor=BORDER, highlightthickness=0,
                 showvalue=False).pack(side="left", fill="x", expand=True)
        tk.Label(row, text="Slow", bg=PANEL, fg=TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")

        self._sec(p, "SIMULATION")
        btn = dict(font=("Helvetica", 10, "bold"), relief="flat",
                   cursor="hand2", pady=7, bd=0)
        self.run_btn = tk.Button(p, text="RUN", bg=ACCENT, fg="white",
                                 command=self.start_sim, **btn)
        self.run_btn.pack(fill="x", padx=10, pady=2)

        self.stop_btn = tk.Button(p, text="STOP", bg=DANGER, fg="white",
                                  command=self.stop_sim, state="disabled", **btn)
        self.stop_btn.pack(fill="x", padx=10, pady=2)

        tk.Button(p, text="RESET", bg=BORDER, fg=TEXT,
                  command=self._reset, **btn).pack(fill="x", padx=10, pady=2)

        self._sec(p, "STATISTICS")
        self.stat_vars = {}
        for lbl in ["Steps", "Collisions", "Distance", "Reward", "Status"]:
            row2 = tk.Frame(p, bg=PANEL)
            row2.pack(fill="x", padx=10, pady=1)
            tk.Label(row2, text=f"{lbl}:", bg=PANEL, fg=TEXT_DIM,
                     width=11, anchor="w",
                     font=("Helvetica", 9)).pack(side="left")
            v = tk.StringVar(value="--")
            self.stat_vars[lbl] = v
            tk.Label(row2, textvariable=v, bg=PANEL, fg=TEXT,
                     font=("Helvetica", 9, "bold")).pack(side="left")

    def _build_right(self, p):
        self._sec(p, "FUZZY ACTION SCORES")
        self.score_cv = tk.Canvas(p, bg=PANEL, height=160, highlightthickness=0)
        self.score_cv.pack(fill="x", padx=6)
        self._draw_scores(np.zeros(8))

        self._sec(p, "GA OPTIMISATION")
        self.ga_status_var = tk.StringVar(value="Not trained yet")
        tk.Label(p, textvariable=self.ga_status_var, bg=PANEL, fg=TEXT,
                 font=("Helvetica", 9), wraplength=220,
                 justify="left").pack(anchor="w", padx=10)

        tk.Button(p, text="TRAIN GA", bg=ACCENT2, fg="white",
                  font=("Helvetica", 10, "bold"), relief="flat",
                  cursor="hand2", pady=6, bd=0,
                  command=self._start_ga).pack(fill="x", padx=10, pady=6)

        self.ga_bar = ttk.Progressbar(p, mode="determinate", maximum=100)
        self.ga_bar.pack(fill="x", padx=10, pady=(0, 4))

        self._sec(p, "GA GENERATION LOG")
        lf = tk.Frame(p, bg=PANEL)
        lf.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.ga_log = tk.Text(lf, bg="#141720", fg=TEXT_DIM,
                              font=("Courier", 8), wrap="word",
                              state="disabled", relief="flat", height=10)
        sb = tk.Scrollbar(lf, command=self.ga_log.yview,
                          bg=PANEL, troughcolor=BORDER)
        self.ga_log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.ga_log.pack(side="left", fill="both", expand=True)

        self._sec(p, "FITNESS HISTORY")
        self.fit_cv = tk.Canvas(p, bg=PANEL, height=80, highlightthickness=0)
        self.fit_cv.pack(fill="x", padx=6, pady=(0, 8))

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_grid(self):
        if self.env is None:
            return
        env  = self.env
        c    = self.canvas
        cell = CELL
        c.delete("all")

        path_set = set(env.path) if self.show_path.get() else set()

        for r in range(env.n_rows):
            for col in range(env.n_cols):
                x0 = col * cell
                y0 = r * cell
                x1 = x0 + cell
                y1 = y0 + cell

                if env.grid[r, col] == 1:
                    fill = CELL_OBS
                elif (r, col) == env.goal_pos:
                    fill = CELL_GOAL
                elif (r, col) == env.start_pos:
                    fill = CELL_START
                elif (r, col) in path_set:
                    fill = CELL_PATH
                else:
                    fill = CELL_FREE

                c.create_rectangle(x0, y0, x1, y1,
                                   fill=fill, outline=BORDER, width=0.8)

        # S / G labels
        sr, sc2 = env.start_pos
        gr, gc  = env.goal_pos
        c.create_text(sc2 * cell + cell // 2, sr * cell + cell // 2,
                      text="S", fill=SUCCESS, font=("Helvetica", 11, "bold"))
        c.create_text(gc * cell + cell // 2, gr * cell + cell // 2,
                      text="G", fill=ACCENT2, font=("Helvetica", 11, "bold"))

        # sensor rays
        if self.show_sensors.get():
            obs = env._get_observation()
            rr, rc = env.robot_pos
            ox = rc * cell + cell // 2
            oy = rr * cell + cell // 2
            for i, (sdr, sdc) in enumerate(env.SENSOR_DIRS):
                d  = obs["sensor_readings"][i]
                ex = (rc + sdc * d) * cell + cell // 2
                ey = (rr + sdr * d) * cell + cell // 2
                c.create_line(ox, oy, ex, ey, fill=SENSOR_COL,
                              width=1, dash=(3, 5))

        # robot
        rr, rc = env.robot_pos
        m  = 5
        x0 = rc * cell + m
        y0 = rr * cell + m
        x1 = rc * cell + cell - m
        y1 = rr * cell + cell - m
        c.create_oval(x0, y0, x1, y1, fill=ROBOT_FILL, outline="white", width=2)
        c.create_text(rc * cell + cell // 2, rr * cell + cell // 2,
                      text="R", fill="white", font=("Helvetica", 9, "bold"))

        # update scroll region so the whole map is reachable
        total_w = env.n_cols * cell
        total_h = env.n_rows * cell
        c.configure(scrollregion=(0, 0, total_w, total_h))

    def _draw_scores(self, scores: np.ndarray):
        cv = self.score_cv
        cv.delete("all")
        names = ["UP", "DOWN", "LEFT", "RIGHT"]
        # pad to 4 if needed
        scores = np.array(scores).flatten()[:4]
        W     = 238
        bw    = W // 4 - 6
        mx    = max(float(np.max(np.abs(scores))), 0.01)
        best  = int(np.argmax(scores))

        for i, (nm, sc) in enumerate(zip(names, scores)):
            bh    = int(abs(sc) / mx * 100)
            x0    = 8 + i * (bw + 8)
            y_top = 120 - bh
            fill  = ACCENT2 if i == best else ACCENT
            cv.create_rectangle(x0, y_top, x0 + bw, 120, fill=fill, outline="")
            cv.create_text(x0 + bw // 2, 132, text=nm,
                           fill=TEXT_DIM, font=("Helvetica", 8))
            if bh > 12:
                cv.create_text(x0 + bw // 2, y_top - 7,
                               text=f"{sc:.2f}",
                               fill=TEXT, font=("Helvetica", 8))

    def _draw_fitness(self):
        cv   = self.fit_cv
        cv.delete("all")
        hist = getattr(self, "_ga_hist", [])
        if len(hist) < 2:
            return
        vals = [h[1] for h in hist]
        mn, mx = min(vals), max(vals)
        rng  = max(mx - mn, 1.0)
        W, H = 238, 78
        pts  = []
        for i, v in enumerate(vals):
            x = int(4 + i / (len(vals) - 1) * (W - 8))
            y = int(H - 4 - (v - mn) / rng * (H - 10))
            pts += [x, y]
        if len(pts) >= 4:
            cv.create_line(pts, fill=SUCCESS, width=2, smooth=True)

    # ── environment ───────────────────────────────────────────────────────────

    def _load_env(self):
        self.env = make_env(self.map_id.get())
        self.env.reset()
        # set canvas to exact map size
        w = self.env.n_cols * CELL
        h = self.env.n_rows * CELL
        self.canvas.config(width=w, height=h)
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self._draw_grid()
        self._reset_stats()
        self.status_var.set("Map loaded. Press RUN to start.")

    def _on_map_change(self):
        self.stop_sim()
        self._load_env()

    def _reset(self):
        self.stop_sim()
        if self.env:
            self.env.reset()
            self._draw_grid()
            self._reset_stats()
            self.status_var.set("Reset — press RUN")

    def _reset_stats(self):
        for v in self.stat_vars.values():
            v.set("--")

    # ── stats ─────────────────────────────────────────────────────────────────

    def _update_stats(self, info, total_reward):
        self.stat_vars["Steps"].set(str(info["steps"]))
        self.stat_vars["Collisions"].set(str(info["collisions"]))
        dist = self.env._get_observation()["distance_to_goal"]
        self.stat_vars["Distance"].set(f"{dist:.2f}")
        self.stat_vars["Reward"].set(f"{total_reward:.1f}")
        done = self.env.robot_pos == self.env.goal_pos
        self.stat_vars["Status"].set("GOAL!" if done else "Running...")

    # ── simulation ────────────────────────────────────────────────────────────

    def start_sim(self):
        if self.running:
            return
        self.env = make_env(self.map_id.get())
        self.env.reset()
        w = self.env.n_cols * CELL
        h = self.env.n_rows * CELL
        self.canvas.config(width=w, height=h)
        self.canvas.configure(scrollregion=(0, 0, w, h))

        if self.ctrl_mode.get() == "hybrid" and self.ga_params is not None:
            self.controller = FuzzyController(self.ga_params)
        else:
            if self.ctrl_mode.get() == "hybrid":
                messagebox.showwarning("GA not trained",
                    "Train the GA first, then switch to Hybrid mode.\n"
                    "Running Fuzzy-only for now.")
            self.controller = FuzzyController(DEFAULT_PARAMS.copy())

        self.running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        mode_lbl = "Fuzzy+GA" if (self.ctrl_mode.get() == "hybrid"
                                   and self.ga_params is not None) else "Fuzzy"
        map_lbl  = "Simple" if self.map_id.get() == 1 else "Complex"
        self.status_var.set(f"Running {mode_lbl} on {map_lbl} map...")
        self._draw_grid()

        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()

    def stop_sim(self):
        self.running = False
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _sim_loop(self):
        env          = self.env
        ctrl         = self.controller
        total_reward = 0.0
        max_steps    = env.n_rows * env.n_cols * 4
        obs          = env._get_observation()

        # loop detection
        recent_positions = []   # last N positions
        LOOP_WINDOW      = 10   # if same cell seen this many times → escape
        escape_counter   = 0    # how many random escape steps remaining

        for _ in range(max_steps):
            if not self.running:
                break

            # ── loop detection ──
            recent_positions.append(env.robot_pos)
            if len(recent_positions) > 20:
                recent_positions.pop(0)

            # count how often current pos appeared recently
            repeat_count = recent_positions.count(env.robot_pos)

            if escape_counter > 0:
                # take a random action to break out
                action  = int(np.random.randint(0, 4))
                scores  = np.zeros(4)
                scores[action] = 1.0
                escape_counter -= 1
            elif repeat_count >= 4:
                # stuck — trigger 3-step random escape
                escape_counter  = 3
                recent_positions.clear()
                action  = int(np.random.randint(0, 4))
                scores  = np.zeros(4)
                scores[action] = 1.0
            else:
                action, scores = ctrl.decide_with_scores(obs)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            self.after(0, self._draw_grid)
            self.after(0, self._draw_scores, scores.copy())
            self.after(0, self._update_stats, info, total_reward)

            time.sleep(self.delay_ms.get() / 1000.0)

            if done:
                success = env.robot_pos == env.goal_pos
                msg = ("GOAL REACHED!" if success else "Time limit reached.")
                self.after(0, self.status_var.set,
                           f"{msg}  Steps={info['steps']}  "
                           f"Collisions={info['collisions']}")
                break

        self.after(0, self.stop_sim)

    # ── GA training ───────────────────────────────────────────────────────────

    def _start_ga(self):
        if self.ga_thread and self.ga_thread.is_alive():
            messagebox.showinfo("GA", "Training already running.")
            return
        self.ga_log.config(state="normal")
        self.ga_log.delete("1.0", "end")
        self.ga_log.config(state="disabled")
        self._ga_hist = []
        self.ga_bar["value"] = 0
        self.ga_status_var.set("Training in progress...")
        self.ga_thread = threading.Thread(target=self._ga_loop, daemon=True)
        self.ga_thread.start()

    def _ga_loop(self):
        n_gen = 15
        pop   = 20

        def cb(gen, best_fit, _params):
            pct = int(gen / n_gen * 100)
            self.after(0, self._ga_tick, gen, n_gen, best_fit, pct)

        ga   = GeneticOptimizer(pop_size=pop, n_generations=n_gen, callback=cb)
        best = ga.run(verbose=False)
        self.ga_params = best
        self._ga_hist  = ga.history
        np.save("best_params.npy", best)
        fit = evaluate_individual(best)
        self.after(0, self._ga_finished, fit)

    def _ga_tick(self, gen, n_gen, best_fit, pct):
        self.ga_bar["value"] = pct
        self.ga_status_var.set(f"Gen {gen}/{n_gen}   best={best_fit:.1f}")
        self.ga_log.config(state="normal")
        self.ga_log.insert("end", f"Gen {gen:3d}  best={best_fit:7.2f}\n")
        self.ga_log.see("end")
        self.ga_log.config(state="disabled")
        self._draw_fitness()

    def _ga_finished(self, fit):
        self.ga_bar["value"] = 100
        self.ga_status_var.set(
            f"Done!  Fitness={fit:.1f}\n"
            "Select 'Fuzzy+GA' then press RUN!")
        self._draw_fitness()
        messagebox.showinfo("GA Complete",
                            f"Best fitness: {fit:.2f}\n\n"
                            "Now select 'Fuzzy + GA (hybrid)' and press RUN!")


if __name__ == "__main__":
    app = RobotNavApp()
    app.mainloop()