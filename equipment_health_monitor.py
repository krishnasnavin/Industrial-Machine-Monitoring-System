import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import torch
import joblib
from threading import Thread, Event
from time import sleep
from torch import nn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from math import radians, cos, sin
import queue
import random

class EquipmentHealthMonitor:
    def __init__(self, root):
        self.root = root
        self.gauges = {}
        root.title("Industrial Equipment Health Monitor")
        root.geometry("1600x900")
        root.configure(bg="#f0f2f5")
        
        self.colors = {
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "danger": "#e74c3c",
            "warning": "#f39c12",
            "dark": "#2c3e50",
            "light": "#ecf0f1",
            "background": "#f0f2f5",
            "panel": "#ffffff",
            "text": "#34495e",
            "gauge_bg": "#ecf0f1",
        }
        
        self.model_loaded = False
        try:
            self.data_scaler = joblib.load("./preprocessor.joblib")
            self.predictive_model = self.load_model()
            self.model_loaded = True
        except FileNotFoundError:
            print("Model files not found. Running in demo mode.")

        self.operating_params = {
            'Vibration (mm/s)': 1.8, 
            'Bearing Temp (°C)': 72.0,
            'Oil Pressure (psi)': 145.0, 
            'Flow Rate (gpm)': 22.0,
            'Motor Load (kW)': 18.5
        }
        self.default_params = self.operating_params.copy()

        self.param_limits = {
            'Vibration (mm/s)': (0, 12),
            'Bearing Temp (°C)': (0, 120),
            'Oil Pressure (psi)': (0, 200),
            'Flow Rate (gpm)': (0, 40),
            'Motor Load (kW)': (0, 30)
        }

        self.sim_active = Event()
        self.data_history = {param: [] for param in self.operating_params}
        self.recent_readings = []
        self.data_queue = queue.Queue()
        
        self.setup_styles()
        self.create_main_interface()
        
        Thread(target=self.simulation_engine, daemon=True).start()
        root.after(100, self.update_interface)

    def load_model(self):
        class EquipmentHealthModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
                self.classifier = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                outputs, _ = self.gru(x)
                attention_weights = torch.softmax(self.attention(outputs), dim=1)
                context = torch.sum(attention_weights * outputs, dim=1)
                return torch.sigmoid(self.classifier(context))

        model = EquipmentHealthModel(input_size=5)
        model.load_state_dict(torch.load("./best_transformer_model.pth", 
                                       map_location=torch.device('cpu')))
        model.eval()
        return model

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Main.TFrame', background=self.colors["background"])
        style.configure('Panel.TFrame', background=self.colors["panel"], 
                        borderwidth=1, relief="solid")
        style.configure('Title.TLabel', background=self.colors["panel"], 
                       foreground=self.colors["dark"], font=('Segoe UI', 22, 'bold'))
        style.configure('Subtitle.TLabel', background=self.colors["panel"], 
                       foreground=self.colors["text"], font=('Segoe UI', 12))
        style.configure('Value.TLabel', background=self.colors["panel"], 
                       foreground=self.colors["primary"], font=('Segoe UI', 20, 'bold'))
        style.configure('Unit.TLabel', background=self.colors["panel"], 
                       foreground=self.colors["text"], font=('Segoe UI', 10))
        style.configure('Primary.TButton', background=self.colors["primary"], 
                       foreground="white", font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.configure('Secondary.TButton', background=self.colors["secondary"], 
                       foreground="white", font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.configure('Danger.TButton', background=self.colors["danger"], 
                       foreground="white", font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('Primary.TButton', background=[('active', '#2980b9'), ('disabled', '#bdc3c7')])
        style.map('Secondary.TButton', background=[('active', '#27ae60'), ('disabled', '#bdc3c7')])
        style.map('Danger.TButton', background=[('active', '#c0392b'), ('disabled', '#bdc3c7')])

    def create_main_interface(self):
        main_frame = ttk.Frame(self.root, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        header_frame = ttk.Frame(main_frame, style='Panel.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.columnconfigure(0, weight=1)
        ttk.Label(header_frame, text="EQUIPMENT HEALTH MONITOR", style='Title.TLabel'
                 ).grid(row=0, column=0, sticky="nsew", padx=20, pady=10)

        content_frame = ttk.Frame(main_frame, style='Main.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True)
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        
        left_panel = ttk.Frame(content_frame, style='Panel.TFrame')
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        self.create_gauges(left_panel)
        self.create_controls(left_panel)
        
        right_panel = ttk.Frame(content_frame, style='Panel.TFrame')
        right_panel.grid(row=0, column=1, sticky="nsew")
        self.create_health_status(right_panel)
        self.create_charts(right_panel)

    def create_gauges(self, parent):
        gauge_notebook = ttk.Notebook(parent)
        gauge_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Mechanical Tab
        mech_frame = ttk.Frame(gauge_notebook, style='Panel.TFrame')
        gauge_notebook.add(mech_frame, text="Mechanical")
        self.gauges['vibration'] = self.create_simple_gauge(
            mech_frame, 
            "Vibration (mm/s)", 
            self.param_limits['Vibration (mm/s)']
        )
        
        # Thermal Tab
        thermal_frame = ttk.Frame(gauge_notebook, style='Panel.TFrame')
        gauge_notebook.add(thermal_frame, text="Thermal")
        self.gauges['temperature'] = self.create_simple_gauge(
            thermal_frame,
            "Bearing Temp (°C)", 
            self.param_limits['Bearing Temp (°C)']
        )
        
        # Hydraulic Tab
        hydraulic_frame = ttk.Frame(gauge_notebook, style='Panel.TFrame')
        gauge_notebook.add(hydraulic_frame, text="Hydraulic")
        self.gauges['pressure'] = self.create_simple_gauge(
            hydraulic_frame,
            "Oil Pressure (psi)", 
            self.param_limits['Oil Pressure (psi)']
        )

    def create_simple_gauge(self, parent, param_name, limits):
        frame = ttk.Frame(parent, style='Panel.TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        ttk.Label(frame, text=param_name, style='Subtitle.TLabel').pack(anchor="center", padx=10, pady=(5, 0))
        
        # Gauge Canvas
        canvas_size = 200
        canvas = tk.Canvas(frame, width=canvas_size, height=canvas_size, 
                          bg=self.colors["panel"], highlightthickness=0)
        canvas.pack(pady=5)
        
        # Coordinates for the arc (centered with padding)
        coord = 20, 20, canvas_size-20, canvas_size-20
        
        # Background arc (270 degrees)
        canvas.create_arc(coord, start=135, extent=270, 
                         outline=self.colors["gauge_bg"], width=8, style="arc")
        
        # Colored status indicators
        for threshold, color in [(0.6, self.colors["secondary"]),
                                (0.8, self.colors["warning"]),
                                (1.0, self.colors["danger"])]:
            canvas.create_arc(coord, start=135+270*threshold, extent=270*(1-threshold),
                             outline=color, width=8, style="arc")
        
        # Foreground arc (current value)
        foreground_arc = canvas.create_arc(coord, start=135, extent=0, 
                                          outline=self.colors["primary"], width=8, style="arc")
        
        # Text display
        text_id = canvas.create_text(canvas_size/2, canvas_size/2, text="0.0",
                                   font=('Segoe UI', 16, 'bold'), fill=self.colors["text"])
        
        # Limits display
        canvas.create_text(canvas_size/2, canvas_size-15, text=f"MIN: {limits[0]}",
                          font=('Segoe UI', 8), fill=self.colors["text"])
        canvas.create_text(canvas_size/2, 15, text=f"MAX: {limits[1]}",
                          font=('Segoe UI', 8), fill=self.colors["text"])
        
        gauge_components = {
            "canvas": canvas,
            "foreground_arc": foreground_arc,
            "text_id": text_id,
            "limits": limits,
            "coord": coord
        }
        
        return gauge_components

    def update_gauge(self, gauge, value):
        canvas = gauge["canvas"]
        arc = gauge["foreground_arc"]
        text_id = gauge["text_id"]
        min_val, max_val = gauge["limits"]
        
        # Calculate angle (270 degree range from 135 to 405)
        normalized = (value - min_val) / (max_val - min_val)
        angle = 270 * normalized
        
        # Update arc
        canvas.itemconfig(arc, extent=angle)
        
        # Update color based on status
        if normalized > 0.8:
            color = self.colors["danger"]
        elif normalized > 0.6:
            color = self.colors["warning"]
        else:
            color = self.colors["secondary"]
        canvas.itemconfig(arc, outline=color)
        
        # Update text value
        canvas.itemconfig(text_id, text=f"{value:.1f}")

    def create_health_status(self, parent):
        health_frame = ttk.Frame(parent, style='Panel.TFrame')
        health_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        health_frame.columnconfigure(0, weight=1)
        
        ttk.Label(health_frame, text="EQUIPMENT HEALTH STATUS", style='Subtitle.TLabel'
                 ).grid(row=0, column=0, sticky="n", pady=(5, 2))
        self.health_meter = self.create_health_meter(health_frame)
        self.health_meter.grid(row=1, column=0, sticky="n", pady=5)
        self.health_label = ttk.Label(health_frame, text="Initializing...", style='Value.TLabel')
        self.health_label.grid(row=2, column=0, sticky="n", pady=5)
        self.recommendation_label = ttk.Label(health_frame, 
                                            text="No recommendations available", 
                                            style='Subtitle.TLabel', 
                                            wraplength=500)
        self.recommendation_label.grid(row=3, column=0, sticky="n", pady=5)

    def create_health_meter(self, parent):
        frame = ttk.Frame(parent, style='Panel.TFrame')
        self.health_canvas = tk.Canvas(frame, height=30, bg=self.colors["panel"], highlightthickness=0)
        self.health_canvas.pack(fill=tk.X, padx=10, pady=10)
        width = 500
        self.health_canvas.create_rectangle(0, 0, width, 20, fill="#e0e0e0", outline="")
        self.health_canvas.create_rectangle(0, 0, width*0.3, 20, fill=self.colors["secondary"], outline="")
        self.health_canvas.create_rectangle(width*0.3, 0, width*0.7, 20, fill=self.colors["warning"], outline="")
        self.health_canvas.create_rectangle(width*0.7, 0, width, 20, fill=self.colors["danger"], outline="")
        self.health_canvas.create_text(width*0.15, 10, text="Good", fill="white", font=('Segoe UI', 9, 'bold'))
        self.health_canvas.create_text(width*0.5, 10, text="Caution", fill="white", font=('Segoe UI', 9, 'bold'))
        self.health_canvas.create_text(width*0.85, 10, text="Critical", fill="white", font=('Segoe UI', 9, 'bold'))
        self.health_indicator = self.health_canvas.create_rectangle(0, 0, 0, 20, fill=self.colors["dark"], outline="")
        return frame

    def create_charts(self, parent):
        charts_frame = ttk.Frame(parent, style='Panel.TFrame')
        charts_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        charts_frame.rowconfigure(0, weight=1)
        charts_frame.columnconfigure(0, weight=1)
        
        chart_notebook = ttk.Notebook(charts_frame)
        chart_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        trend_frame = ttk.Frame(chart_notebook, style='Panel.TFrame')
        chart_notebook.add(trend_frame, text="Trend Analysis")
        self.temp_fig = Figure(figsize=(6, 3), facecolor=self.colors["panel"])
        self.temp_ax = self.temp_fig.add_subplot(111, facecolor=self.colors["panel"])
        self.temp_ax.tick_params(axis='both', colors=self.colors["text"])
        self.temp_ax.set_title("Bearing Temperature Trend", color=self.colors["dark"], pad=20)
        self.temp_ax.set_ylabel("Temperature (°C)", color=self.colors["text"])
        self.temp_ax.set_xlabel("Time (samples)", color=self.colors["text"])
        self.temp_line, = self.temp_ax.plot([], [], color=self.colors["primary"], linewidth=2, label='Actual')
        self.temp_ax.axhline(y=90, color=self.colors["danger"], linestyle='--', label='Warning Level')
        self.temp_ax.legend(facecolor=self.colors["panel"], edgecolor='none', labelcolor=self.colors["text"])
        self.temp_canvas = FigureCanvasTkAgg(self.temp_fig, master=trend_frame)
        self.temp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        vib_frame = ttk.Frame(chart_notebook, style='Panel.TFrame')
        chart_notebook.add(vib_frame, text="Vibration Analysis")
        self.vib_fig = Figure(figsize=(6, 3), facecolor=self.colors["panel"])
        self.vib_ax = self.vib_fig.add_subplot(111, facecolor=self.colors["panel"])
        self.vib_ax.tick_params(axis='both', colors=self.colors["text"])
        self.vib_ax.set_title("Vibration Spectrum", color=self.colors["dark"], pad=20)
        self.vib_ax.set_ylabel("Amplitude (mm/s)", color=self.colors["text"])
        self.vib_ax.set_xlabel("Frequency (Hz)", color=self.colors["text"])
        self.vib_bars = self.vib_ax.bar([1, 2, 3, 4], [0, 0, 0, 0], color=self.colors["primary"])
        self.vib_canvas = FigureCanvasTkAgg(self.vib_fig, master=vib_frame)
        self.vib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_controls(self, parent):
        controls_frame = ttk.Frame(parent, style='Panel.TFrame')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        sim_frame = ttk.Frame(controls_frame, style='Panel.TFrame')
        sim_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sim_frame, text="SIMULATION CONTROLS", style='Subtitle.TLabel').pack(anchor="w", padx=5, pady=(5,2))
        btn_frame = ttk.Frame(sim_frame, style='Panel.TFrame')
        btn_frame.pack(fill=tk.X, pady=5)
        self.start_btn = ttk.Button(btn_frame, text="START MONITORING", style='Primary.TButton', command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        self.stop_btn = ttk.Button(btn_frame, text="STOP MONITORING", style='Danger.TButton', command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        self.reset_btn = ttk.Button(btn_frame, text="RESET PARAMETERS", style='Secondary.TButton', command=self.reset_parameters)
        self.reset_btn.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        
        params_frame = ttk.Frame(controls_frame, style='Panel.TFrame')
        params_frame.pack(fill=tk.X, pady=5)
        ttk.Label(params_frame, text="PARAMETER ADJUSTMENT", style='Subtitle.TLabel').pack(anchor="w", padx=5, pady=(5,2))
        self.param_controls = {}
        for param in self.operating_params:
            p_frame = ttk.Frame(params_frame, style='Panel.TFrame')
            p_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(p_frame, text=param, style='Subtitle.TLabel', width=20).pack(side=tk.LEFT, padx=5)
            var = tk.DoubleVar(value=self.operating_params[param])
            value_label = ttk.Label(p_frame, textvariable=var, style='Value.TLabel', width=6)
            value_label.pack(side=tk.LEFT, padx=5)
            btn_frame = ttk.Frame(p_frame, style='Panel.TFrame')
            btn_frame.pack(side=tk.RIGHT, padx=5)
            dec_btn = ttk.Button(btn_frame, text="-", style='Primary.TButton', command=lambda p=param: self.adjust_parameter(p, -1))
            dec_btn.pack(side=tk.LEFT, padx=2)
            inc_btn = ttk.Button(btn_frame, text="+", style='Primary.TButton', command=lambda p=param: self.adjust_parameter(p, 1))
            inc_btn.pack(side=tk.LEFT, padx=2)
            self.param_controls[param] = {'value_label': value_label, 'var': var}

    def update_health_meter(self, probability):
        width = 500
        pos = width * probability
        self.health_canvas.coords(self.health_indicator, 0, 0, pos, 20)
        if probability > 0.8:
            status = "CRITICAL CONDITION"
            recommendation = "Immediate shutdown required. Contact maintenance team."
            self.health_label.config(style='Danger.TLabel')
        elif probability > 0.6:
            status = "HIGH RISK"
            recommendation = "Schedule maintenance soon. Monitor parameters closely."
            self.health_label.config(style='Warning.TLabel')
        elif probability > 0.3:
            status = "MODERATE RISK"
            recommendation = "Normal operation but monitor trends. Schedule inspection."
            self.health_label.config(style='Warning.TLabel')
        else:
            status = "NORMAL"
            recommendation = "Equipment operating within normal parameters."
            self.health_label.config(style='Secondary.TLabel')
            
        self.health_label.config(text=f"{status} - Risk: {probability:.1%}")
        self.recommendation_label.config(text=recommendation)

    def adjust_parameter(self, param, direction):
        steps = {
            'Vibration (mm/s)': 0.1,
            'Bearing Temp (°C)': 0.5,
            'Oil Pressure (psi)': 1.0,
            'Flow Rate (gpm)': 0.5,
            'Motor Load (kW)': 0.5
        }
        current = self.operating_params[param]
        new_val = current + direction * steps[param]
        min_val, max_val = self.param_limits[param]
        new_val = max(min_val, min(max_val, new_val))
        self.operating_params[param] = new_val
        self.param_controls[param]['var'].set(f"{new_val:.1f}")
        self.data_queue.put((self.operating_params.copy(), self.data_history.copy()))

    def start_monitoring(self):
        self.sim_active.set()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_monitoring(self):
        self.sim_active.clear()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def reset_parameters(self):
        self.operating_params = self.default_params.copy()
        for param in self.operating_params:
            self.param_controls[param]['var'].set(f"{self.operating_params[param]:.1f}")
        for param in self.data_history:
            self.data_history[param] = []
        self.temp_line.set_data([], [])
        self.temp_ax.relim()
        self.temp_ax.autoscale_view()
        self.temp_canvas.draw()
        self.update_health_meter(0.0)

    def simulation_engine(self):
        while True:
            if self.sim_active.is_set():
                for param in self.operating_params:
                    current = self.operating_params[param]
                    fluctuation = random.uniform(-0.5, 0.5) * (self.param_limits[param][1] * 0.02)
                    new_val = current + fluctuation
                    min_val, max_val = self.param_limits[param]
                    self.operating_params[param] = max(min_val, min(max_val, new_val))
                for param, value in self.operating_params.items():
                    self.data_history[param].append(value)
                    self.data_history[param] = self.data_history[param][-100:]
                self.recent_readings.append(self.operating_params.copy())
                self.recent_readings = self.recent_readings[-10:]
                self.data_queue.put((self.operating_params.copy(), self.data_history.copy()))
                if len(self.recent_readings) == 10:
                    self.predict_failure()
            sleep(0.5)

    def predict_failure(self):
        if not (self.model_loaded and self.data_scaler and self.predictive_model):
            # Calculate risk based on parameter values
            risk = 0.0
            for param, value in self.operating_params.items():
                min_val, max_val = self.param_limits[param]
                normalized_val = (value - min_val) / (max_val - min_val)
                
                # Increase risk based on proximity to limits
                if normalized_val > 0.8:
                    risk = max(risk, normalized_val)
                elif normalized_val > 0.6:
                    risk = max(risk, normalized_val * 0.8)
            
            self.update_health_meter(risk)
            return

    def update_interface(self):
        try:
            while True:
                current_values, history = self.data_queue.get_nowait()
                
                # Update gauges
                self.update_gauge(self.gauges['vibration'], current_values['Vibration (mm/s)'])
                self.update_gauge(self.gauges['temperature'], current_values['Bearing Temp (°C)'])
                self.update_gauge(self.gauges['pressure'], current_values['Oil Pressure (psi)'])
                
                # Update charts
                self.temp_line.set_data(np.arange(len(history['Bearing Temp (°C)'])), 
                                      history['Bearing Temp (°C)'])
                self.temp_ax.relim()
                self.temp_ax.autoscale_view()
                self.temp_canvas.draw()
                
                if len(history['Vibration (mm/s)']) >= 10:
                    vib_data = np.fft.fft(history['Vibration (mm/s)'][-10:])
                    vib_magnitude = np.abs(vib_data[:4])
                    for i, rect in enumerate(self.vib_bars):
                        rect.set_height(vib_magnitude[i])
                    self.vib_ax.relim()
                    self.vib_ax.autoscale_view()
                    self.vib_canvas.draw()
                
                # Update parameter controls
                for param, value in current_values.items():
                    if param in self.param_controls:
                        self.param_controls[param]['var'].set(f"{value:.1f}")
        except queue.Empty:
            pass
        self.root.after(100, self.update_interface)

if __name__ == "__main__":
    root = tk.Tk()
    app = EquipmentHealthMonitor(root)
    root.mainloop()