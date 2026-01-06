import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QProgressBar,
                           QFileDialog, QTabWidget, QSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg
import numpy as np
from acis.environment.cache_env import CacheEnvironment

class CacheSimulationThread(QThread):
    """Thread to run cache simulation and emit progress"""
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    
    def __init__(self, config, trace_data):
        super().__init__()
        self.config = config
        self.trace_data = trace_data
        self.running = True
        
    def run(self):
        cache_size = self.config['cache_size']
        ways = self.config['ways']
        block_size = self.config['block_size']

        # Build occurrence map for lookahead (block -> list of positions)
        occ_map = {}
        for pos, a in enumerate(self.trace_data):
            block = int(a) // block_size
            occ_map.setdefault(block, []).append(pos)

        # Initialize environments with occ_map so agent can use next-use info
        lru_cache = CacheEnvironment(cache_size=cache_size, ways=ways, block_size=block_size,
                                     occ_map=occ_map, trace=self.trace_data, trace_length=len(self.trace_data))
        agent_cache = CacheEnvironment(cache_size=cache_size, ways=ways, block_size=block_size,
                                       occ_map=occ_map, trace=self.trace_data, trace_length=len(self.trace_data))
        
        lru_hits = 0
        agent_hits = 0
        lru_energy = 0
        agent_energy = 0
        total = len(self.trace_data)
        
        for i, addr in enumerate(self.trace_data):
            if not self.running:
                break
                
            # Process LRU
            lru_hit = lru_cache.is_hit(addr)
            lru_cache.step(None, addr)  # None = use LRU policy
            if lru_hit:
                lru_hits += 1
            lru_energy += 3 if not lru_hit else 1
            
            # Process Agent using a simple heuristic: evict the way with farthest next-use
            agent_hit = agent_cache.is_hit(addr)
            # compute agent action only on miss
            if not agent_hit:
                # agent can inspect the set state; next_use_norm is every 3rd element starting at index 2
                state = agent_cache._state_from_address(addr)
                next_use_norms = state[2::3]
                # choose the way with largest next_use_norm (farthest or never used)
                try:
                    action = int(np.argmax(next_use_norms))
                except Exception:
                    action = None
            else:
                action = None

            agent_cache.step(action, addr, pos=i)
            if agent_hit:
                agent_hits += 1
            agent_energy += 3 if not agent_hit else 1
            
            # Emit progress every 100 accesses
            if i % 100 == 0 or i == total - 1:
                metrics = {
                    'progress': (i + 1) / total * 100,
                    'lru_hit_rate': (lru_hits / (i + 1)) * 100,
                    'agent_hit_rate': (agent_hits / (i + 1)) * 100,
                    'lru_energy': lru_energy,
                    'agent_energy': agent_energy,
                    'accesses': i + 1,
                    'lru_hits': lru_hits,
                    'agent_hits': agent_hits
                }
                self.update_signal.emit(metrics)
        
        self.finished_signal.emit()
    
    def stop(self):
        self.running = False

class MainInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A-CIS: Cache Intelligence System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Setup main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Add configuration tab
        config_widget = self._create_config_tab()
        tabs.addTab(config_widget, "Configuration")
        
        # Add monitoring tab
        monitor_widget = self._create_monitor_tab()
        tabs.addTab(monitor_widget, "Performance Monitor")
        
        # Initialize data storage
        self.metrics_history = {
            'timestamps': [],
            'lru_hit_rates': [],
            'agent_hit_rates': [],
            'lru_energy': [],
            'agent_energy': []
        }
        
        # Set dark theme
        self._set_dark_theme()
        
    def _create_config_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Cache configuration
        self.cache_size = QSpinBox()
        self.cache_size.setRange(64, 16384)
        self.cache_size.setValue(1024)
        layout.addRow("Cache Size (blocks):", self.cache_size)
        
        self.ways = QSpinBox()
        self.ways.setRange(1, 16)
        self.ways.setValue(4)
        layout.addRow("Ways:", self.ways)
        
        self.block_size = QSpinBox()
        self.block_size.setRange(16, 256)
        self.block_size.setValue(64)
        layout.addRow("Block Size (bytes):", self.block_size)
        
        # Load trace button
        self.load_button = QPushButton("Load Trace File")
        self.load_button.clicked.connect(self._load_trace)
        layout.addRow(self.load_button)
        
        # Start simulation button
        self.start_button = QPushButton("Start Simulation")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_simulation)
        layout.addRow(self.start_button)
        
        return widget
        
    def _create_monitor_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress section
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(QLabel("Progress:"))
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_widget)
        
        # Statistics section
        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        
        # LRU Stats
        lru_group = self._create_stat_group("LRU Cache")
        self.lru_stats = lru_group
        stats_layout.addWidget(lru_group)
        
        # Agent Stats
        agent_group = self._create_stat_group("AI Agent Cache")
        self.agent_stats = agent_group
        stats_layout.addWidget(agent_group)
        
        layout.addWidget(stats_widget)
        
        # Graphs section
        graphs_widget = QWidget()
        graphs_layout = QVBoxLayout(graphs_widget)
        
        # Hit rate graph
        self.hit_rate_plot = pg.PlotWidget(title="Cache Hit Rate")
        self.hit_rate_plot.setLabel('left', 'Hit Rate (%)')
        self.hit_rate_plot.setLabel('bottom', 'Accesses')
        self.hit_rate_plot.addLegend()
        self.hit_rate_plot.showGrid(x=True, y=True)
        self.lru_curve = self.hit_rate_plot.plot(pen='r', name='LRU')
        self.agent_curve = self.hit_rate_plot.plot(pen='g', name='Agent')
        graphs_layout.addWidget(self.hit_rate_plot)
        
        # Energy graph
        self.energy_plot = pg.PlotWidget(title="Energy Consumption")
        self.energy_plot.setLabel('left', 'Energy Units')
        self.energy_plot.setLabel('bottom', 'Accesses')
        self.energy_plot.addLegend()
        self.energy_plot.showGrid(x=True, y=True)
        self.lru_energy_curve = self.energy_plot.plot(pen='r', name='LRU')
        self.agent_energy_curve = self.energy_plot.plot(pen='g', name='Agent')
        graphs_layout.addWidget(self.energy_plot)
        
        layout.addWidget(graphs_widget)
        return widget
    
    def _create_stat_group(self, title):
        group = QWidget()
        layout = QVBoxLayout(group)
        
        # Title
        header = QLabel(title)
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)
        
        # Stats
        stats = {
            'Hit Rate': '0%',
            'Miss Rate': '0%',
            'Total Hits': '0',
            'Energy Usage': '0 units'
        }
        
        stat_widgets = {}
        for name, value in stats.items():
            stat_layout = QHBoxLayout()
            label = QLabel(f"{name}:")
            value_label = QLabel(value)
            value_label.setStyleSheet("font-family: monospace;")
            stat_layout.addWidget(label)
            stat_layout.addWidget(value_label)
            stat_layout.addStretch()
            layout.addLayout(stat_layout)
            stat_widgets[name] = value_label
            
        group.stats = stat_widgets
        return group
    
    def _set_dark_theme(self):
        app = QApplication.instance()
        app.setStyle('Fusion')
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)
    
    def _load_trace(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Trace File",
            "",
            "All Files (*.*)"
        )
        
        if filename:
            try:
                if filename.endswith('.npy'):
                    from acis.utils.trace_generator import load_trace
                    self.trace_data = load_trace(filename)
                else:
                    # Generate sample trace for demonstration
                    from acis.utils.trace_generator import generate_sample_trace
                    self.trace_data = generate_sample_trace(size=10000, pattern_type='mixed')
                    # Save for future use
                    from acis.utils.trace_generator import save_trace
                    save_path = os.path.join(os.path.dirname(filename), 'sample_trace.npy')
                    save_trace(self.trace_data, save_path)
                self.start_button.setEnabled(True)
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error", f"Failed to load trace: {str(e)}")
    
    def _start_simulation(self):
        config = {
            'cache_size': self.cache_size.value(),
            'ways': self.ways.value(),
            'block_size': self.block_size.value()
        }
        
        # Clear previous data
        self.metrics_history = {
            'timestamps': [],
            'lru_hit_rates': [],
            'agent_hit_rates': [],
            'lru_energy': [],
            'agent_energy': []
        }
        
        # Disable configuration
        self.start_button.setEnabled(False)
        self.load_button.setEnabled(False)
        
        # Start simulation thread
        self.sim_thread = CacheSimulationThread(config, self.trace_data)
        self.sim_thread.update_signal.connect(self._update_metrics)
        self.sim_thread.finished_signal.connect(self._simulation_finished)
        self.sim_thread.start()
    
    def _update_metrics(self, metrics):
        # Update progress bar
        self.progress_bar.setValue(int(metrics['progress']))
        
        # Update stats
        self.lru_stats.stats['Hit Rate'].setText(f"{metrics['lru_hit_rate']:.2f}%")
        self.lru_stats.stats['Miss Rate'].setText(f"{100 - metrics['lru_hit_rate']:.2f}%")
        self.lru_stats.stats['Total Hits'].setText(str(metrics['lru_hits']))
        self.lru_stats.stats['Energy Usage'].setText(f"{metrics['lru_energy']} units")
        
        self.agent_stats.stats['Hit Rate'].setText(f"{metrics['agent_hit_rate']:.2f}%")
        self.agent_stats.stats['Miss Rate'].setText(f"{100 - metrics['agent_hit_rate']:.2f}%")
        self.agent_stats.stats['Total Hits'].setText(str(metrics['agent_hits']))
        self.agent_stats.stats['Energy Usage'].setText(f"{metrics['agent_energy']} units")
        
        # Update history
        self.metrics_history['timestamps'].append(metrics['accesses'])
        self.metrics_history['lru_hit_rates'].append(metrics['lru_hit_rate'])
        self.metrics_history['agent_hit_rates'].append(metrics['agent_hit_rate'])
        self.metrics_history['lru_energy'].append(metrics['lru_energy'])
        self.metrics_history['agent_energy'].append(metrics['agent_energy'])
        
        # Update plots
        self.lru_curve.setData(
            self.metrics_history['timestamps'],
            self.metrics_history['lru_hit_rates']
        )
        self.agent_curve.setData(
            self.metrics_history['timestamps'],
            self.metrics_history['agent_hit_rates']
        )
        self.lru_energy_curve.setData(
            self.metrics_history['timestamps'],
            self.metrics_history['lru_energy']
        )
        self.agent_energy_curve.setData(
            self.metrics_history['timestamps'],
            self.metrics_history['agent_energy']
        )
    
    def _simulation_finished(self):
        self.start_button.setEnabled(True)
        self.load_button.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    window = MainInterface()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()