#!/usr/bin/env python

import os
import time
import psutil
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
import platform
import subprocess
from joblib import Parallel, delayed

class PerformanceMonitor:
    def __init__(self, config: Dict):
        """
        Initialize the performance monitor.
        
        Args:
            config (Dict): Configuration dictionary containing pipeline settings
        """
        self.config = config
        self.start_time = time.time()
        self.metrics = {
            'module': [],
            'cpu_percent': [],
            'cpu_percent_per_core': [],  # CPU percentage per core
            'cores_used': [],  # Number of cores used
            'max_cores_available': [],  # Maximum cores available for this module
            'memory_usage': [],
            'disk_read': [],
            'disk_write': [],
            'duration': [],
            'timestamp': []
        }
        self.process = psutil.Process(os.getpid())
        self.output_dir = Path(config['GENERAL']['OUTPUT'])
        self.benchmark_file = self.output_dir / f"{config['GENERAL']['RUN_ID']}_performance_benchmark.csv"
        
        # Get system information
        self.total_cores = psutil.cpu_count()
        self.parallel_mode = config.get('GENERAL', {}).get('PARALLEL', False)
        self.num_threads = int(os.environ.get('OMP_NUM_THREADS', 1))
        self.ncpu = config.get('GENERAL', {}).get('NCPU', 1)
        
        # Track joblib processes
        self.joblib_processes = set()
        
        # Initialize disk I/O tracking based on platform
        self.platform = platform.system().lower()
        self.supports_io_counters = hasattr(self.process, 'io_counters')
        
        if self.supports_io_counters:
            self.last_disk_io = self.process.io_counters()
            self.io_method = 'psutil'
        elif self.platform == 'darwin':  # macOS
            self.io_method = 'iostat'
            self.disk_name = self._get_macos_disk_name()
            if self.disk_name:
                # Test if we can actually monitor this disk
                if self._test_disk_monitoring():
                    self.last_disk_io = self._get_macos_disk_io()
                else:
                    self.io_method = 'none'
                    self.last_disk_io = None
                    logging.warning("Could not monitor the detected disk, falling back to no I/O monitoring")
            else:
                self.io_method = 'none'
                self.last_disk_io = None
                logging.warning("Could not determine disk name for I/O monitoring on macOS")
        else:
            self.io_method = 'none'
            self.last_disk_io = None
            logging.warning("Disk I/O monitoring is not available on this platform")
        
    def _test_disk_monitoring(self) -> bool:
        """
        Test if we can monitor the detected disk.
        
        Returns:
            bool: True if monitoring is possible, False otherwise
        """
        try:
            # Try to get a single reading from iostat
            subprocess.check_output(['iostat', '-I', self.disk_name, '1', '1'], 
                                 stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as e:
            logging.warning(f"Disk monitoring test failed: {e.output.decode() if e.output else str(e)}")
            return False
        except Exception as e:
            logging.warning(f"Disk monitoring test failed: {e}")
            return False
        
    def _get_macos_disk_name(self) -> Optional[str]:
        """
        Get the main disk name on macOS.
        
        Returns:
            Optional[str]: Disk name if found, None otherwise
        """
        try:
            # First try to get the boot disk
            boot_disk = subprocess.check_output(['diskutil', 'info', '/']).decode()
            for line in boot_disk.split('\n'):
                if 'Device Identifier' in line:
                    disk_id = line.split(':')[-1].strip()
                    # Try to get the parent disk if this is a partition
                    if 's' in disk_id:
                        parent_disk = disk_id.split('s')[0]
                        # Test if we can monitor the parent disk
                        if self._test_disk_name(parent_disk):
                            return parent_disk
                    return disk_id
            
            # Fallback to listing all disks
            disk_list = subprocess.check_output(['diskutil', 'list']).decode()
            for line in disk_list.split('\n'):
                if 'disk0' in line:
                    disk_id = line.split()[0]
                    if self._test_disk_name(disk_id):
                        return disk_id
        except Exception as e:
            logging.warning(f"Failed to get disk name: {e}")
        return None
        
    def _test_disk_name(self, disk_name: str) -> bool:
        """
        Test if a disk name can be monitored.
        
        Args:
            disk_name (str): Disk name to test
            
        Returns:
            bool: True if the disk can be monitored, False otherwise
        """
        try:
            subprocess.check_output(['iostat', '-I', disk_name, '1', '1'], 
                                 stderr=subprocess.STDOUT)
            return True
        except:
            return False
        
    def _get_macos_disk_io(self) -> Dict:
        """
        Get disk I/O statistics on macOS using iostat.
        
        Returns:
            Dict: Dictionary containing read and write bytes
        """
        if not self.disk_name:
            return {'read_bytes': 0, 'write_bytes': 0}
            
        try:
            # Get I/O statistics for the specific disk
            iostat = subprocess.check_output(['iostat', '-I', self.disk_name, '1', '1']).decode()
            lines = iostat.split('\n')
            
            # Find the line with actual data (skip headers)
            data_line = None
            for line in lines:
                if line.strip() and not line.startswith('Device'):
                    data_line = line
                    break
                    
            if data_line:
                values = data_line.split()
                if len(values) >= 3:  # At least KB_read and KB_wrtn should be present
                    return {
                        'read_bytes': float(values[1]) * 1024,  # KB_read to bytes
                        'write_bytes': float(values[2]) * 1024   # KB_wrtn to bytes
                    }
        except Exception as e:
            logging.warning(f"Failed to get disk I/O statistics: {e}")
        return {'read_bytes': 0, 'write_bytes': 0}
        
    def _get_cpu_metrics(self) -> Dict:
        """
        Get CPU metrics accounting for parallel processing.
        
        Returns:
            Dict: Dictionary containing CPU metrics
        """
        # Get CPU usage for main process
        main_cpu_percent = self.process.cpu_percent()
        
        # Get CPU usage for joblib worker processes
        joblib_cpu_percent = 0
        current_processes = set()
        active_workers = 0
        
        # Find all child processes
        try:
            for child in self.process.children(recursive=True):
                current_processes.add(child.pid)
                try:
                    worker_cpu = child.cpu_percent()
                    joblib_cpu_percent += worker_cpu
                    if worker_cpu > 0:  # Count only active workers
                        active_workers += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Update joblib processes
        self.joblib_processes = current_processes
        
        # Calculate total CPU usage
        total_cpu_percent = main_cpu_percent + joblib_cpu_percent
        
        # Calculate effective number of cores being used
        if self.parallel_mode:
            # For joblib, each worker can use up to 100% of a core
            # We have NCPU workers, and we track how many are actually active
            cores_used = active_workers
            effective_cores = self.ncpu
        else:
            cores_used = 1
            effective_cores = 1
            
        # Calculate CPU percentage per core
        cpu_percent_per_core = total_cpu_percent / effective_cores if effective_cores > 0 else 0
        
        return {
            'cpu_percent': total_cpu_percent,
            'cpu_percent_per_core': cpu_percent_per_core,
            'cores_used': cores_used,
            'max_cores_available': effective_cores
        }
        
    def start_module(self, module_name: str) -> None:
        """
        Start monitoring a new module.
        
        Args:
            module_name (str): Name of the module being monitored
        """
        self.current_module = module_name
        self.module_start_time = time.time()
        
        if self.io_method == 'psutil':
            self.last_disk_io = self.process.io_counters()
        elif self.io_method == 'iostat':
            self.last_disk_io = self._get_macos_disk_io()
        
    def record_metrics(self) -> None:
        """
        Record current performance metrics.
        """
        current_time = time.time()
        duration = current_time - self.module_start_time
        
        # Get CPU metrics
        cpu_metrics = self._get_cpu_metrics()
        
        # Get memory usage
        memory_info = self.process.memory_info()
        memory_usage = memory_info.rss / 1024 / 1024  # Convert to MB
        
        # Get disk I/O based on platform
        if self.io_method == 'psutil':
            current_io = self.process.io_counters()
            disk_read = (current_io.read_bytes - self.last_disk_io.read_bytes) / 1024 / 1024  # Convert to MB
            disk_write = (current_io.write_bytes - self.last_disk_io.write_bytes) / 1024 / 1024  # Convert to MB
            self.last_disk_io = current_io
        elif self.io_method == 'iostat':
            current_io = self._get_macos_disk_io()
            disk_read = (current_io['read_bytes'] - self.last_disk_io['read_bytes']) / 1024 / 1024  # Convert to MB
            disk_write = (current_io['write_bytes'] - self.last_disk_io['write_bytes']) / 1024 / 1024  # Convert to MB
            self.last_disk_io = current_io
        else:
            disk_read = 0
            disk_write = 0
        
        # Record metrics
        self.metrics['module'].append(self.current_module)
        self.metrics['cpu_percent'].append(cpu_metrics['cpu_percent'])
        self.metrics['cpu_percent_per_core'].append(cpu_metrics['cpu_percent_per_core'])
        self.metrics['cores_used'].append(cpu_metrics['cores_used'])
        self.metrics['max_cores_available'].append(cpu_metrics['max_cores_available'])
        self.metrics['memory_usage'].append(memory_usage)
        self.metrics['disk_read'].append(disk_read)
        self.metrics['disk_write'].append(disk_write)
        self.metrics['duration'].append(duration)
        self.metrics['timestamp'].append(datetime.now().isoformat())
        
    def end_module(self) -> None:
        """
        End monitoring of current module and record final metrics.
        """
        self.record_metrics()
        
    def save_benchmark(self) -> None:
        """
        Save performance metrics to a CSV file.
        """
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.benchmark_file, index=False)
        logging.info(f"Performance benchmark saved to {self.benchmark_file}")
        
    def get_summary(self) -> Dict:
        """
        Generate a summary of performance metrics.
        
        Returns:
            Dict: Summary statistics for each metric
        """
        df = pd.DataFrame(self.metrics)
        summary = {
            'total_duration': time.time() - self.start_time,
            'modules': {},
            'platform': platform.platform(),
            'io_method': self.io_method,
            'parallel_mode': self.parallel_mode,
            'total_cores': self.total_cores,
            'num_threads': self.num_threads
        }
        
        for module in df['module'].unique():
            module_data = df[df['module'] == module]
            summary['modules'][module] = {
                'avg_cpu_percent': module_data['cpu_percent'].mean(),
                'max_cpu_percent': module_data['cpu_percent'].max(),
                'avg_cpu_percent_per_core': module_data['cpu_percent_per_core'].mean(),
                'max_cpu_percent_per_core': module_data['cpu_percent_per_core'].max(),
                'max_cores_used': module_data['cores_used'].max(),
                'max_cores_available': module_data['max_cores_available'].max(),
                'avg_memory_usage_mb': module_data['memory_usage'].mean(),
                'max_memory_usage_mb': module_data['memory_usage'].max(),
                'duration_seconds': module_data['duration'].sum()
            }
            
            # Include disk I/O metrics if available
            if self.io_method != 'none':
                summary['modules'][module].update({
                    'total_disk_read_mb': module_data['disk_read'].sum(),
                    'total_disk_write_mb': module_data['disk_write'].sum()
                })
            
        return summary

def setup_performance_monitor(config: Dict) -> PerformanceMonitor:
    """
    Set up and return a performance monitor instance.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        PerformanceMonitor: Initialized performance monitor
    """
    return PerformanceMonitor(config) 