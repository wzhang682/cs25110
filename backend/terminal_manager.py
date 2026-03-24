#backend/terminal_manager.py
import subprocess
import threading
import queue
import sys
from pathlib import Path

class TerminalManager:
    def __init__(self):
        self.process = None
        self.output_queue = queue.Queue()

    def start(self):
        project_root = Path(__file__).parent.parent

        self.process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",       
            errors="replace",        
            bufsize=1  # line buffered
        )

        threading.Thread(
            target=self._read_output,
            daemon=True
        ).start()

    def _read_output(self):
        for line in self.process.stdout:
            self.output_queue.put(line)

    def send_input(self, text: str):
        if self.process and self.process.stdin:
            self.process.stdin.write(text + "\n")
            self.process.stdin.flush()

    def get_output(self):
        outputs = []
        while not self.output_queue.empty():
            outputs.append(self.output_queue.get())
        return outputs

    def is_finished(self):
        return self.process.poll() is not None