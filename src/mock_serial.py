"""
mock_serial.py
==============
A drop-in replacement for pyserial that works entirely in-process.

Instead of bytes travelling over a real COM port, they travel through a
thread-safe queue in memory. No drivers, no virtual COM ports required.

How it works:
  - MockSerial has two sides: a WRITE side (sensor) and a READ side (student)
  - A shared queue sits between them
  - Sensor calls .write(bytes) -> bytes go into the queue
  - Student calls .read(n)    -> bytes come out of the queue

The API is intentionally identical to pyserial's Serial class for the
methods the testbench uses, so the sensor classes and student template
don't need any changes — just swap the transport.

Usage:
    from mock_serial import MockSerialPair

    write_end, read_end = MockSerialPair.create()

    # Pass write_end to the sensor, read_end to the student parser
    sensor = ECGSensor(serial_override=write_end, ...)
    parser = PacketParser(read_end, protocol)
"""

import queue
import threading
import time


class MockSerial:
    """
    One end of a virtual serial connection backed by a queue.
    Mimics the pyserial Serial interface for read/write/is_open/close.
    """

    def __init__(self, tx_queue: queue.Queue, name: str = "MockSerial"):
        self._queue   = tx_queue
        self.name     = name        # displayed in print statements
        self.port     = name        # pyserial compat attribute
        self.baudrate = 115200      # cosmetic only — no real baud limiting
        self.is_open  = True
        self._lock    = threading.Lock()

    # ── Write side ────────────────────────────────────────────────────────
    def write(self, data: bytes) -> int:
        """Push bytes into the queue one byte at a time."""
        if not self.is_open:
            raise IOError("MockSerial port is closed.")
        for byte in data:
            self._queue.put(bytes([byte]))
        return len(data)

    # ── Read side ─────────────────────────────────────────────────────────
    def read(self, size: int = 1) -> bytes:
        """
        Pull exactly `size` bytes from the queue.
        Blocks until all bytes are available (mirrors pyserial blocking read).
        """
        if not self.is_open:
            return b''
        result = b''
        while len(result) < size:
            try:
                result += self._queue.get(timeout=2.0)
            except queue.Empty:
                if not self.is_open:
                    return result   # port closed while waiting
        return result

    def read_until(self, expected: bytes = b'\n', size: int = None) -> bytes:
        """Read until the expected byte sequence is found (pyserial compat)."""
        result = b''
        while True:
            byte = self.read(1)
            if not byte:
                break
            result += byte
            if result.endswith(expected):
                break
            if size and len(result) >= size:
                break
        return result

    # ── Control ───────────────────────────────────────────────────────────
    def close(self):
        self.is_open = False

    def flush(self):
        pass   # no-op — queue is already synchronous

    @property
    def in_waiting(self) -> int:
        """Approximate number of bytes waiting to be read."""
        return self._queue.qsize()

    def __repr__(self):
        return f"MockSerial(name={self.name!r}, is_open={self.is_open})"


class MockSerialPair:
    """
    Factory that creates a matched write/read pair sharing one queue.

        write_end, read_end = MockSerialPair.create("ECG")

    Pass write_end to the sensor (it calls .write()).
    Pass read_end  to the parser  (it calls .read()).
    """

    @staticmethod
    def create(name: str = "Sensor") -> tuple:
        """
        Returns (write_end, read_end) — two MockSerial objects sharing a queue.
        """
        shared_queue = queue.Queue()
        write_end = MockSerial(shared_queue, name=f"{name}:TX")
        read_end  = MockSerial(shared_queue, name=f"{name}:RX")
        return write_end, read_end
