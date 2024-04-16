# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import socket

import numpy as np

def recv_data(sock, data):
    """Fetches binary data from socket."""
    blen = data.itemsize * data.size
    buf = np.zeros(blen, np.byte)

    bpos = 0
    while bpos < blen:
        timeout = False
        try:
            bpart = 1
            bpart = sock.recv_into(buf[bpos:], blen - bpos)  # type: ignore[arg-type]
        except socket.timeout:
            print(" @SOCKET:   Timeout in status recvall, trying again!")
            timeout = True
        if not timeout and bpart == 0:
            raise RuntimeError("Socket disconnected!")
        bpos += bpart

    if not isinstance(data, np.ndarray):
        return np.frombuffer(buf[0:blen], data.dtype)[0]

    return np.frombuffer(buf[0:blen], data.dtype).reshape(data.shape)


def send_data(sock, data):
    """Sends binary data to socket."""
    if not isinstance(data, np.ndarray):
        data = np.array([data], data.dtype)
    buf = data.tobytes()
    sock.send(buf)
