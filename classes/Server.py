from typing import List
from Client.py import Client
from Method.py import Method


class Server:
    Methods : Method
    Clients: List[Client]
