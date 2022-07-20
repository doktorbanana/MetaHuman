from pythonosc import udp_client
import argparse

class Communicator:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.client = self.connect()

    def connect(self):
        self.parser.add_argument("--ip", default="127.0.0.1")
        self.parser.add_argument("--port", type=int, default=5005)
        args = self.parser.parse_args()
        return udp_client.SimpleUDPClient(args.ip, args.port)

    def send(self, address, msg):
        self.client.send_message(address, msg)

