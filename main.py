from src.audio.listener import Listener
import threading
import time

def main():
    def run_listener():
        listener = Listener()
        listener.listen()

    listener_thread = threading.Thread(target=run_listener)
    listener_thread.start()

    while True:
        time.sleep(1)
        pass

if __name__ == "__main__":
    main()