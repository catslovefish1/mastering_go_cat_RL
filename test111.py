import os, signal, faulthandler, time
faulthandler.enable()
faulthandler.register(signal.SIGUSR1)               # pkill -USR1 … to dump stacks
faulthandler.dump_traceback_later(10, repeat=True)  # auto-dump every 10s if stuck
print(f"[PID {os.getpid()}] test111 start", flush=True)
