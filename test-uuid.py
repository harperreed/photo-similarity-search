import socket
import uuid
import logging

host_name = socket.gethostname()
unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, host_name + str(uuid.getnode()))
print(f"Running on machine ID: {unique_id}")
