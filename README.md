define to many attributes will cause difficult to use.

so I use operator overload like [] and << to make it easy to use.

# Ros Server Connection

first, how to connect to ros server?
```python
from utils import ros_context


HOST = 'localhost'  # change to your ros master ip
PORT = 9090  # change to your ros master port

def main():
    with ros_context(HOST, PORT) as (client, topics, services):
        print('Connected to ROS server successfully.')
        """
        write your codes here
        """
    print('ROS connection closed.')

if __name__ == '__main__':
    main()

```

if you need your script run in ros context the whole time, maybe you can use this way:

(this way is also useful for other context managers, such as database connection, etc.)
```python
from utils import ros_context
import atexit


manager = ros_context('localhost', 9090)
atexit.register(lambda: manager.__exit__(None, None, None))
client, topics, services = manager.__enter__()


def main():
    pass
    """
    write your codes here
    """

if __name__ == '__main__':
    main()

```

# Publish and Subscribe Topic

There are two ways to publish topics:

one way is just like 'wait_for_message' function

the code will subscribe for the topics, when get the message, immediately unsubscribe it. like this:

```python
from utils import ros_context
import atexit


manager = ros_context('localhost', 9090)
atexit.register(lambda: manager.__exit__(None, None, None))
client, topics, services = manager.__enter__()

print(topics['/rosout'])  # subcribe to /rosout topic

""" expected output:
{'msg': {'header': {'seq': 21, 'stamp': {'secs': 1753524442, 'nsecs': 323735952}, 'frame_id': ''}, 'level': 2, 'name': '/rosbridge_websocket', 'msg': '[Client 4] Subscribed to /rosout', 'file': 'protocol.py', 'function': 'RosbridgeProtocol.log', 'line': 403, 'topics': ['/client_count', '/connected_clients', '/rosout']}}
ROS client terminated
"""

```

the second way is subcribe for the topic for a very long time use + and - operator. like this:

```python
from utils import ros_context
import atexit


manager = ros_context('localhost', 9090)
atexit.register(lambda: manager.__exit__(None, None, None))
client, topics, services = manager.__enter__()


# first way:
print(topics['/cmd_vel'])
print(topics['/cmd_vel'])
print(topics['/cmd_vel'])
# In this way, the topic repeatedly subscribed and unsubscribed for three times.

#second way:
topics += '/cmd_vel'
print(topics['/cmd_vel'])
print(topics['/cmd_vel'])
print(topics['/cmd_vel'])
topics -= '/cmd_vel'
# In this way, the topic is subscribed when "topics += '/cmd_vel'" is executed, and unsubscribed when "topics -= '/cmd_vel'" is executed.

```
if this topic message is frequently accesed, you can use the second way to subscribe it.

if not frequently accesed, you can use the first way to subscribe it.


# Call Service
```python
from utils import ros_context
import atexit


manager = ros_context('localhost', 9090)
atexit.register(lambda: manager.__exit__(None, None, None))
client, topics, services = manager.__enter__()

req = {'topic': '/rosout'}
resp = services['/rosapi/topic_type'] << req
print(resp)

""" expected output:
{'type': 'rosgraph_msgs/Log'}
ROS client terminated
"""

```

