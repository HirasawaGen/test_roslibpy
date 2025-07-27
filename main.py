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
