import time
from contextlib import contextmanager
import roslibpy
from roslibpy import Ros

'''
TODO:
1. add some comments to explain the code.
2. user variable `subscribed_topics` to mark subscribed topics, maybe not a good idea
3. `roslibpy.Topic` object maybe initialized multiple times, no bug, but it's not efficient.
4. maybe `client.terminate()` is already unsubscribed the topics, maybe I don't need to unsubscribe by myself.
'''

@contextmanager
def ros_context(host: str, port: int):
    client = Ros(host=host, port=port)
    client.run()

    subscribed_topics: set[str] = set()
    messages: dict[str, dict] = {}

    callback_factory = lambda topic_name: (lambda msg: messages.__setitem__(topic_name, {'msg': msg, 'updated': True}))
    # callback_factory = lambda topic_name: (lambda msg: print(f"Received message on topic '{topic_name}': {msg}"))

    def _get_topic(topic_name: str):
        '''
        if topic type is 'unknown', the service call '/rosapi/topic_type' will stuck in for unknow reason.
        no warn no error, just stuck.
        this is an awful error.
        so I hard code the topic type for cmd_vel and scan here. avoid they stuck.
        '''
        topic_type: str
        match topic_name:
            case _ if 'cmd_vel' in topic_name:
                topic_type = 'geometry_msgs/Twist'
            case _ if'scan' in topic_name:
                topic_type ='sensor_msgs/LaserScan'
            case _:
                topic_type_service = roslibpy.Service(client, "/rosapi/topic_type", "rosapi/TopicType")
                result = topic_type_service.call(roslibpy.ServiceRequest({'topic': topic_name}))
                topic_type = result['type']
        return roslibpy.Topic(
            client,
            topic_name,
            topic_type
        )

    def _topics__add__(self, topic_name: str):
        nonlocal subscribed_topics
        if topic_name in subscribed_topics:
            return self
        topic = _get_topic(topic_name)
        topic.subscribe(callback_factory(topic_name))
        subscribed_topics |= {topic_name}
        return self

    def _topics__sub__(self, topic_name: str):
        nonlocal subscribed_topics
        if topic_name not in subscribed_topics:
            return self
        topic = _get_topic(topic_name)
        topic.unsubscribe()
        subscribed_topics -= {topic_name}
        return self

    def _topics__getitem__(self, topic_name: str):
        '''
        TODO
        '''
        topic = _get_topic(topic_name)
        if topic_name in subscribed_topics:  # longtime subscribed topics
            return messages[topic_name]            
        else:  # once subscribed topics
            topic.subscribe(callback_factory(topic_name))
            start_time = time.time()
            while messages.get(topic_name) is None:
                time.sleep(0.1)
                if time.time() - start_time > 10.0:
                    raise TimeoutError(f"Timeout waiting for '{topic_name}' to be published")
            msg = messages[topic_name]
            del msg['updated']
            del messages[topic_name]
            topic.unsubscribe()
            return msg

    def _topics__setitem__(self, topic_name: str, msg: dict):
        topic = _get_topic(topic_name)
        topic.publish(msg)

    def _get_service(service_name: str):
        service_type: str
        match service_name:
            case _:
                service_type_service = roslibpy.Service(client, "/rosapi/service_type", "rosapi/ServiceType")
                result = service_type_service.call(roslibpy.ServiceRequest({'service': service_name}))
                service_type = result['type']
        return roslibpy.Service(
            client,
            service_name,
            service_type
        )
    
    def _services__getitem__(self, service_name: str):
        service = _get_service(service_name)
        return type('RosServiceHandler', (object,), {
            '__lshift__': lambda self, req: service.call(roslibpy.ServiceRequest(req)),
        })()

    def _services__setitem__(self, service_name: str, value):
        service = _get_service(service_name)
        service.call(roslibpy.ServiceRequest(value))


    topics = type('RosTopics', (object,), {
        '__add__': _topics__add__,  # TODO: bug in using, we need to fix it
        '__sub__': _topics__sub__,
        '__getitem__': _topics__getitem__,
        '__setitem__': _topics__setitem__,
        '__iter__': lambda self: iter(map(lambda topic_name: (topic_name, _get_topic(topic_name)), client.get_topics())),
    })()  # a singleton object

    services = type('RosService', (object,), {
        '__getitem__': _services__getitem__,
        '__setitem__': _services__setitem__,
        '__iter__': lambda self: iter(map(lambda service_name: (service_name, _get_service(service_name)), client.get_services())),
    })()

    
    yield client, topics, services

    for topic_name in subscribed_topics:
        topic = _get_topic(topic_name)
        topic.unsubscribe()

    client.terminate()

    
