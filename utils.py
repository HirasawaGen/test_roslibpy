import time
from contextlib import contextmanager
import roslibpy
from roslibpy import Ros


@contextmanager
def ros_context(host: str, port: int):
    client = Ros(host=host, port=port)
    client.run()

    subscribed_topics: set[str] = set()
    messages: dict[str, dict] = {}

    callback_factory = lambda topic_name: (lambda msg: messages.__setitem__(topic_name, msg))
    # callback_factory = lambda topic_name: (lambda msg: print(f"Received message on topic '{topic_name}': {msg}"))

    def _get_topic(topic_name: str):
        return roslibpy.Topic(
            client,
            topic_name,
            client.get_topic_type(topic_name)
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
        topic = _get_topic(topic_name)
        if topic_name not in subscribed_topics:
            topic.subscribe(callback_factory(topic_name))
        start_time = time.time()
        while messages.get(topic_name) is None:
            time.sleep(0.1)
            if time.time() - start_time > 10.0:
                raise TimeoutError(f"Timeout waiting for '{topic_name}' to be published")
        msg = messages[topic_name]
        # messages[topic_name] = None
        if topic_name not in subscribed_topics:
            topic.unsubscribe()
        return msg

    def _topics__setitem__(self, topic_name: str, msg: dict):
        topic = _get_topic(topic_name)
        topic.publish(msg)

    def _get_service(service_name: str):
        return roslibpy.Service(
            client,
            service_name,
            client.get_service_type(service_name)
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

    client.terminate()

    for topic_name in subscribed_topics:
        topic = _get_topic(topic_name)
        topic.unsubscribe()

