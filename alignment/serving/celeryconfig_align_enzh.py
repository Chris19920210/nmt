from kombu import Exchange, Queue


exchange = Exchange("tasks_align_enzh")
exchange.durable = False
queue = Queue("tasks_align_enzh", exchange, routing_key="tasks_align_enzh")
queue.durable = False
queue.no_ack = True

CELERY_QUEUES = (
   queue,
)

CELERY_ROUTES = {
    'tasks_align_enzh.alignment': {"queue": "tasks_align_enzh", "routing_key": "tasks_align_enzh"},
}
