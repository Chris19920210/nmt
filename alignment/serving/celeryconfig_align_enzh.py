from kombu import Exchange, Queue


CELERY_QUEUES = (
    Queue("tasks_align_enzh", Exchange("tasks_align_enzh"), routing_key="tasks_align_enzh"),
)

CELERY_ROUTES = {
    'tasks_align_enzh.alignment': {"queue": "tasks_align_enzh", "routing_key": "tasks_align_enzh"},
}
