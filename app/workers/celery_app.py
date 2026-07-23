from celery import Celery
from celery.schedules import crontab

from app.core.redis_settings import redis_settings


celery_app = Celery(
    "alloviewer",
    broker=redis_settings.redis_url,
    backend=redis_settings.redis_url,
    include=[
        "app.workers.tasks_fcxm",
        "app.workers.tasks_image",
        "app.workers.tasks_cleanup",
    ],
)

celery_app.conf.task_routes = {
    "app.workers.tasks_fcxm.run_fcxm_job_task": {"queue": "cpu"},
    "app.workers.tasks_image.run_image_analysis_task": {"queue": "image"},
    "app.workers.tasks_cleanup.cleanup_runtime_data_task": {
        "queue": "maintenance"
    },
    "app.workers.tasks_cleanup.cleanup_low_disk_space_task": {
        "queue": "maintenance"
    },
}

celery_app.conf.task_track_started = True
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_acks_late = True

celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]

celery_app.conf.timezone = "Europe/Berlin"

celery_app.conf.beat_schedule = {
    "cleanup-runtime-data-daily": {
        "task": "app.workers.tasks_cleanup.cleanup_runtime_data_task",
        "schedule": crontab(hour=3, minute=0),
    },
    "cleanup-low-disk-space-every-15-minutes": {
        "task": "app.workers.tasks_cleanup.cleanup_low_disk_space_task",
        "schedule": crontab(minute="*/15"),
    },
}
