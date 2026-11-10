"""Background ingestion: requests are queued and one worker thread applies them, so the
API answers at once and big documents never block questions."""

import logging
import queue
import threading
import uuid

log = logging.getLogger(__name__)


class Ingestor:
    def __init__(self, store):
        self.store = store
        self.jobs = {}
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, action, doc_id, text=None, groups=None):
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {"id": job_id, "action": action, "doc_id": doc_id, "status": "queued"}
        self._q.put((job_id, action, doc_id, text, groups))
        return self.jobs[job_id]

    def _run(self):
        while True:
            job_id, action, doc_id, text, groups = self._q.get()
            try:
                if action == "upsert":
                    if not text or not groups:
                        raise ValueError("text and groups are required")
                    self.store.upsert(doc_id, text, groups)
                elif action == "delete":
                    if not self.store.delete(doc_id):
                        raise ValueError("no such document")
                self.jobs[job_id]["status"] = "done"
            except Exception as err:
                self.jobs[job_id].update(status="failed", error=str(err))
                log.warning("ingest %s failed: %s", job_id, err)
            finally:
                self._q.task_done()

    def wait(self):
        """Block until the queue is empty (used by tests and the seed script)."""
        self._q.join()
