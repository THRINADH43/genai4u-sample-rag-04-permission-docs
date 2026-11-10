"""HTTP API. The caller says who they are with X-User; their groups come from users.json.
(In a real system X-User would come from a login; here it is trusted, as the brief allows.)

POST   /documents        {"id","text","groups"}   editors only -> 202 {"job"}
DELETE /documents/<id>                              editors only -> 202 {"job"}
GET    /jobs/<job>                                  -> 200 job status
POST   /ask              {"question"}               -> text/event-stream of the answer
"""

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from docservice.answer import FakeLLM, GroqLLM, answer_stream
from docservice.ingest import Ingestor
from docservice.search import hybrid_search
from docservice.store import DocumentStore

log = logging.getLogger("docservice")


def load_users(path="users.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_handler(store, ingestor, users, llm):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _user(self):
            return users.get(self.headers.get("X-User", ""))

        def _body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")

        def do_GET(self):
            if self._user() is None:
                return self._json(401, {"error": "unknown X-User"})
            if self.path.startswith("/jobs/"):
                job = ingestor.jobs.get(self.path[6:])
                return self._json(200, job) if job else self._json(404, {"error": "no such job"})
            self._json(404, {"error": "not found"})

        def do_DELETE(self):
            user = self._user()
            if user is None:
                return self._json(401, {"error": "unknown X-User"})
            if not user.get("editor"):
                return self._json(403, {"error": "only editors can change documents"})
            if self.path.startswith("/documents/"):
                return self._json(202, ingestor.submit("delete", self.path[len("/documents/"):]))
            self._json(404, {"error": "not found"})

        def do_POST(self):
            user = self._user()
            if user is None:
                return self._json(401, {"error": "unknown X-User"})
            try:
                body = self._body()
            except ValueError:
                return self._json(400, {"error": "body must be JSON"})
            if self.path == "/documents":
                if not user.get("editor"):
                    return self._json(403, {"error": "only editors can change documents"})
                if not body.get("id") or not body.get("text") or not body.get("groups"):
                    return self._json(400, {"error": "id, text and groups are required"})
                return self._json(202, ingestor.submit("upsert", body["id"], body["text"], body["groups"]))
            if self.path == "/ask":
                question = (body.get("question") or "").strip()
                if not question:
                    return self._json(400, {"error": "question is required"})
                hits = hybrid_search(store, user["groups"], question)
                log.info("user=%s question_words=%d hits=%s", self.headers["X-User"], len(question.split()),
                         [c.cite for c, _ in hits])
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for piece in answer_stream(question, hits, llm):
                    self.wfile.write(f"data: {json.dumps(piece)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self._json(404, {"error": "not found"})

        def log_message(self, *args):
            pass

    return Handler


def main():
    logging.basicConfig(level=logging.INFO)
    store = DocumentStore()
    ingestor = Ingestor(store)
    for name in sorted(os.listdir("seed")):
        doc = json.load(open(os.path.join("seed", name)))
        ingestor.submit("upsert", doc["id"], doc["text"], doc["groups"])
    llm = GroqLLM() if os.environ.get("GROQ_API_KEY") else FakeLLM()
    port = int(os.environ.get("PORT", "8070"))
    print(f"listening on http://127.0.0.1:{port} with {type(llm).__name__}")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(store, ingestor, load_users(), llm)).serve_forever()


if __name__ == "__main__":
    main()
