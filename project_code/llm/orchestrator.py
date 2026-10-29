# project_code/llm/orchestrator.py
"""
Plain (non-agent) orchestrator: answers operator questions using RAG over the
PDF manual (project_code/llm/main_llm_rag.py). No google.adk agents/tools are
used - this just calls run_rag directly against an OpenAI-compatible endpoint.
"""

import logging
import time

from openai import OpenAI

from project_code.app_config.settings import app_settings, log_app_settings
from project_code.llm.main_llm_rag import run_rag, build_chunks
from project_code.utils.logger_utils import log_boxed
from project_code.llm.qa_cache import QACache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)


class Orchestrator:
    """Owns the shared RAG resources (LLM client, manual chunks/index, QA cache)
    and exposes the two entry points used by callers:
      - run_rag_directly(question): plain RAG call, no cache.
      - run_single_question_cached(question): checks semantic QA cache first.
    """

    def __init__(self):
        self.llm_client = OpenAI(
            api_key=app_settings.llm.api_key,
            base_url=app_settings.llm.base_url,
        )

        self.manual_chunks = build_chunks(
            app_settings.database.manual_pdf_path,
            app_settings.rag,
            pdf_reader=app_settings.database.pdf_reader,
        )

        self.qa_cache = QACache(
            cache_dir=app_settings.qa_cache.cache_dir,
            embedding_model_name=app_settings.rag.embedding_model,
            similarity_threshold=app_settings.qa_cache.similarity_threshold,
        ) if app_settings.qa_cache.enabled else None

    # ------------------------------------------------------------------
    # Internal helper shared by both public entry points
    # ------------------------------------------------------------------
    def _run_rag(self, question: str):
        """Runs RAG directly over the operator manual (no cache)."""
        start_time = time.time()
        answer = run_rag(
            self.llm_client, self.manual_chunks, question,
            top_k=app_settings.rag.top_k,
            search_mode=app_settings.rag.search_mode,
            bm25_top_k=app_settings.rag.bm25_top_k,
            rrf_k=app_settings.rag.rrf_k,
        )
        end_time = time.time()
        logger.info("Elapsed time: %.2f seconds", end_time - start_time)
        return answer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_rag_directly(self, question: str):
        """Answers a question using main_llm_rag directly (no cache),
        driven entirely by the rag.* settings in conf.yaml."""
        start_qa = time.time()
        answer = self._run_rag(question)
        end_qa = time.time()
        log_boxed("Run RAG Directly", [f"{(end_qa - start_qa):.2f} sec {answer}"])
        return answer

    def run_single_question_cached(self, question: str):
        """Checks the semantic QA cache before calling the LLM; stores new answers."""
        if self.qa_cache is not None:
            cached_answer = self.qa_cache.lookup(question)
            if cached_answer is not None:
                logger.info("Serving answer from cache for: %s", question)
                return cached_answer

        answer = self._run_rag(question)

        if self.qa_cache is not None:
            self.qa_cache.add(question, answer)

        return answer

    # ------------------------------------------------------------------
    # Batch / interactive helpers (optional, kept for local testing)
    # ------------------------------------------------------------------
    def run_test_questions(self):
        l_questions = [
            "where is the drive en switch located ?",
            "where is the shooter led located ?",
            "where is the push buttun located ?",
            "What is the joystick used for ?",
            "How do I use the joystick ?",
            "What happens if I press the joystick ?",
        ]
        for q in l_questions:
            self.run_single_question_cached(q)

    def run_interactive(self):
        logger.info("\nIntelligent Operator Assistance - ready.")
        logger.info("Type your question and press Enter (type 'exit' or 'quit' to stop).\n")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                logger.info("\nExiting.")
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                logger.info("Exiting.")
                break

            answer = self.run_single_question_cached(question)
            if answer is not None:
                log_boxed("Assistant", [str(answer)])


def main():
    log_app_settings()
    orchestrator = Orchestrator()
    orchestrator.run_single_question_cached("first warmup question")
    orchestrator.run_single_question_cached("What is the joystick used for ?")
    orchestrator.run_single_question_cached("where is the drive en switch located ?")
    orchestrator.run_single_question_cached("where is the push buttun located ?")


if __name__ == "__main__":
    main()