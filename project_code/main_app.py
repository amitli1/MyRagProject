import logging
from project_code.llm.orchestrator import Orchestrator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(funcName)s] %(message)s")
    logger = logging.getLogger(__name__)

    orchestrator = Orchestrator()

    l_questions = ["מהו לחץ האוויר המומלץ למילוי בגלגלים ?"]
    for text in l_questions:
        orchestrator.run_rag_directly(text)
        orchestrator.run_single_question_cached(text)