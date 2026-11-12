# import os
# os.environ.setdefault("HF_HUB_OFFLINE", "1")
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from project_code.llm.orchestrator import Orchestrator
from project_code.utils.utils import print_version

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(funcName)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    print_version()

    orchestrator = Orchestrator()

    #l_questions = ["בריכבה ספורטיבית , אופני כביש , מהו מיקום רכיבה אופטימאלי לגובה 180 סמ ?"]
    #l_questions = ["מה האחראיות אינה מכסה ?"]
    l_questions = ["האם אופניים לילדים קטנים ניתן להתקין מושב לתינוק ?"]
    l_questions = ["האם אופניים לילדים קטנים ניתן להתקין מנשא ?"]
    for text in l_questions:
        orchestrator.run_rag_directly(text)
        orchestrator.run_single_question_cached(text)