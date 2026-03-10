"""
DermIQ — Skin Disease RAG Intelligence Engine
LangGraph 3-node pipeline: Retrieve → BuildContext → Generate
Embeddings : sentence-transformers (free, local, no extra langchain package)
LLM        : Cohere command-r-plus via langchain-cohere
Vector DB  : FAISS (local, no server needed)
"""

import os
import time
import logging
import operator
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional

# ── Must be set BEFORE any tensorflow/transformers import ────────────────────
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from dotenv import load_dotenv

# ── All imports from langchain_core 0.2.x — no deprecated shims ─────────────
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings          # abstract base class
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, END

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("dermiq-rag")


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM EMBEDDINGS
#  Wraps sentence-transformers directly — zero dependency on langchain-huggingface
#  which has an irreconcilable version conflict with langchain-core 0.2.x
# ─────────────────────────────────────────────────────────────────────────────
class SentenceTransformerEmbeddings(Embeddings):
    """
    Lightweight LangChain-compatible embeddings using sentence-transformers.
    Implements the Embeddings interface so FAISS and any retriever works natively.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("Embedding model ready")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents — used during indexing."""
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string — used during retrieval."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:        str
    retrieved_docs:  Annotated[List[Document], operator.add]
    context:         str
    answer:          str
    sources:         List[str]
    confidence:      str
    processing_time: float


# ─────────────────────────────────────────────────────────────────────────────
#  DERMIQ RAG ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class DermIQEngine:
    """
    Production-grade RAG engine for skin disease classification.
    A 3-node LangGraph state machine makes every step explicit and auditable.

    Flow:
        [retrieve] ─► [build_context] ─► [generate] ─► END
    """

    EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_PATH = "dermiq_vectorstore"
    DOCS_DIR          = "docs/skin_diseases"

    SYSTEM_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are DermIQ, an expert dermatology clinical assistant with deep knowledge "
            "in skin disease classification, diagnosis, and treatment protocols. "
            "You provide accurate, evidence-based medical information.\n\n"
            "RETRIEVED CLINICAL KNOWLEDGE:\n"
            "─────────────────────────────────────────────────────────────────────\n"
            "{context}\n"
            "─────────────────────────────────────────────────────────────────────\n\n"
            "PATIENT/USER QUESTION:\n{question}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer using ONLY the provided clinical knowledge above\n"
            "- Structure your response clearly with relevant sections\n"
            "- Include classification, symptoms, causes, and treatment if asked\n"
            "- If the answer is not fully covered in the context, state that clearly\n"
            "- Always recommend consulting a board-certified dermatologist for "
            "personal medical decisions\n"
            "- Provide ICD-10 codes when relevant\n"
            "- Be thorough but accessible — explain medical terms briefly\n\n"
            "CLINICAL RESPONSE:"
        ),
    )

    def __init__(self) -> None:
        self._llm:          Optional[ChatCohere]                    = None
        self._vectorstore:  Optional[FAISS]                         = None
        self._embeddings:   Optional[SentenceTransformerEmbeddings] = None
        self._graph                                                  = None
        self._is_ready:     bool                                     = False

    # ── Lazy initializers ─────────────────────────────────────────────────────

    def _get_embeddings(self) -> SentenceTransformerEmbeddings:
        if self._embeddings is None:
            self._embeddings = SentenceTransformerEmbeddings(self.EMBED_MODEL)
        return self._embeddings

    def _get_llm(self) -> ChatCohere:
        if self._llm is None:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "COHERE_API_KEY is not set. "
                    "Add it to your .env file: COHERE_API_KEY=your_key_here"
                )
            self._llm = ChatCohere(
                cohere_api_key=api_key,
                model="command-r-plus-08-2024",
                temperature=0.2,
                max_tokens=1500,
            )
            logger.info("Cohere LLM (command-r-plus) initialized")
        return self._llm

    # ── Document ingestion ────────────────────────────────────────────────────

    def _load_and_chunk_documents(self) -> List[Document]:
        """Read every .txt file in DOCS_DIR and split into overlapping chunks."""
        docs_path = Path(self.DOCS_DIR)
        if not docs_path.exists():
            raise FileNotFoundError(
                f"Documents directory not found: '{self.DOCS_DIR}'. "
                "Make sure the docs/skin_diseases folder exists next to server.py."
            )

        txt_files = list(docs_path.glob("*.txt"))
        if not txt_files:
            raise ValueError(
                f"No .txt files found in '{self.DOCS_DIR}'. "
                "Add skin disease text files to that folder."
            )

        raw_docs: List[Document] = []
        for fp in sorted(txt_files):
            content = fp.read_text(encoding="utf-8")
            raw_docs.append(Document(
                page_content=content,
                metadata={"source": fp.name, "disease_file": fp.stem},
            ))
            logger.info(f"  Loaded: {fp.name}  ({len(content):,} chars)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ".", " "],
            length_function=len,
        )
        chunks = splitter.split_documents(raw_docs)
        logger.info(f"Chunking complete: {len(chunks)} chunks from {len(raw_docs)} files")
        return chunks

    # ── Vector store ──────────────────────────────────────────────────────────

    def build_vectorstore(self, force_rebuild: bool = False) -> None:
        """Build a new FAISS index or load an existing one from disk."""
        store_path = Path(self.VECTOR_STORE_PATH)
        embeddings = self._get_embeddings()

        if store_path.exists() and not force_rebuild:
            logger.info("Loading existing FAISS index from disk...")
            self._vectorstore = FAISS.load_local(
                str(store_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                f"FAISS index loaded — {self._vectorstore.index.ntotal} vectors"
            )
        else:
            logger.info("Building FAISS index from documents...")
            chunks = self._load_and_chunk_documents()
            self._vectorstore = FAISS.from_documents(chunks, embeddings)
            self._vectorstore.save_local(str(store_path))
            logger.info(
                f"FAISS index built and saved — "
                f"{self._vectorstore.index.ntotal} vectors"
            )

    # ── LangGraph nodes ───────────────────────────────────────────────────────

    def _node_retrieve(self, state: RAGState) -> dict:
        """
        Node 1 — Retrieve
        MMR search (Maximal Marginal Relevance) for diverse, relevant chunks.
        """
        query = state["question"]
        retriever = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.65},
        )
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} docs for: '{query[:70]}'")
        return {"retrieved_docs": docs}

    def _node_build_context(self, state: RAGState) -> dict:
        """
        Node 2 — BuildContext
        Deduplicate sources, assemble numbered reference blocks,
        and calculate a simple confidence score.
        """
        docs = state["retrieved_docs"]

        if not docs:
            return {
                "context":    "No relevant information found in the knowledge base.",
                "sources":    [],
                "confidence": "LOW",
            }

        sources_seen: set  = set()
        blocks:       List[str] = []
        sources:      List[str] = []

        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "Unknown")
            blocks.append(
                f"[Ref {i} | Source: {src}]\n{doc.page_content.strip()}"
            )
            if src not in sources_seen:
                sources_seen.add(src)
                sources.append(src)

        context = "\n\n".join(blocks)

        if len(docs) >= 5 and len(context) > 2000:
            confidence = "HIGH"
        elif len(docs) >= 3:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        return {"context": context, "sources": sources, "confidence": confidence}

    def _node_generate(self, state: RAGState) -> dict:
        """
        Node 3 — Generate
        Send context + question to Cohere and return the grounded answer.
        """
        llm    = self._get_llm()
        prompt = self.SYSTEM_PROMPT.format(
            context=state["context"],
            question=state["question"],
        )
        t0       = time.time()
        response = llm.invoke(prompt)
        elapsed  = round(time.time() - t0, 2)

        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        return {"answer": answer, "processing_time": elapsed}

    # ── Graph assembly ────────────────────────────────────────────────────────

    def _build_graph(self):
        wf = StateGraph(RAGState)

        wf.add_node("retrieve",      self._node_retrieve)
        wf.add_node("build_context", self._node_build_context)
        wf.add_node("generate",      self._node_generate)

        wf.set_entry_point("retrieve")
        wf.add_edge("retrieve",      "build_context")
        wf.add_edge("build_context", "generate")
        wf.add_edge("generate",      END)

        return wf.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize(self, force_rebuild: bool = False) -> None:
        """Boot the full pipeline. Call once at application startup."""
        logger.info("╔══════════════════════════════════════╗")
        logger.info("║   DermIQ RAG Engine — Initializing   ║")
        logger.info("╚══════════════════════════════════════╝")
        self.build_vectorstore(force_rebuild=force_rebuild)
        self._graph    = self._build_graph()
        self._is_ready = True
        logger.info("✓ DermIQ Engine is ready for queries")

    def query(self, question: str) -> dict:
        """Run a clinical question through the full RAG pipeline."""
        if not self._is_ready or self._graph is None:
            raise RuntimeError(
                "DermIQ engine is not initialized. Call .initialize() first."
            )
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()
        initial: RAGState = {
            "question":        question,
            "retrieved_docs":  [],
            "context":         "",
            "answer":          "",
            "sources":         [],
            "confidence":      "UNKNOWN",
            "processing_time": 0.0,
        }

        final = self._graph.invoke(initial)
        return {
            "question":        question,
            "answer":          final["answer"],
            "sources":         final["sources"],
            "confidence":      final["confidence"],
            "processing_time": final["processing_time"],
            "docs_retrieved":  len(final["retrieved_docs"]),
        }

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def get_stats(self) -> dict:
        if not self._vectorstore:
            return {"status": "not_initialized"}
        return {
            "total_vectors":   self._vectorstore.index.ntotal,
            "embedding_model": self.EMBED_MODEL,
            "llm_model":       "cohere/command-r-plus",
            "status":          "ready" if self._is_ready else "initializing",
        }