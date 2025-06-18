# 1. Installa la libreria client
# pip install qdrant-client sentence-transformers

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from dotenv import load_dotenv

# Carica le variabili d'ambiente (qui va bene, vengono caricate una volta)
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Dichiarazioni globali per client e modello, ma li inizializzeremo in un'altra funzione
# Questo evita che vengano inizializzati all'importazione del modulo
client: QdrantClient = None
embedding_model: SentenceTransformer = None


async def initialize_qdrant_and_model():
    """
    Inizializza il client Qdrant e il modello di embedding.
    Questa funzione dovrebbe essere chiamata all'avvio dell'applicazione FastAPI.
    """
    global client, embedding_model

    if client is None:
        print(f"DEBUG: Inizializzazione client Qdrant su {QDRANT_HOST}:{QDRANT_PORT}")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        try:
            # Verifica la connessione a Qdrant
            health = client.get_collections()
            print(f"DEBUG: Connessione a Qdrant riuscita. Collezioni esistenti: {len(health.collections)}")
        except Exception as e:
            print(
                f"ERRORE CRITICO: Impossibile connettersi a Qdrant. Assicurati che il server sia in esecuzione. Errore: {e}")
            raise  # Rilancia l'eccezione per bloccare l'avvio di FastAPI se Qdrant non è disponibile

    if embedding_model is None:
        print("DEBUG: Caricamento modello di embedding...")
        # Questo può scaricare il modello la prima volta, potrebbe essere bloccante
        # In un contesto puramente async, si userebbe run_in_threadpool di Starlette
        # Per ora, lo lasciamo così, ma è il punto dove il download iniziale blocca.
        # Se Qdrant è il problema, questa riga non verrà nemmeno raggiunta.
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("DEBUG: Modello di embedding caricato.")

    # Crea la collezione (una volta sola, o ricreala se necessario)
    # È meglio controllare prima se esiste, per evitare di ricrearla ad ogni avvio in produzione
    # Per lo sviluppo, recreate_collection va bene, ma in produzione valuterei un "get_collection" e poi "create_collection"
    # Questa operazione è sincrona
    try:
        print("DEBUG: Tentativo di ricreare/verificare la collezione 'csv_embeddings'...")
        client.recreate_collection(
            collection_name="csv_embeddings",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("DEBUG: Collezione 'csv_embeddings' pronta.")
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile creare/ricreare la collezione Qdrant. Errore: {e}")
        raise  # Rilancia l'eccezione


# Le funzioni salva_embedding e recupera_simili ora useranno le variabili globali
async def salva_embedding(question: str, answer: str):
    if client is None or embedding_model is None:
        raise RuntimeError("Qdrant client o embedding model non inizializzati.")

    # Esegui l'encoding in un threadpool per non bloccare l'event loop di FastAPI
    # Questo è importante per le operazioni CPU-intensive come encode()
    # from starlette.concurrency import run_in_threadpool # Potrebbe essere necessario importarla in cima
    # embedding = await run_in_threadpool(embedding_model.encode, question)
    embedding = embedding_model.encode(question).tolist()  # Per semplicità per ora, ma considera run_in_threadpool

    payload = {"question": question, "answer": answer}
    client.upsert(
        collection_name="csv_embeddings",
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=payload
            )
        ]
    )
    print(f"DEBUG: Embedding salvato per la domanda: '{question[:30]}...'")


async def recupera_simili(question: str, top_k: int = 3):
    if client is None or embedding_model is None:
        raise RuntimeError("Qdrant client o embedding model non inizializzati.")

    # Esegui l'encoding in un threadpool
    # from starlette.concurrency import run_in_threadpool # Potrebbe essere necessario importarla in cima
    # query_vector = await run_in_threadpool(embedding_model.encode, question)
    query_vector = embedding_model.encode(question).tolist()  # Per semplicità per ora

    results = client.search(
        collection_name="csv_embeddings",
        query_vector=query_vector,
        limit=top_k
    )
    print(f"DEBUG: Recuperati {len(results)} risultati simili per la domanda: '{question[:30]}...'")
    return [f"Q: {r.payload['question']}\nA: {r.payload['answer']}" for r in results]