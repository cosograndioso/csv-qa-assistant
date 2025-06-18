# main.py

import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
import google.generativeai as genai

# Importa moduli interni
from back.qdrant_utils import salva_embedding, recupera_simili, initialize_qdrant_and_model
from back.routes import csv_routes, statistics

# Carica le variabili d'ambiente
load_dotenv()

# Configurazione
class Settings(BaseSettings):
    MONGO_URI: str
    API_KEY: str
    QDRANT_HOST: str
    QDRANT_PORT: int

    class Config:
        env_file = ".env"

settings = Settings()

if not settings.API_KEY:
    raise RuntimeError("⚠️ API_KEY mancante nel file .env")

if not settings.MONGO_URI:
    raise RuntimeError("⚠️ MONGO_URI mancante nel file .env")

# Configura l'API di Gemini
genai.configure(api_key=settings.API_KEY)

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan: avvio e chiusura risorse
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("➡️ Avvio connessione MongoDB...")
    try:
        app.state.mongo = AsyncIOMotorClient(settings.MONGO_URI)
        await app.state.mongo.admin.command('ping')

        app.state.db_csv_analytics = app.state.mongo.csv_analytics
        app.state.csv_uploads_collection = app.state.db_csv_analytics.get_collection("csv_uploads")
        app.state.query_history_collection = app.state.db_csv_analytics.get_collection("query_history")

        logger.info("✅ Connessione MongoDB stabilita ai database.")

        # Inizializzazione Qdrant + modello embedding
        logger.info("➡️ Inizializzazione Qdrant client e modello di embedding...")
        await initialize_qdrant_and_model()
        logger.info("✅ Qdrant client e modello di embedding inizializzati.")

    except ServerSelectionTimeoutError:
        logger.critical("❌ Impossibile connettersi a MongoDB. Controlla MONGO_URI.", exc_info=True)
        raise RuntimeError("Impossibile connettersi a MongoDB. Controlla MONGO_URI.")
    except Exception as e:
        logger.critical(f"❌ Errore generico durante la connessione/inizializzazione: {e}", exc_info=True)
        raise RuntimeError(f"Errore generico durante l'avvio: {e}")

    yield

    logger.info("🔌 Chiusura connessione MongoDB...")
    app.state.mongo.close()

# Istanza dell'app FastAPI
app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route incluse
app.include_router(csv_routes.router)
app.include_router(statistics.router)
