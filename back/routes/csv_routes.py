from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
import pandas as pd
import io
import datetime
import logging
import google.generativeai as genai
from pymongo.errors import ServerSelectionTimeoutError
from back.qdrant_utils import salva_embedding, recupera_simili






router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload_csv", tags=["CSV"])
async def upload_csv(request: Request, file: UploadFile = File(...)):
    logger.info(f"📅 Richiesta di upload ricevuta per il file: {file.filename}")

    if file.content_type != "text/csv":
        logger.warning(f"⚠️ Tipo di file non supportato: {file.content_type}. Richiesto text/csv.")
        raise HTTPException(status_code=400, detail="Formato file non valido. Carica un file CSV.")

    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        except UnicodeDecodeError:
            logger.warning("Tentativo di decodifica UTF-8 fallito, provando latin-1.")
            df = pd.read_csv(io.StringIO(contents.decode("latin-1")))

        records = df.to_dict("records")

        if not records:
            logger.warning("⚠️ CSV vuoto o non valido dopo la lettura.")
            raise HTTPException(status_code=400, detail="CSV vuoto o non valido")

        db = request.app.state.csv_uploads_collection
        await db.delete_many({})
        logger.info(f"🗑️ Record precedenti eliminati.")

        result = await db.insert_many(records)
        logger.info(f"✅ {len(result.inserted_ids)} righe inserite nel database.")

        return {"filename": file.filename, "rows_inserted": len(result.inserted_ids)}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Il file CSV è vuoto.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Errore durante il parsing del CSV: {e}")
    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, detail="MongoDB non raggiungibile.")
    except Exception as e:
        logger.error(f"❌ Errore durante upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query_csv", tags=["CSV"])
async def query_csv(request: Request, question: str = Form(...)):
    logger.info(f"✉️ Ricevuta query: '{question}'")

    try:
        db = request.app.state.csv_uploads_collection
        history = request.app.state.query_history_collection

        cursor = db.find({})
        rows = await cursor.to_list(length=None)

        if not rows:
            raise HTTPException(status_code=404, detail="Nessun dato CSV caricato.")

        # Blocco di lettura del database
        df = pd.DataFrame(rows)
        logger.info(f"📊 {len(df)} righe lette dal database.")

        # Calcolo statistiche
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        df = df.dropna(subset=['release_year', 'total_sales'])
        df['release_year'] = df['release_year'].astype(int)
        df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce').fillna(0)

        sales_per_year = df.groupby('release_year')['total_sales'].sum()
        total_sales = sales_per_year.sum()
        percentuali = (sales_per_year / total_sales * 100).round(2)

        stats_text = "\n".join([
            f"{anno}: {round(sales_per_year[anno], 2)} milioni, {percentuali[anno]}%"
            for anno in sales_per_year.index
        ])

        # Recupero domande simili con Qdrant
        query_context_list = await recupera_simili(question, top_k=3)
        query_context = "\n".join(query_context_list) or "Nessuna conversazione simile."

        # Costruzione del prompt
        data_for_model = df.head(50).to_string(index=False)

        prompt = f"""Sei un assistente intelligente specializzato nell'analisi di dati CSV.
Il tuo compito è rispondere alla domanda dell'utente basandoti sui dati forniti. 
Utilizza anche il contesto di domande precedenti se rilevanti.
Se l'informazione richiesta non è presente nel dataset o non può essere dedotta,
dichiara chiaramente che non puoi fornire la risposta basandoti sui dati disponibili.
Rispondi in modo conciso, diretto e professionale.

🧠 Domande simili:
{query_context}

📊 Statistiche vendite per anno:
{stats_text}

📄 Campione del dataset:
{data_for_model}

Domanda attuale: {question}

Risposta:
"""

        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
            )
        )

        if not response.parts:
            raise HTTPException(status_code=500, detail="Output del modello AI vuoto.")

        answer = response.text.strip()
        logger.info("✅ Risposta generata.")

        await history.insert_one({
            "question": question,
            "answer": answer,
            "timestamp": datetime.datetime.now(),
        })

        await salva_embedding(question, answer)

        return {"answer": answer}

    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, detail="MongoDB non raggiungibile.")
    except Exception as e:
        logger.error(f"❌ Errore durante query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
