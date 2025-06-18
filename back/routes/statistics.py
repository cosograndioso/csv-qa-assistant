from fastapi import APIRouter, HTTPException
from fastapi import Request
import pandas as pd

router = APIRouter(tags=["Analytics"])

@router.get("/sales_stats_by_year")
async def sales_stats_by_year(request: Request):
    try:
        cursor = request.app.state.csv_uploads_collection.find({})
        rows = await cursor.to_list(length=None)
        if not rows:
            raise HTTPException(status_code=404, detail="Nessun dato disponibile.")

        df = pd.DataFrame(rows)
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        df = df.dropna(subset=['release_year', 'total_sales'])
        df['release_year'] = df['release_year'].astype(int)
        df['total_sales'] = pd.to_numeric(df['total_sales'], errors='coerce').fillna(0)

        sales_per_year = df.groupby('release_year')['total_sales'].sum()
        total = sales_per_year.sum()
        percentuali = (sales_per_year / total * 100).round(2)

        result = [
            {"year": int(anno), "sales": round(sales_per_year[anno], 2), "percentage": round(percentuali[anno], 2)}
            for anno in sales_per_year.index
        ]
        return {"stats": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Errore durante il calcolo delle statistiche.")
