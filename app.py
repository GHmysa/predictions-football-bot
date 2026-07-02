import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


def _warmup() -> None:
    try:
        from ml.poisson import fit_or_load
        from ml.predict import _data, _model
        fit_or_load()
        _data()
        _model()
        print("[APP] ML cache prêt")
    except Exception as e:
        print(f"[APP] Warmup warning: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_warmup)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
app.state.templates = templates

from routers import public as public_router
from routers import admin as admin_router

app.include_router(public_router.router)
app.include_router(admin_router.router)
