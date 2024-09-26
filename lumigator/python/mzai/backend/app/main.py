import contextlib
import os
import sys

from fastapi import FastAPI
from loguru import logger
from sqlalchemy import Engine

from app.api.router import api_router
from app.api.tags import TAGS_METADATA
from app.db import engine
from app.records.base import BaseRecord


def create_app(engine: Engine) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # TODO: Remove this once switching to Alembic for migrations
        BaseRecord.metadata.create_all(engine)
        yield

    app = FastAPI(title="Lumigator Backend", lifespan=lifespan, openapi_tags=TAGS_METADATA)

    main_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:"
            "<cyan>{function}</cyan>:"
            "<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=main_log_level,
        colorize=True,
    )

    app.include_router(api_router)

    @app.get("/")
    def get_root():
        return {"Hello": "Lumigator!🐊"}

    return app


app = create_app(engine)