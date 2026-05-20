from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "quantassist"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "quantassist"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    embedding_batch_size: int = 64
    max_concurrent_batches: int = 8

    faiss_index_path: str = "./data/faiss.index"
    top_k: int = 5
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
