from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Tokens y credenciales
    BEARER_TOKEN: str
    AZURE_ENDPOINT: str
    AZURE_KEY: str

    # API Authentication - Tokens permitidos para acceder a la API
    API_TOKENS: str = ""  # Tokens separados por comas

    # Admin Token - Token especial para gestionar el mantenedor de tokens
    ADMIN_TOKEN: str = ""  # Token de administrador para endpoints de gestión

    # Database - PostgreSQL
    DB_HOST: str = "pgbouncer"
    DB_PORT: int = 5432
    DB_USER: str = "admin"
    DB_PASSWORD: str = ""
    DB_NAME: str = "pruebas"

    # Cache settings
    CACHE_ENABLED: bool = True  # Habilitar/deshabilitar caché globalmente
    
    # HTTP Timeouts
    TIMEOUT_CONNECT: float = 5.0
    TIMEOUT_READ: float = 30.0
    TIMEOUT_WRITE: float = 10.0
    
    # Azure DI
    TIMEOUT_AZURE_BASE: int = 60
    TIMEOUT_AZURE_PER_PAGE: int = 2
    TIMEOUT_AZURE_MAX: int = 600
    
    # Conversión
    TIMEOUT_EXCEL_BASE: int = 30
    TIMEOUT_EXCEL_PER_MB: int = 5
    TIMEOUT_EXCEL_MAX: int = 300
    
    # Retry
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_MIN: float = 2.0
    RETRY_BACKOFF_MAX: float = 10.0
    
    # Límites
    MAX_FILE_SIZE_MB: int = 100

    # Thread Pool - Configuración dinámica del executor
    EXECUTOR_MAX_WORKERS: int = 32  # Máximo número de workers (se calcula como min(32, cpu_count * 4))
    EXECUTOR_MIN_WORKERS: int = 4   # Mínimo número de workers como fallback

    # Timeout para procesamiento de calidad - Adaptativo según número de páginas
    TIMEOUT_QUALITY_BASE: int = 30       # Tiempo base en segundos para cualquier documento
    TIMEOUT_QUALITY_PER_PAGE: int = 5    # Segundos adicionales por cada página del documento

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def calcular_timeout_azure(num_paginas: int) -> int:
    """Calcula timeout para Azure DI basado en número de páginas."""
    settings = get_settings()
    timeout = settings.TIMEOUT_AZURE_BASE + (num_paginas * settings.TIMEOUT_AZURE_PER_PAGE)
    return min(timeout, settings.TIMEOUT_AZURE_MAX)


def calcular_timeout_excel(file_size_bytes: int) -> int:
    """Calcula timeout para conversión Excel basado en tamaño."""
    settings = get_settings()
    file_size_mb = file_size_bytes / (1024 * 1024)
    timeout = settings.TIMEOUT_EXCEL_BASE + int(file_size_mb * settings.TIMEOUT_EXCEL_PER_MB)
    return min(timeout, settings.TIMEOUT_EXCEL_MAX)


def calcular_timeout_calidad(num_paginas: int) -> int:
    """
    Calcula timeout para procesamiento de calidad basado en número de páginas.

    Formula: TIMEOUT_QUALITY_BASE + (num_paginas * TIMEOUT_QUALITY_PER_PAGE)
    """
    settings = get_settings()
    timeout = settings.TIMEOUT_QUALITY_BASE + (num_paginas * settings.TIMEOUT_QUALITY_PER_PAGE)
    return timeout


def get_valid_api_tokens() -> set:
    """Obtiene el conjunto de tokens API válidos desde la configuración."""
    settings = get_settings()
    if not settings.API_TOKENS:
        return set()
    tokens = [token.strip() for token in settings.API_TOKENS.split(',') if token.strip()]
    return set(tokens)