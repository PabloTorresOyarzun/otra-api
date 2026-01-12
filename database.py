"""
Módulo de persistencia para caché de resultados de procesamiento.
Almacena clasificaciones y procesamientos en PostgreSQL.
"""
import asyncpg
import json
import hashlib
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
from contextlib import asynccontextmanager

from config import get_settings

logger = logging.getLogger(__name__)

SCHEMA_NAME = "sgd_cache"


class DatabaseManager:
    """Gestor de conexiones y operaciones de base de datos."""

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._settings = get_settings()

    async def initialize(self):
        """Inicializa el pool de conexiones y crea el schema si no existe."""
        if self._pool is not None:
            return

        try:
            self._pool = await asyncpg.create_pool(
                host=self._settings.DB_HOST,
                port=self._settings.DB_PORT,
                user=self._settings.DB_USER,
                password=self._settings.DB_PASSWORD,
                database=self._settings.DB_NAME,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            await self._create_schema()
            logger.info("Pool de conexiones PostgreSQL inicializado")
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise

    async def close(self):
        """Cierra el pool de conexiones."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Pool de conexiones cerrado")

    @asynccontextmanager
    async def connection(self):
        """Context manager para obtener una conexión del pool."""
        if not self._pool:
            await self.initialize()
        async with self._pool.acquire() as conn:
            yield conn

    async def _create_schema(self):
        """Crea el schema y tablas si no existen."""
        async with self.connection() as conn:
            await conn.execute(f"""
                CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};
                
                CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.despachos_procesados (
                    id SERIAL PRIMARY KEY,
                    codigo_despacho VARCHAR(50) NOT NULL,
                    tipo_operacion VARCHAR(20) NOT NULL,
                    documentos_hash VARCHAR(64) NOT NULL,
                    cliente VARCHAR(255),
                    estado VARCHAR(50),
                    tipo VARCHAR(50),
                    total_documentos_segmentados INTEGER,
                    resultado JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(codigo_despacho, tipo_operacion)
                );
                
                CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.documentos_procesados (
                    id SERIAL PRIMARY KEY,
                    archivo_hash VARCHAR(64) NOT NULL,
                    nombre_archivo VARCHAR(255) NOT NULL,
                    tipo_operacion VARCHAR(20) NOT NULL,
                    total_documentos_segmentados INTEGER,
                    resultado JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(archivo_hash, tipo_operacion)
                );
                
                CREATE INDEX IF NOT EXISTS idx_despachos_codigo 
                    ON {SCHEMA_NAME}.despachos_procesados(codigo_despacho);
                CREATE INDEX IF NOT EXISTS idx_despachos_hash 
                    ON {SCHEMA_NAME}.despachos_procesados(documentos_hash);
                CREATE INDEX IF NOT EXISTS idx_documentos_hash 
                    ON {SCHEMA_NAME}.documentos_procesados(archivo_hash);
            """)
            logger.info(f"Schema {SCHEMA_NAME} verificado/creado")


def calcular_hash_documentos(documentos: List[Dict]) -> str:
    """
    Calcula un hash único basado en los documentos de un despacho.
    Permite detectar cambios en la documentación.
    """
    contenido = []
    for doc in documentos:
        if isinstance(doc, dict):
            nombre = doc.get("nombre_documento", doc.get("nombre", ""))
            doc_id = doc.get("documento_id", doc.get("id", ""))
            contenido.append(f"{nombre}:{doc_id}")
    
    contenido.sort()
    texto = "|".join(contenido)
    return hashlib.sha256(texto.encode()).hexdigest()


def calcular_hash_archivo(file_bytes: bytes) -> str:
    """Calcula el hash SHA256 de un archivo."""
    return hashlib.sha256(file_bytes).hexdigest()


class CacheRepository:
    """Repositorio para operaciones de caché."""

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    async def obtener_despacho(
        self, 
        codigo_despacho: str, 
        tipo_operacion: str,
        documentos_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene un despacho procesado del caché.
        Si se proporciona documentos_hash, verifica que coincida.
        """
        async with self._db.connection() as conn:
            if documentos_hash:
                row = await conn.fetchrow(f"""
                    SELECT * FROM {SCHEMA_NAME}.despachos_procesados
                    WHERE codigo_despacho = $1 
                    AND tipo_operacion = $2
                    AND documentos_hash = $3
                """, codigo_despacho, tipo_operacion, documentos_hash)
            else:
                row = await conn.fetchrow(f"""
                    SELECT * FROM {SCHEMA_NAME}.despachos_procesados
                    WHERE codigo_despacho = $1 AND tipo_operacion = $2
                """, codigo_despacho, tipo_operacion)
            
            if row:
                return {
                    "id": row["id"],
                    "codigo_despacho": row["codigo_despacho"],
                    "tipo_operacion": row["tipo_operacion"],
                    "documentos_hash": row["documentos_hash"],
                    "cliente": row["cliente"],
                    "estado": row["estado"],
                    "tipo": row["tipo"],
                    "total_documentos_segmentados": row["total_documentos_segmentados"],
                    "resultado": json.loads(row["resultado"]) if isinstance(row["resultado"], str) else row["resultado"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None

    async def guardar_despacho(
        self,
        codigo_despacho: str,
        tipo_operacion: str,
        documentos_hash: str,
        cliente: str,
        estado: str,
        tipo: str,
        total_documentos_segmentados: int,
        resultado: Dict[str, Any]
    ) -> int:
        """Guarda o actualiza un despacho procesado."""
        async with self._db.connection() as conn:
            row = await conn.fetchrow(f"""
                INSERT INTO {SCHEMA_NAME}.despachos_procesados 
                (codigo_despacho, tipo_operacion, documentos_hash, cliente, estado, tipo, 
                 total_documentos_segmentados, resultado)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (codigo_despacho, tipo_operacion) 
                DO UPDATE SET 
                    documentos_hash = EXCLUDED.documentos_hash,
                    cliente = EXCLUDED.cliente,
                    estado = EXCLUDED.estado,
                    tipo = EXCLUDED.tipo,
                    total_documentos_segmentados = EXCLUDED.total_documentos_segmentados,
                    resultado = EXCLUDED.resultado,
                    updated_at = NOW()
                RETURNING id
            """, codigo_despacho, tipo_operacion, documentos_hash, cliente, estado, tipo,
                total_documentos_segmentados, json.dumps(resultado, default=str))
            
            logger.info(f"Despacho {codigo_despacho} guardado en caché ({tipo_operacion})")
            return row["id"]

    async def obtener_documento(
        self,
        archivo_hash: str,
        tipo_operacion: str
    ) -> Optional[Dict[str, Any]]:
        """Obtiene un documento procesado del caché."""
        async with self._db.connection() as conn:
            row = await conn.fetchrow(f"""
                SELECT * FROM {SCHEMA_NAME}.documentos_procesados
                WHERE archivo_hash = $1 AND tipo_operacion = $2
            """, archivo_hash, tipo_operacion)
            
            if row:
                return {
                    "id": row["id"],
                    "archivo_hash": row["archivo_hash"],
                    "nombre_archivo": row["nombre_archivo"],
                    "tipo_operacion": row["tipo_operacion"],
                    "total_documentos_segmentados": row["total_documentos_segmentados"],
                    "resultado": json.loads(row["resultado"]) if isinstance(row["resultado"], str) else row["resultado"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None

    async def guardar_documento(
        self,
        archivo_hash: str,
        nombre_archivo: str,
        tipo_operacion: str,
        total_documentos_segmentados: int,
        resultado: Dict[str, Any]
    ) -> int:
        """Guarda o actualiza un documento procesado."""
        async with self._db.connection() as conn:
            row = await conn.fetchrow(f"""
                INSERT INTO {SCHEMA_NAME}.documentos_procesados 
                (archivo_hash, nombre_archivo, tipo_operacion, total_documentos_segmentados, resultado)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (archivo_hash, tipo_operacion) 
                DO UPDATE SET 
                    nombre_archivo = EXCLUDED.nombre_archivo,
                    total_documentos_segmentados = EXCLUDED.total_documentos_segmentados,
                    resultado = EXCLUDED.resultado,
                    updated_at = NOW()
                RETURNING id
            """, archivo_hash, nombre_archivo, tipo_operacion, 
                total_documentos_segmentados, json.dumps(resultado, default=str))
            
            logger.info(f"Documento {nombre_archivo} guardado en caché ({tipo_operacion})")
            return row["id"]

    async def verificar_cambios_despacho(
        self,
        codigo_despacho: str,
        tipo_operacion: str,
        nuevo_hash: str
    ) -> Dict[str, Any]:
        """
        Verifica si hay cambios en los documentos de un despacho.
        Retorna información sobre el estado del caché.
        """
        async with self._db.connection() as conn:
            row = await conn.fetchrow(f"""
                SELECT documentos_hash, updated_at 
                FROM {SCHEMA_NAME}.despachos_procesados
                WHERE codigo_despacho = $1 AND tipo_operacion = $2
            """, codigo_despacho, tipo_operacion)
            
            if not row:
                return {
                    "existe_cache": False,
                    "hay_cambios": True,
                    "hash_actual": None,
                    "hash_nuevo": nuevo_hash
                }
            
            hash_actual = row["documentos_hash"]
            hay_cambios = hash_actual != nuevo_hash
            
            return {
                "existe_cache": True,
                "hay_cambios": hay_cambios,
                "hash_actual": hash_actual,
                "hash_nuevo": nuevo_hash,
                "ultima_actualizacion": row["updated_at"]
            }

    async def eliminar_cache_despacho(
        self,
        codigo_despacho: str,
        tipo_operacion: Optional[str] = None
    ) -> int:
        """Elimina el caché de un despacho."""
        async with self._db.connection() as conn:
            if tipo_operacion:
                result = await conn.execute(f"""
                    DELETE FROM {SCHEMA_NAME}.despachos_procesados
                    WHERE codigo_despacho = $1 AND tipo_operacion = $2
                """, codigo_despacho, tipo_operacion)
            else:
                result = await conn.execute(f"""
                    DELETE FROM {SCHEMA_NAME}.despachos_procesados
                    WHERE codigo_despacho = $1
                """, codigo_despacho)
            
            count = int(result.split()[-1])
            logger.info(f"Eliminados {count} registros de caché para despacho {codigo_despacho}")
            return count


# Instancias globales
db_manager = DatabaseManager()
cache_repo = CacheRepository(db_manager)